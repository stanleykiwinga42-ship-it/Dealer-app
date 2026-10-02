"""
monitor.py
Runs in the background on GitHub Actions (about every 5 minutes), even when the app is
closed. It looks for a gold setup, locks it into signals.json, tracks it with 1-minute
candles, and sends phone notifications through ntfy.sh.

Settings come from environment variables (set in the workflow file):
  NTFY_TOPIC  your private ntfy topic (stored as a GitHub secret)
  MIN_SCORE   minimum grade score to create a signal (default 60)
  OFFSET      optional fixed TradingView-minus-futures offset (default: auto)
  APP_URL     optional link opened when you tap the notification
"""

import os
from datetime import datetime, timezone

import numpy as np
import requests

import cboe
import dealer_positioning as dp
import intraday as idy
import journal as jr
import trade_plan as tp

TOPIC = os.environ.get("NTFY_TOPIC", "").strip()
MIN_SCORE = int(os.environ.get("MIN_SCORE", "60") or 60)
OFFSET_FIXED = os.environ.get("OFFSET", "").strip()
APP_URL = os.environ.get("APP_URL", "").strip()
DAYS = 30


def notify(title, body, priority="default"):
    print(f"[notify:{priority}] {title} | {body}")
    if not TOPIC:
        print("NTFY_TOPIC not set: notification not sent")
        return
    headers = {"Title": title, "Priority": priority}
    if APP_URL:
        headers["Click"] = APP_URL
    try:
        requests.post(f"https://ntfy.sh/{TOPIC}", data=body.encode("utf-8"),
                      headers=headers, timeout=15)
    except Exception as e:
        print("notify failed:", e)


def levels_text(t):
    return (f"{t['side']} gold\nEntry {t['entry']:,.2f}\nSL {t['sl']:,.2f}\n"
            f"TP1 {t['tp1']:,.2f} | TP2 {t['tp2']:,.2f}")


def analyze(now):
    gc = idy.live_price("GC=F")
    if gc is None:
        raise RuntimeError("could not get the gold price")
    if OFFSET_FIXED:
        offset = float(OFFSET_FIXED)
    else:
        auto = idy.live_price("XAUUSD=X")
        offset = (auto - gc) if auto else 0.0
        if abs(offset) > 150:
            print(f"offset {offset:.1f} looks wrong, using 0")
            offset = 0.0
    price = gc + offset

    parts, cot = [], None
    try:
        cot = dp.analyze_cot(dp.fetch_cot(dp.COT_CODES["GOLD"]))
        parts.append(("COT", cot["score"], 0.4))
    except Exception as e:
        print("COT failed:", e)

    op = lv = exp = chain = spot = ratio = None
    opts_ok = False
    try:
        try:
            spot, chain = cboe.fetch_chain("GLD", DAYS)
        except Exception as e:
            print("Cboe failed, using Yahoo:", e)
            spot, chain = dp.fetch_gld_chain(DAYS)
        op = dp.analyze_options(spot, chain)
        lv = tp.clean_levels(spot, chain)
        ratio = gc / spot
        opts_ok = bool(lv["ok"])
        if opts_ok:
            exp = idy.expiry_info(chain, spot)
    except Exception as e:
        print("Options failed:", e)

    if opts_ok:
        parts.append(("Flow", float(np.clip(1.0 - op["pc_vol"], -1, 1)) * 0.5, 0.2))
        if op["skew"] is not None:
            parts.append(("Skew", float(np.clip(-op["skew"] * 5, -1, 1)), 0.2))
        if op["flip"]:
            parts.append(("Vs gamma flip", 0.3 if spot > op["flip"] else -0.3, 0.2))
    total = sum(s * w for _, s, w in parts) / sum(w for _, _, w in parts) if parts else 0.0

    levels = []
    if opts_ok:
        if lv["put_wall"]:
            levels.append(("GLD put wall", tp.to_tv(lv["put_wall"], ratio, offset)))
        if lv["call_wall"]:
            levels.append(("GLD call wall", tp.to_tv(lv["call_wall"], ratio, offset)))
        if op["flip"]:
            levels.append(("Gamma flip", tp.to_tv(op["flip"], ratio, offset)))
    pin = tp.to_tv(exp["max_pain"], ratio, offset) if (exp and exp["max_pain"]) else None

    trend = idy.trend_snapshot(idy.get_bars("GC=F", "15m", "5d"),
                               idy.get_bars("GC=F", "1h", "1mo"))
    atr = trend["atr15"] or price * 0.0025
    plan = idy.build_plan(price, total, atr, levels, trend,
                          (op["net_gex"] > 0) if opts_ok else None,
                          idy.session_info(now), pin,
                          exp["dte"] if exp else None, cot["pct"] if cot else None)
    return plan, price, offset


def main():
    now = datetime.now(timezone.utc)
    trades = jr.load()
    changed = False

    # 1) track the active signal with 1-minute candles
    t = jr.active(trades)
    if t is not None:
        bars = idy.get_bars("GC=F", "1m", "5d")
        if bars is not None:
            c, events = jr.replay(t, bars)
            new = events[int(t.get("notified", 0)):]
            for e in new:
                extra = ""
                if c["status"] not in jr.ACTIVE and c.get("r") is not None and e == events[-1]:
                    extra = f"\nResult: {c['r']:+.2f}R"
                notify(f"Gold: {e}", levels_text(t) + extra, "high")
            if new or any(t.get(k) != c[k] for k in jr.STATE_KEYS):
                for k in jr.STATE_KEYS:
                    t[k] = c[k]
                t["notified"] = len(events)
                changed = True
        else:
            print("no 1-minute bars; cannot track the signal this run")
        t = jr.active(trades)

    # 2) look for a new setup
    if t is None and jr.can_create(trades, now) and not idy.session_info(now)["closed"]:
        try:
            plan, price, offset = analyze(now)
        except Exception as e:
            print("analysis failed:", e)
            plan = None
        if plan and plan["side"] != "WAIT" and plan["score"] >= MIN_SCORE:
            nt = jr.new_trade(plan, offset, price, now)
            trades.append(nt)
            changed = True
            body = (f"{plan['setup']}\nEntry {nt['entry']:,.2f} ({nt['entry_type']})\n"
                    f"SL {nt['sl']:,.2f} (risk {nt['risk_pts']:.1f} pts)\n"
                    f"TP1 {nt['tp1']:,.2f} | TP2 {nt['tp2']:,.2f}\n"
                    f"{plan['expect']}\nTradingView prices (offset {offset:+.1f}).")
            notify(f"NEW {plan['side']} SETUP: Gold, grade {plan['grade']} "
                   f"({plan['score']}/100)", body, "urgent")
        elif plan:
            print(f"no qualifying setup: {plan['side']} score {plan.get('score')}")

    if changed:
        jr.save(trades)
        print("signals.json updated")
    else:
        print("no changes")


if __name__ == "__main__":
    main()
