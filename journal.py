"""
journal.py
Locks each trade signal (entry, stop, targets) and tracks it with 1-minute candles:
entry filled? stop hit? TP1/TP2 hit? Keeps a history with stats.

Management rule used for results: half off at TP1, stop moves to entry after TP1,
the rest runs to TP2. If a stop and a target are touched in the same 1-minute
candle, the stop is assumed first (pessimistic).

Signals are stored in signals.json inside the repo. The background monitor
(monitor.py) writes it; the app only reads and displays it.
"""

import copy
import json
import os
import tempfile

import pandas as pd

ACTIVE = ("PENDING", "OPEN", "TP1_HIT")
FILLED_RESULTS = ("WON_TP2", "LOST_SL", "TP1_THEN_BE", "TIME_EXIT", "MANUAL_EXIT")
PENDING_HOURS = 6      # cancel a limit order that has not filled after this long
OPEN_HOURS = 10        # intraday only: force an exit after this long
COOLDOWN_MIN = 30      # wait after a signal closes before creating the next one
STATE_KEYS = ("status", "filled_at", "tp1_at", "closed_at", "close_price", "r")


def _path():
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "signals.json")
    try:
        with open(p, "a"):
            pass
        return p
    except Exception:
        return os.path.join(tempfile.gettempdir(), "signals.json")


def load():
    try:
        with open(_path()) as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def save(trades):
    p = _path()
    tmp = p + ".tmp"
    with open(tmp, "w") as f:
        json.dump(trades, f, default=float, indent=1)
    os.replace(tmp, p)


def active(trades):
    return next((t for t in trades if t["status"] in ACTIVE), None)


def can_create(trades, now):
    if active(trades):
        return False
    closed = [pd.Timestamp(t["closed_at"]) for t in trades if t.get("closed_at")]
    if closed and pd.Timestamp(now) - max(closed) < pd.Timedelta(minutes=COOLDOWN_MIN):
        return False
    return True


def new_trade(plan, offset, price, now):
    iso = now.isoformat()
    return {
        "id": iso, "created": iso, "checked_until": iso, "notified": 0,
        "side": plan["side"], "setup": plan["setup"], "grade": plan["grade"],
        "score": int(plan["score"]), "entry": float(plan["entry"]),
        "entry_type": plan["entry_type"], "sl": float(plan["sl"]),
        "tp1": float(plan["tp1"]), "tp2": float(plan["tp2"]),
        "rr1": float(plan["rr1"]), "rr2": float(plan["rr2"]),
        "risk_pts": float(plan["risk_pts"]), "offset": float(offset),
        "price_at_signal": float(price), "status": "PENDING",
        "filled_at": None, "tp1_at": None, "closed_at": None,
        "close_price": None, "r": None,
        "reasons": list(plan.get("reasons", [])), "warnings": list(plan.get("warnings", [])),
    }


def close(t, status, price, r, ts=None):
    t["status"] = status
    t["closed_at"] = (ts or pd.Timestamp.now(tz="UTC")).isoformat()
    t["close_price"] = None if price is None else float(price)
    t["r"] = float(r)


def evaluate(t, bars):
    """Walk 1-minute candles since the last check. Mutates t. Returns (t, events)."""
    events = []
    if t["status"] not in ACTIVE or bars is None or bars.empty:
        return t, events
    d = 1 if t["side"] == "LONG" else -1
    off = t["offset"]
    entry, sl, tp1, tp2 = t["entry"], t["sl"], t["tp1"], t["tp2"]
    rr1, rr2, risk = t["rr1"], t["rr2"], t["risk_pts"]

    b = bars.copy()
    b.index = b.index.tz_localize("UTC") if b.index.tz is None else b.index.tz_convert("UTC")
    b = b[b.index > pd.Timestamp(t.get("checked_until") or t["created"])]

    for ts, row in b.iterrows():
        hi, lo, cl = row["High"] + off, row["Low"] + off, row["Close"] + off
        adv = lo if d == 1 else hi      # worst price for the trade in this candle
        fav = hi if d == 1 else lo      # best price for the trade in this candle
        t["checked_until"] = ts.isoformat()
        status = t["status"]

        if status == "PENDING":
            if ts - pd.Timestamp(t["created"]) > pd.Timedelta(hours=PENDING_HOURS):
                close(t, "EXPIRED", None, 0.0, ts)
                events.append("EXPIRED: entry never filled")
                break
            if d * (adv - entry) <= 0:
                t["status"] = "OPEN"
                t["filled_at"] = ts.isoformat()
                events.append("ENTRY FILLED")
                if d * (adv - sl) <= 0:
                    close(t, "LOST_SL", sl, -1.0, ts)
                    events.append("STOP LOSS HIT")
                    break
                continue
            if d * (fav - tp1) >= 0:
                close(t, "MISSED", None, 0.0, ts)
                events.append("MISSED: price ran to TP1 without filling the entry")
                break
            continue

        stop = sl if status == "OPEN" else entry
        if d * (adv - stop) <= 0:
            if status == "OPEN":
                close(t, "LOST_SL", sl, -1.0, ts)
                events.append("STOP LOSS HIT")
            else:
                close(t, "TP1_THEN_BE", entry, 0.5 * rr1, ts)
                events.append("STOPPED AT ENTRY after TP1")
            break
        if d * (fav - tp2) >= 0:
            close(t, "WON_TP2", tp2, 0.5 * rr1 + 0.5 * rr2, ts)
            events.append("TP2 HIT")
            break
        if status == "OPEN" and d * (fav - tp1) >= 0:
            t["status"] = "TP1_HIT"
            t["tp1_at"] = ts.isoformat()
            events.append("TP1 HIT: stop moved to entry")
        if t.get("filled_at") and ts - pd.Timestamp(t["filled_at"]) > pd.Timedelta(hours=OPEN_HOURS):
            run = d * (cl - entry) / risk
            r = run if t["status"] == "OPEN" else 0.5 * rr1 + 0.5 * run
            close(t, "TIME_EXIT", cl, r, ts)
            events.append("TIME EXIT (intraday limit reached)")
            break
    return t, events


def replay(t, bars):
    """
    Re-run a trade from its creation time over the candles. Deterministic, so it can be
    run again and again without double counting. Does NOT change t.
    Returns (updated_copy, all_events_since_creation).
    """
    c = copy.deepcopy(t)
    c.update({"status": "PENDING", "filled_at": None, "tp1_at": None, "closed_at": None,
              "close_price": None, "r": None, "checked_until": c["created"]})
    return evaluate(c, bars)


def history_df(trades):
    rows = []
    for t in sorted(trades, key=lambda x: x["created"], reverse=True):
        if t["status"] in ACTIVE:
            continue
        rows.append({
            "Created (UTC)": pd.Timestamp(t["created"]).strftime("%m-%d %H:%M"),
            "Side": t["side"], "Setup": t["setup"], "Grade": t["grade"],
            "Entry": round(t["entry"], 1), "SL": round(t["sl"], 1),
            "TP1": round(t["tp1"], 1), "TP2": round(t["tp2"], 1),
            "Result": t["status"], "R": None if t["r"] is None else round(t["r"], 2),
        })
    return pd.DataFrame(rows)


def stats(trades):
    f = [t for t in trades if t["status"] in FILLED_RESULTS and t["r"] is not None]
    n = len(f)
    if n == 0:
        return None
    wins = sum(1 for t in f if t["r"] > 0)
    total = sum(t["r"] for t in f)
    return {"trades": n, "wins": wins, "win_rate": wins / n * 100,
            "total_r": total, "avg_r": total / n}
