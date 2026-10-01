"""
intraday.py
Intraday (short-term) tools: live price, trend/RSI/ATR on 15m and 1h bars,
trading sessions, option expiry and max pain, CME OI paste parser, and an
intraday trade-plan builder with a confluence grade.

Rule-based framework, not a proven edge. Not financial advice.
"""

import re
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf


# ---------------------------------------------------------------- prices and bars
def live_price(symbol="GC=F"):
    """Latest 1-minute close (Yahoo can lag 1-2 minutes). Falls back to daily close."""
    try:
        h = yf.Ticker(symbol).history(period="1d", interval="1m")
        if not h.empty:
            return float(h["Close"].iloc[-1])
        h = yf.Ticker(symbol).history(period="5d")
        return float(h["Close"].iloc[-1])
    except Exception:
        return None


def get_bars(symbol, interval, period):
    try:
        h = yf.Ticker(symbol).history(period=period, interval=interval).dropna(subset=["Close"])
        return h if len(h) > 60 else None
    except Exception:
        return None


def _atr(h, n=14):
    tr = pd.concat([h["High"] - h["Low"],
                    (h["High"] - h["Close"].shift()).abs(),
                    (h["Low"] - h["Close"].shift()).abs()], axis=1).max(axis=1)
    return float(tr.rolling(n).mean().iloc[-1])


def _rsi(c, n=14):
    d = c.diff()
    up = d.clip(lower=0).rolling(n).mean()
    dn = (-d.clip(upper=0)).rolling(n).mean()
    rs = up / dn.replace(0, np.nan)
    v = (100 - 100 / (1 + rs)).iloc[-1]
    return None if pd.isna(v) else float(v)


def _dir(c):
    e20, e50 = c.ewm(span=20).mean().iloc[-1], c.ewm(span=50).mean().iloc[-1]
    last = c.iloc[-1]
    if last > e20 > e50:
        return 1
    if last < e20 < e50:
        return -1
    return 0


def trend_snapshot(m15, h1):
    out = {"atr15": None, "rsi15": None, "dir15": 0, "dir1h": 0}
    if m15 is not None:
        out["atr15"] = _atr(m15)
        out["rsi15"] = _rsi(m15["Close"])
        out["dir15"] = _dir(m15["Close"])
    if h1 is not None:
        out["dir1h"] = _dir(h1["Close"])
    return out


# ---------------------------------------------------------------- sessions
def session_info(now=None):
    """Approximate gold sessions in UTC. Score 0-10 for liquidity/volatility quality."""
    now = now or datetime.now(timezone.utc)
    wd, h = now.weekday(), now.hour
    closed = (wd == 5) or (wd == 4 and h >= 21) or (wd == 6 and h < 22)
    if closed:
        return {"name": "Market closed", "score": 0, "closed": True}
    if 12 <= h < 16:
        return {"name": "London/New York overlap (best liquidity)", "score": 10, "closed": False}
    if 7 <= h < 12:
        return {"name": "London session", "score": 7, "closed": False}
    if 16 <= h < 21:
        return {"name": "New York session", "score": 7, "closed": False}
    return {"name": "Asia / quiet hours (thin, choppy)", "score": 3, "closed": False}


# ---------------------------------------------------------------- expiry and max pain
def max_pain(sub):
    if sub is None or sub.empty:
        return None
    c = sub[sub.side == "call"]
    p = sub[sub.side == "put"]
    best, best_v = None, None
    for k in np.sort(sub.strike.unique()):
        v = (np.maximum(k - c.strike.values, 0) * c.openInterest.values).sum() + \
            (np.maximum(p.strike.values - k, 0) * p.openInterest.values).sum()
        if best_v is None or v < best_v:
            best, best_v = float(k), v
    return best


def expiry_info(chain, spot):
    """Next option expiry, days left, and the max-pain strike (GLD terms)."""
    if chain is None or chain.empty:
        return None
    today = pd.Timestamp.today().normalize()
    nxt = sorted(chain.expiry.unique())[0]
    sub = chain[(chain.expiry == nxt) & (chain.openInterest > 0)
                & (chain.strike >= spot * 0.9) & (chain.strike <= spot * 1.1)]
    return {"date": pd.Timestamp(nxt).date(), "dte": int((pd.Timestamp(nxt) - today).days),
            "max_pain": max_pain(sub)}


# ---------------------------------------------------------------- CME paste parser
def parse_oi_table(text):
    """Lines of: strike call_oi put_oi (separated by commas, spaces or tabs, plain numbers)."""
    rows = []
    for line in (text or "").splitlines():
        parts = [x for x in re.split(r"[,\t; ]+", line.strip()) if x]
        if len(parts) >= 3:
            try:
                rows.append((float(parts[0]), float(parts[1]), float(parts[2])))
            except ValueError:
                continue
    return pd.DataFrame(rows, columns=["strike", "call_oi", "put_oi"]) if rows else None


def cme_levels(df, fut_price):
    """Walls and max pain from a pasted CME table (futures price terms)."""
    above = df[(df.strike > fut_price) & (df.call_oi > 0)]
    below = df[(df.strike < fut_price) & (df.put_oi > 0)]
    out = {"call_wall": None, "put_wall": None, "max_pain": None}
    if not above.empty:
        out["call_wall"] = float(above.loc[above.call_oi.idxmax(), "strike"])
    if not below.empty:
        out["put_wall"] = float(below.loc[below.put_oi.idxmax(), "strike"])
    long = pd.concat([
        pd.DataFrame({"strike": df.strike, "side": "call", "openInterest": df.call_oi}),
        pd.DataFrame({"strike": df.strike, "side": "put", "openInterest": df.put_oi}),
    ])
    out["max_pain"] = max_pain(long[long.openInterest > 0])
    return out


# ---------------------------------------------------------------- the plan
def build_plan(price, total, atr, levels, trend, gamma_pos, session, pin, dte, cot_pct):
    """
    price      current price (TradingView terms)
    total      bias score -1..+1 (COT + options)
    atr        15m ATR in points
    levels     list of (name, tv_price)
    trend      dict from trend_snapshot
    gamma_pos  True / False / None (unknown)
    pin        max-pain price in TradingView terms, or None
    dte        days to the next option expiry, or None
    """
    if session["closed"]:
        return {"side": "WAIT", "reasons": ["Market is closed."], "warnings": []}

    near = lambda p: abs(price - p)
    res = sorted([(n, p) for n, p in levels if p >= price and near(p) <= 0.7 * atr],
                 key=lambda x: near(x[1]))
    sup = sorted([(n, p) for n, p in levels if p <= price and near(p) <= 0.7 * atr],
                 key=lambda x: near(x[1]))
    t15, t1h = trend["dir15"], trend["dir1h"]

    d, setup, anchor = 0, "", None
    if gamma_pos is not False and (res or sup):
        if res and (not sup or near(res[0][1]) <= near(sup[0][1])):
            d, setup, anchor = -1, f"Fade resistance ({res[0][0]})", res[0][1]
        else:
            d, setup, anchor = 1, f"Fade support ({sup[0][0]})", sup[0][1]
    elif gamma_pos is False and t15 != 0 and t15 == t1h:
        d, setup = t15, "Trend continuation (negative gamma)"
    elif abs(total) >= 0.25 and t15 != 0 and t15 == int(np.sign(total)):
        d, setup = t15, "Bias with trend"

    if d == 0:
        return {"side": "WAIT",
                "reasons": ["No setup: price is not at a key level and trend/bias do not line up."],
                "warnings": ["Wait for price to reach a level, or for trend and bias to agree."]}

    if anchor is not None:
        entry, entry_type, sl = anchor, "Limit at level", anchor - d * 0.8 * atr
    else:
        entry, entry_type = price - d * 0.25 * atr, "Limit pullback"
        sl = entry - d * 1.0 * atr
    risk = abs(entry - sl)

    opp = sorted([p for n, p in levels if (p > entry + 0.2 * risk if d == 1 else p < entry - 0.2 * risk)],
                 key=lambda p: abs(p - entry))
    blocker = opp[0] if opp else None
    rr_b = abs(blocker - entry) / risk if blocker is not None else None
    tp1, tp2 = entry + d * risk, entry + d * 2 * risk
    if blocker is not None and rr_b < 2.0:
        tp2 = blocker
        if rr_b < 1.0:
            tp1 = blocker

    # ---- confluence score
    s, notes, warns = 0.0, [], []
    if int(np.sign(total)) == d:
        s += min(abs(total) / 0.4, 1) * 20
        notes.append("Dealer/COT bias agrees with the trade.")
    elif abs(total) >= 0.15:
        s -= 15
        warns.append("Dealer/COT bias points the other way.")
    s += 20 if anchor is not None else 12
    agree = (t15 == d) + (t1h == d)
    against = (t15 == -d) + (t1h == -d)
    s += float(np.clip(7.5 * agree - 5 * against, -10, 15))
    if agree == 2:
        notes.append("15m and 1h trend both agree.")
    if against:
        warns.append("Trend on the 15m/1h chart is against this trade.")
    r = trend["rsi15"]
    if r is not None:
        if d == 1:
            s += 5 if r <= 65 else (-5 if r >= 75 else 0)
        else:
            s += 5 if r >= 35 else (-5 if r <= 25 else 0)
        if (d == 1 and r >= 75) or (d == -1 and r <= 25):
            warns.append("Price already stretched (RSI): chasing risk.")
    s += session["score"]
    if gamma_pos is True and anchor is not None:
        s += 10
        notes.append("Positive gamma: fading levels fits.")
    elif gamma_pos is False and anchor is None:
        s += 10
        notes.append("Negative gamma: moves extend, trend trade fits.")
    elif gamma_pos is None:
        s += 4
    if pin is not None and dte is not None and dte <= 1:
        warns.append(f"Options expire in {dte} day(s): pinning and erratic moves near the "
                     f"max-pain level are common.")
        if abs(pin - entry) <= 3 * atr:
            if (pin - entry) * d > 0:
                s += 10
                notes.append("Max-pain magnet sits in the direction of the trade.")
            else:
                s -= 5
                warns.append("Max-pain magnet sits against the trade.")
    if rr_b is None or rr_b >= 1.5:
        s += 10
    elif rr_b >= 1.0:
        s += 5
    else:
        s -= 10
        warns.append(f"Next level is only {rr_b:.1f}R away and caps the profit.")
    if cot_pct is not None and ((d == 1 and cot_pct >= 90) or (d == -1 and cot_pct <= 10)):
        s -= 10
        warns.append("Funds are crowded in this direction (COT): reversal risk.")

    score = int(max(0, min(100, round(s))))
    if score >= 70:
        grade, expect = "A (strong)", "Good entry. Take it if order flow confirms at the level."
    elif score >= 50:
        grade, expect = "B (mid)", "Mid entry. Only with clear order-flow confirmation, and use smaller size."
    else:
        grade, expect = "C (weak)", "Worse entry. Skip it."

    return {
        "side": "LONG" if d == 1 else "SHORT", "setup": setup, "entry": entry,
        "entry_type": entry_type, "sl": sl, "tp1": tp1, "tp2": tp2,
        "rr1": abs(tp1 - entry) / risk, "rr2": abs(tp2 - entry) / risk, "risk_pts": risk,
        "score": score, "grade": grade, "expect": expect, "reasons": notes, "warnings": warns,
        "time_stop": "Intraday only: close by the end of this session. If price has not moved "
                     "0.5R in about 8 candles (15m), cut or reduce.",
    }
