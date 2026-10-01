"""
trade_plan.py
Helpers for: aligning futures/GLD levels to the TradingView price,
cleaning options levels, measuring volatility (ATR), and building a
rule-based trade plan (entry, stop loss, take profits, quality grade).

This is a rule-based framework, not a proven edge. Not financial advice.
"""

import numpy as np
import pandas as pd
import yfinance as yf


# ---------------------------------------------------------------- price alignment
def spot_from_yahoo():
    """Spot gold from Yahoo (XAUUSD=X). Can be stale: verify against TradingView."""
    try:
        h = yf.Ticker("XAUUSD=X").history(period="5d")
        if not h.empty:
            return float(h["Close"].iloc[-1])
    except Exception:
        pass
    return None


def tv_offset(tv_price, gc_price):
    """TradingView price minus futures price. Add this to any futures level."""
    if tv_price and gc_price and tv_price > 0:
        return float(tv_price - gc_price)
    return 0.0


def to_tv(gld_level, ratio, offset):
    """Convert a GLD strike to gold futures terms, then to the TradingView price."""
    return float(gld_level * ratio + offset)


# ---------------------------------------------------------------- options cleaning
def clean_levels(spot, chain):
    """
    Walls only from strikes within 10% of price that actually have open interest.
    Also reports data quality, because Yahoo sometimes returns zero OI.
    """
    band = chain[(chain.strike >= spot * 0.9) & (chain.strike <= spot * 1.1)]
    if band.empty:
        return {"ok": False, "share": 0.0, "total_oi": 0, "call_wall": None, "put_wall": None}
    share = float((band.openInterest > 0).mean())
    total = float(band.openInterest.sum())
    calls = band[band.side == "call"].groupby("strike")["openInterest"].sum()
    puts = band[band.side == "put"].groupby("strike")["openInterest"].sum()
    above = calls[(calls.index > spot) & (calls > 0)]
    below = puts[(puts.index < spot) & (puts > 0)]
    return {
        "ok": bool(total >= 5000 and share >= 0.3),
        "share": share,
        "total_oi": total,
        "call_wall": float(above.idxmax()) if not above.empty else None,
        "put_wall": float(below.idxmax()) if not below.empty else None,
    }


# ---------------------------------------------------------------- volatility
def get_atr(symbol="GC=F", length=14):
    """Average true range. Hourly first, then daily. Returns (atr, label) or (None, None)."""
    for period, interval, label in (("1mo", "1h", "1H"), ("3mo", "1d", "daily")):
        try:
            h = yf.Ticker(symbol).history(period=period, interval=interval)
            if len(h) > length + 1:
                tr = pd.concat([
                    h["High"] - h["Low"],
                    (h["High"] - h["Close"].shift()).abs(),
                    (h["Low"] - h["Close"].shift()).abs(),
                ], axis=1).max(axis=1)
                return float(tr.rolling(length).mean().iloc[-1]), label
        except Exception:
            continue
    return None, None


# ---------------------------------------------------------------- the plan
def build_plan(price, total, comps, atr, levels, gamma_pos, opts_ok, cot_pct=None):
    """
    price     current price in TradingView terms
    total     overall bias score (-1..+1)
    comps     dict name -> score, used to check agreement
    atr       volatility in price points
    levels    list of (name, tv_price)
    gamma_pos True/False/None
    """
    if total >= 0.15:
        side, d = "LONG", 1
    elif total <= -0.15:
        side, d = "SHORT", -1
    else:
        return {"side": "WAIT", "reasons": ["Signals are mixed or weak (bias near zero)."],
                "warnings": ["No edge right now. Wait for a clearer reading or a level."]}

    reasons, warnings = [], []

    # Entry: nearest level on the favourable side within 1.5 ATR, else market
    cands = [(n, p) for n, p in levels
             if (p <= price if d == 1 else p >= price) and abs(price - p) <= 1.5 * atr]
    if cands:
        name, lvl = min(cands, key=lambda x: abs(price - x[1]))
        entry, entry_type, at_level = lvl, f"Limit at {name}", True
    else:
        entry, entry_type, at_level = price, "Market (no key level nearby)", False

    sl = entry - d * 1.0 * atr
    risk = abs(entry - sl)
    tp1 = entry + d * 1.5 * risk

    opp = sorted([p for n, p in levels if (p > entry if d == 1 else p < entry)],
                 key=lambda p: abs(p - entry))
    blocker = opp[0] if opp else None
    rr_blocker = abs(blocker - entry) / risk if blocker else None
    tp2_level = next((p for p in opp if 1.5 * risk < abs(p - entry) <= 4 * risk), None)
    tp2 = tp2_level if tp2_level is not None else entry + d * 2.5 * risk

    # Quality score
    score = min(abs(total) / 0.5, 1) * 30
    nz = [v for v in comps.values() if abs(v) > 1e-9]
    agree = sum(1 for v in nz if np.sign(v) == d) / len(nz) if nz else 0
    score += agree * 25
    score += 20 if at_level else 5
    score += 15 if blocker is None else min(rr_blocker / 1.5, 1) * 15
    if opts_ok and gamma_pos is not None:
        fits = (gamma_pos and at_level) or ((not gamma_pos) and not at_level)
        score += 10 if fits else 4
    else:
        score += 5

    if cot_pct is not None and ((d == 1 and cot_pct >= 90) or (d == -1 and cot_pct <= 10)):
        score -= 15
        warnings.append("Funds are already crowded in this direction (COT). Reversal risk.")
    if rr_blocker is not None and rr_blocker < 1.5:
        warnings.append(f"A key level sits only {rr_blocker:.1f}R away and may block TP1.")
    if not opts_ok:
        warnings.append("Options data is poor right now, so the plan uses COT and ATR only.")
    if not at_level:
        warnings.append("Entry is not at a key level. Prefer waiting for a pullback to one.")

    score = int(max(0, min(100, round(score))))
    grade = "GOOD" if score >= 70 else "MID" if score >= 45 else "WEAK (skip)"

    reasons.append(f"Bias {total:+.2f}; {int(round(agree * 100))}% of signals agree.")
    if at_level:
        reasons.append(f"Entry sits at a key level ({entry_type.replace('Limit at ', '')}).")
    if gamma_pos is not None and opts_ok:
        reasons.append("Positive gamma: expect ranges, fading levels works better." if gamma_pos
                       else "Negative gamma: expect fast moves, breakouts work better.")

    return {
        "side": side, "entry": entry, "entry_type": entry_type, "sl": sl,
        "tp1": tp1, "tp2": tp2, "rr1": 1.5, "rr2": abs(tp2 - entry) / risk,
        "risk_pts": risk, "score": score, "grade": grade,
        "reasons": reasons, "warnings": warnings,
    }
