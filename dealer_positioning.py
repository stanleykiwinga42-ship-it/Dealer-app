#!/usr/bin/env python3
"""
Dealer / institutional positioning analyzer for XAUUSD and GBPJPY.

FREE DATA THIS SCRIPT FETCHES
  1. CFTC COT (Disaggregated, futures only): gold, GBP, JPY   -> weekly bias
  2. GLD options chain (yfinance)                             -> OI walls, GEX, gamma flip,
                                                                 skew, unusual volume
  3. Gold futures price (GC=F) to convert GLD levels to gold price

NOT AVAILABLE AS FREE APIs (no legal way to auto-fetch):
  CME QuikStrike, CME Vol2Vol, SpotGamma, Menthor Q, Bookmap.
  Read those by eye and pass the key levels with --levels (gold price).

INSTALL:  pip install requests pandas numpy yfinance
RUN:      python dealer_positioning.py
          python dealer_positioning.py --levels 3350,3400,3450 --days 45

This is decision support, not financial advice. GEX here assumes customers
are long calls / short puts, which is often wrong in gold. Treat as a proxy.
"""

import argparse
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import yfinance as yf

warnings.filterwarnings("ignore")

CFTC_URL = "https://publicreporting.cftc.gov/resource/72hh-3qpy.json"
COT_CODES = {"GOLD": "088691", "GBP": "096742", "JPY": "097741"}
RISK_FREE = 0.04


# ----------------------------------------------------------------------------
# 1. COT
# ----------------------------------------------------------------------------
def fetch_cot(code: str, weeks: int = 156) -> pd.DataFrame:
    params = {
        "$where": f"cftc_contract_market_code='{code}'",
        "$order": "report_date_as_yyyy_mm_dd DESC",
        "$limit": weeks,
    }
    r = requests.get(CFTC_URL, params=params, timeout=30)
    r.raise_for_status()
    df = pd.DataFrame(r.json())
    if df.empty:
        raise ValueError(f"No COT data for code {code}")

    def col(*names):
        for n in names:
            if n in df.columns:
                return pd.to_numeric(df[n], errors="coerce")
        raise KeyError(f"None of {names} in COT columns")

    out = pd.DataFrame({
        "date": pd.to_datetime(df["report_date_as_yyyy_mm_dd"]),
        "mm_long": col("m_money_positions_long_all"),
        "mm_short": col("m_money_positions_short_all"),
        "swap_long": col("swap_positions_long_all"),
        "swap_short": col("swap__positions_short_all", "swap_positions_short_all"),
        "oi": col("open_interest_all"),
    })
    out["mm_net"] = out["mm_long"] - out["mm_short"]
    out["swap_net"] = out["swap_long"] - out["swap_short"]
    return out.sort_values("date").reset_index(drop=True)


def analyze_cot(df: pd.DataFrame) -> dict:
    last = df.iloc[-1]
    pct = float((df["mm_net"] <= last["mm_net"]).mean() * 100)
    chg4 = float(last["mm_net"] - df["mm_net"].iloc[-5]) if len(df) > 5 else 0.0
    trend = int(np.sign(chg4))

    if pct >= 90:
        state, score = "CROWDED LONG (reversal risk)", -0.5
    elif pct <= 10:
        state, score = "CROWDED SHORT (squeeze risk)", 0.5
    else:
        state, score = ("Funds adding longs" if trend > 0 else
                        "Funds adding shorts" if trend < 0 else "Flat"), 0.5 * trend
    return {
        "date": last["date"].date(),
        "mm_net": int(last["mm_net"]),
        "pct": pct,
        "chg4": int(chg4),
        "trend": trend,
        "swap_net": int(last["swap_net"]),
        "state": state,
        "score": score,
    }


# ----------------------------------------------------------------------------
# 2. GLD options
# ----------------------------------------------------------------------------
def bs_gamma(S, K, T, sigma):
    """Black-Scholes gamma. S can be scalar or array broadcastable with K."""
    sqrtT = np.sqrt(T)
    d1 = (np.log(S / K) + (RISK_FREE + 0.5 * sigma ** 2) * T) / (sigma * sqrtT)
    return np.exp(-0.5 * d1 ** 2) / np.sqrt(2 * np.pi) / (S * sigma * sqrtT)


def fetch_gld_chain(max_days: int):
    t = yf.Ticker("GLD")
    spot = float(t.history(period="5d")["Close"].iloc[-1])
    today = pd.Timestamp.today().normalize()
    frames = []
    for exp in t.options:
        exp_dt = pd.Timestamp(exp)
        dte = (exp_dt - today).days
        if dte < 0 or dte > max_days:
            continue
        ch = t.option_chain(exp)
        for side, d in (("call", ch.calls), ("put", ch.puts)):
            d = d.copy()
            d["side"] = side
            d["expiry"] = exp_dt
            d["dte"] = max(dte, 1)
            frames.append(d)
    if not frames:
        raise ValueError("No GLD option expiries in range")
    df = pd.concat(frames, ignore_index=True)
    for c in ("openInterest", "volume", "impliedVolatility", "strike"):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    return spot, df


def analyze_options(spot: float, df: pd.DataFrame) -> dict:
    calls = df[df.side == "call"]
    puts = df[df.side == "put"]

    # --- OI walls
    call_oi = calls.groupby("strike")["openInterest"].sum()
    put_oi = puts.groupby("strike")["openInterest"].sum()
    above = call_oi[call_oi.index > spot]
    below = put_oi[put_oi.index < spot]
    call_wall = float(above.idxmax()) if not above.empty else None
    put_wall = float(below.idxmax()) if not below.empty else None

    # --- GEX and gamma flip
    g = df[(df.impliedVolatility > 0.01) & (df.openInterest > 0)].copy()
    sign = np.where(g.side == "call", 1.0, -1.0)
    K = g.strike.values[None, :]
    T = (g.dte.values / 365.0)[None, :]
    sig = g.impliedVolatility.values[None, :]
    oi = g.openInterest.values[None, :]
    sgn = sign[None, :]

    def net_gex_at(S):
        S = np.atleast_1d(S)[:, None]
        return (sgn * bs_gamma(S, K, T, sig) * oi * 100 * S ** 2 * 0.01).sum(axis=1)

    net_now = float(net_gex_at(spot)[0])
    grid = spot * np.linspace(0.90, 1.10, 161)
    curve = net_gex_at(grid)
    flip = None
    crossings = np.where(np.sign(curve[:-1]) != np.sign(curve[1:]))[0]
    if len(crossings):
        i = crossings[np.argmin(np.abs(grid[crossings] - spot))]
        flip = float(grid[i])

    # --- Skew (put IV minus call IV, ~5% OTM, first expiry >= 7 dte)
    skew = None
    exps = sorted(df[df.dte >= 7].expiry.unique())
    if exps:
        e = df[df.expiry == exps[0]]
        p = e[(e.side == "put") & (e.impliedVolatility > 0.01)]
        c = e[(e.side == "call") & (e.impliedVolatility > 0.01)]
        if not p.empty and not c.empty:
            p_iv = p.iloc[(p.strike - spot * 0.95).abs().argsort()[:1]].impliedVolatility.iloc[0]
            c_iv = c.iloc[(c.strike - spot * 1.05).abs().argsort()[:1]].impliedVolatility.iloc[0]
            skew = float(p_iv - c_iv)

    # --- Flow
    pc_oi = float(put_oi.sum() / max(call_oi.sum(), 1))
    call_vol, put_vol = calls.volume.sum(), puts.volume.sum()
    pc_vol = float(put_vol / max(call_vol, 1))

    unusual = df[(df.volume > 500) & (df.volume > 3 * df.openInterest.clip(lower=1))]
    unusual = unusual.sort_values("volume", ascending=False).head(5)
    unusual = unusual[["side", "strike", "expiry", "volume", "openInterest"]]

    return {
        "call_wall": call_wall, "put_wall": put_wall,
        "net_gex": net_now, "flip": flip, "skew": skew,
        "pc_oi": pc_oi, "pc_vol": pc_vol, "unusual": unusual,
    }


# ----------------------------------------------------------------------------
# 3. Prices
# ----------------------------------------------------------------------------
def last_price(symbol: str) -> float:
    return float(yf.Ticker(symbol).history(period="5d")["Close"].iloc[-1])


# ----------------------------------------------------------------------------
# 4. Report
# ----------------------------------------------------------------------------
def label(score: float) -> str:
    if score >= 0.5:
        return "BULLISH"
    if score >= 0.15:
        return "MILD BULLISH"
    if score <= -0.5:
        return "BEARISH"
    if score <= -0.15:
        return "MILD BEARISH"
    return "NEUTRAL"


def gold_report(levels_manual, days):
    print("\n" + "=" * 60)
    print("XAUUSD (GOLD)")
    print("=" * 60)
    scores = []
    notes = []

    # COT
    cot = None
    try:
        cot = analyze_cot(fetch_cot(COT_CODES["GOLD"]))
        print(f"\n[COT {cot['date']}] Managed money net: {cot['mm_net']:,} "
              f"(percentile {cot['pct']:.0f}, 4w change {cot['chg4']:+,})")
        print(f"  Swap dealers net: {cot['swap_net']:,}")
        print(f"  State: {cot['state']}")
        scores.append(("COT", cot["score"], 0.4))
    except Exception as e:
        print(f"\n[COT] failed: {e}")

    # Options
    try:
        gc = last_price("GC=F")
        spot, chain = fetch_gld_chain(days)
        ratio = gc / spot
        op = analyze_options(spot, chain)
        conv = lambda x: None if x is None else x * ratio
        print(f"\n[PRICE] GC=F {gc:,.1f} | GLD {spot:,.2f} | ratio {ratio:.3f}")
        print(f"[GLD OPTIONS, next {days} days]")
        print(f"  Call wall (resistance): GLD {op['call_wall']} -> gold ~{conv(op['call_wall'])}")
        print(f"  Put wall  (support):    GLD {op['put_wall']} -> gold ~{conv(op['put_wall'])}")
        if op["flip"]:
            print(f"  Gamma flip: GLD {op['flip']:.2f} -> gold ~{conv(op['flip']):,.0f}")
        regime = "POSITIVE gamma (mean-reverting, range)" if op["net_gex"] > 0 \
            else "NEGATIVE gamma (moves accelerate, trend)"
        print(f"  Net GEX: {op['net_gex']:,.0f} -> {regime}")
        if op["skew"] is not None:
            print(f"  Skew (put IV - call IV, 5% OTM): {op['skew']:+.3f}")
        print(f"  Put/Call OI {op['pc_oi']:.2f} | Put/Call volume {op['pc_vol']:.2f}")
        if not op["unusual"].empty:
            print("  Unusual volume (vol > 3x OI):")
            print(op["unusual"].to_string(index=False))

        # Scores
        flow = float(np.clip((1.0 - op["pc_vol"]) , -1, 1)) * 0.5
        scores.append(("Flow", flow, 0.2))
        if op["skew"] is not None:
            scores.append(("Skew", float(np.clip(-op["skew"] * 5, -1, 1)), 0.2))
        if op["flip"]:
            pos = 0.3 if spot > op["flip"] else -0.3
            scores.append(("Vs gamma flip", pos, 0.2))

        notes.append(regime)
        if op["call_wall"] and op["put_wall"]:
            notes.append(f"Expected range from walls: gold ~{conv(op['put_wall']):,.0f} "
                         f"to ~{conv(op['call_wall']):,.0f}")

        # Manual levels
        if levels_manual:
            print("\n[YOUR QUIKSTRIKE LEVELS]")
            for lv in sorted(levels_manual):
                d = lv - gc
                side = "above (resistance)" if d > 0 else "below (support)"
                print(f"  {lv:,.0f}: {abs(d):,.1f} pts {side}")
    except Exception as e:
        print(f"\n[OPTIONS] failed: {e}")

    if scores:
        total = sum(s * w for _, s, w in scores) / sum(w for _, _, w in scores)
        print("\n--- ADVICE ---")
        for n, s, w in scores:
            print(f"  {n:<14} {s:+.2f}  (weight {w})")
        print(f"  Overall: {label(total)} ({total:+.2f})")
        for n in notes:
            print(f"  - {n}")
        print("  - Confirm at the levels with order flow (absorption / iceberg) "
              "before entering. Fade walls in positive gamma, "
              "trade breakouts in negative gamma.")


def gbpjpy_report():
    print("\n" + "=" * 60)
    print("GBPJPY")
    print("=" * 60)
    try:
        gbp = analyze_cot(fetch_cot(COT_CODES["GBP"]))
        jpy = analyze_cot(fetch_cot(COT_CODES["JPY"]))
    except Exception as e:
        print(f"[COT] failed: {e}")
        return
    print(f"\n[GBP futures] net {gbp['mm_net']:,} | pct {gbp['pct']:.0f} | 4w {gbp['chg4']:+,} | {gbp['state']}")
    print(f"[JPY futures] net {jpy['mm_net']:,} | pct {jpy['pct']:.0f} | 4w {jpy['chg4']:+,} | {jpy['state']}")

    # Funds long GBP and short JPY (JPY pct low) = crowd long GBPJPY
    momentum = (gbp["trend"] - jpy["trend"]) / 2
    crowd = (gbp["pct"] - jpy["pct"]) / 100
    print("\n--- ADVICE ---")
    print(f"  Momentum bias: {label(momentum * 0.6)} ({momentum:+.2f})")
    if crowd > 0.6:
        print("  - Crowd is heavily long GBPJPY (long GBP, short JPY): "
              "watch for a sharp unwind, especially on BoJ or risk-off news.")
    elif crowd < -0.6:
        print("  - Crowd is heavily short GBPJPY: squeeze risk on the upside.")
    else:
        print("  - Positioning not extreme: trade price action and levels.")
    print("  - No public dealer options data for GBPJPY. Use this as bias only; "
          "entries need your SMC / order flow levels.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", default="", help="Comma-separated gold levels from QuikStrike")
    ap.add_argument("--days", type=int, default=45, help="Max days to expiry to include")
    args = ap.parse_args()
    levels = [float(x) for x in args.levels.split(",") if x.strip()]

    print(f"Dealer Positioning Report | {datetime.now():%Y-%m-%d %H:%M}")
    gold_report(levels, args.days)
    gbpjpy_report()
    print("\nNot financial advice. Options-derived levels are estimates.")


if __name__ == "__main__":
    sys.exit(main())
