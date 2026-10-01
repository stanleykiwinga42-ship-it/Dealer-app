"""
Dealer Positioning Dashboard (Streamlit app)
Files needed in the same folder: dealer_positioning.py, trade_plan.py
Run locally: streamlit run app.py
"""

import numpy as np
import pandas as pd
import streamlit as st

import dealer_positioning as dp
import trade_plan as tp

st.set_page_config(page_title="Dealer Positioning", page_icon="🥇", layout="wide")


@st.cache_data(ttl=900)
def load_cot(code: str) -> pd.DataFrame:
    return dp.fetch_cot(code)


@st.cache_data(ttl=900)
def load_gld(days: int):
    return dp.fetch_gld_chain(days)


@st.cache_data(ttl=300)
def load_price(symbol: str) -> float:
    return dp.last_price(symbol)


@st.cache_data(ttl=300)
def load_tv_auto():
    return tp.spot_from_yahoo()


@st.cache_data(ttl=900)
def load_atr():
    return tp.get_atr("GC=F")


# ---------------------------------------------------------------- sidebar
st.sidebar.title("Settings")
tv_input = st.sidebar.number_input(
    "TradingView XAUUSD price now (0 = auto)", min_value=0.0, value=0.0, step=0.1,
    help="Type the live price from your TradingView chart. This is the most reliable way "
         "to match the app to your chart.")
days = st.sidebar.slider("Options expiry window (days)", 7, 90, 45)
levels_text = st.sidebar.text_input(
    "QuikStrike levels (futures price, comma-separated)", "",
    help="Type the levels exactly as shown on CME QuikStrike. The app shifts them to "
         "TradingView prices.")
if st.sidebar.button("Refresh data"):
    st.cache_data.clear()
    st.rerun()
manual = [float(x) for x in levels_text.split(",") if x.strip()]

st.title("Dealer Positioning Dashboard")
st.caption("Free data: CFTC COT + GLD options. Rule-based framework, not financial advice.")

tab_gold, tab_gj = st.tabs(["XAUUSD (Gold)", "GBPJPY"])

# ---------------------------------------------------------------- gold
with tab_gold:
    parts, comps = [], {}
    cot = cot_df = op = chain = lv = None
    spot = gc = ratio = None

    # COT
    try:
        cot_df = load_cot(dp.COT_CODES["GOLD"])
        cot = dp.analyze_cot(cot_df)
        parts.append(("COT", cot["score"], 0.4))
    except Exception as e:
        st.error(f"COT data failed: {e}")

    # Prices
    try:
        gc = load_price("GC=F")
    except Exception:
        gc = None

    # Options
    opts_ok = False
    try:
        spot, chain = load_gld(days)
        op = dp.analyze_options(spot, chain)
        lv = tp.clean_levels(spot, chain)
        if gc:
            ratio = gc / spot
        opts_ok = bool(lv["ok"] and ratio)
    except Exception as e:
        st.warning(f"Options data unavailable. Reason: {e}")

    if opts_ok:
        parts.append(("Flow", float(np.clip(1.0 - op["pc_vol"], -1, 1)) * 0.5, 0.2))
        if op["skew"] is not None:
            parts.append(("Skew", float(np.clip(-op["skew"] * 5, -1, 1)), 0.2))
        if op["flip"]:
            parts.append(("Vs gamma flip", 0.3 if spot > op["flip"] else -0.3, 0.2))
    elif lv is not None:
        st.warning(f"Options data looks unreliable right now (only {lv['share'] * 100:.0f}% of "
                   f"strikes have open interest). Options levels are ignored; the plan uses "
                   f"COT and volatility only.")

    # TradingView alignment
    tv = tv_input if tv_input > 0 else load_tv_auto()
    tv_source = "typed" if tv_input > 0 else "auto (Yahoo, verify vs your chart)"
    offset = tp.tv_offset(tv, gc)
    price = tv or gc

    if price is None:
        st.error("Could not get a gold price. Type the TradingView price in the sidebar.")
    else:
        total = sum(s * w for _, s, w in parts) / sum(w for _, _, w in parts) if parts else 0.0
        comps = {n: s for n, s, _ in parts}

        c1, c2 = st.columns(2)
        c1.metric("TradingView XAUUSD", f"{price:,.2f}", tv_source if tv else "futures price")
        c2.metric("Offset (TV minus futures)", f"{offset:+.2f}")
        c3, c4 = st.columns(2)
        c3.metric("Bias", dp.label(total), f"{total:+.2f}")
        c4.metric("Gamma regime",
                  ("Positive" if op["net_gex"] > 0 else "Negative") if opts_ok else "n/a")

        # Levels in TradingView prices
        levels = []
        if opts_ok:
            if lv["put_wall"]:
                levels.append(("Put wall", tp.to_tv(lv["put_wall"], ratio, offset)))
            if lv["call_wall"]:
                levels.append(("Call wall", tp.to_tv(lv["call_wall"], ratio, offset)))
            if op["flip"]:
                levels.append(("Gamma flip", tp.to_tv(op["flip"], ratio, offset)))
        for m in manual:
            levels.append((f"QuikStrike {m:g}", m + offset))

        # Trade plan
        atr, atr_label = load_atr()
        if atr is None:
            atr, atr_label = price * 0.004, "estimate"

        plan = tp.build_plan(price, total, comps, atr, levels,
                             (op["net_gex"] > 0) if opts_ok else None, opts_ok,
                             cot["pct"] if cot else None)

        st.subheader("Trade plan")
        if plan["side"] == "WAIT":
            st.info("WAIT. " + plan["reasons"][0])
        else:
            msg = f"{plan['side']}  |  Quality: {plan['grade']} ({plan['score']}/100)"
            if plan["grade"] == "GOOD":
                st.success(msg)
            elif plan["grade"] == "MID":
                st.warning(msg)
            else:
                st.error(msg)
            p1, p2 = st.columns(2)
            p1.metric("Entry", f"{plan['entry']:,.2f}", plan["entry_type"])
            p2.metric("Stop loss", f"{plan['sl']:,.2f}", f"risk {plan['risk_pts']:.1f} pts")
            p3, p4 = st.columns(2)
            p3.metric("TP1", f"{plan['tp1']:,.2f}", f"{plan['rr1']:.1f}R")
            p4.metric("TP2", f"{plan['tp2']:,.2f}", f"{plan['rr2']:.1f}R")
            for r in plan["reasons"]:
                st.write("• " + r)
        for w in plan["warnings"]:
            st.warning(w)
        st.caption(f"Stop distance uses {atr_label} ATR(14) = {atr:.1f} pts. "
                   f"Entry above/below price are limit orders at levels; check before placing.")

        # Levels table
        if levels:
            st.subheader("Key levels (TradingView prices)")
            rows = [{"Level": n, "Price": round(p, 1), "Distance": round(p - price, 1),
                     "Role": "Resistance" if p > price else "Support"}
                    for n, p in sorted(levels, key=lambda x: x[1], reverse=True)]
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    # COT section
    if cot is not None:
        st.subheader("Funds positioning (COT)")
        st.write(f"**{cot['state']}** | net {cot['mm_net']:,} | percentile {cot['pct']:.0f} | "
                 f"4w change {cot['chg4']:+,} | report {cot['date']}")
        st.line_chart(cot_df.set_index("date")[["mm_net", "swap_net"]])

    # Options detail
    if opts_ok:
        st.subheader("Options detail")
        k1, k2, k3 = st.columns(3)
        k1.metric("Put/Call OI", f"{op['pc_oi']:.2f}")
        k2.metric("Put/Call volume", f"{op['pc_vol']:.2f}")
        k3.metric("Skew", f"{op['skew']:+.3f}" if op["skew"] is not None else "n/a")

        near = chain[(chain.strike > spot * 0.9) & (chain.strike < spot * 1.1)
                     & (chain.openInterest > 0)].copy()
        if not near.empty:
            near["level"] = (near.strike * ratio + offset).round().astype(int)
            st.caption("Open interest by strike (TradingView price)")
            st.bar_chart(near.pivot_table(index="level", columns="side",
                                          values="openInterest", aggfunc="sum").fillna(0))
        if not op["unusual"].empty:
            st.caption("Unusual volume (volume > 3x OI)")
            st.dataframe(op["unusual"], hide_index=True, use_container_width=True)

    if parts:
        st.subheader("Score breakdown")
        st.dataframe(pd.DataFrame(parts, columns=["Signal", "Score", "Weight"]),
                     hide_index=True, use_container_width=True)

    st.caption("Confirm at levels with order flow before entry. GLD-based levels are a proxy "
               "for COMEX gold. Futures-to-spot offset changes over time, so retype the "
               "TradingView price each session.")

# ---------------------------------------------------------------- GBPJPY
with tab_gj:
    try:
        gbp_df = load_cot(dp.COT_CODES["GBP"])
        jpy_df = load_cot(dp.COT_CODES["JPY"])
        gbp, jpy = dp.analyze_cot(gbp_df), dp.analyze_cot(jpy_df)
        momentum = (gbp["trend"] - jpy["trend"]) / 2
        crowd = (gbp["pct"] - jpy["pct"]) / 100

        c1, c2, c3 = st.columns(3)
        c1.metric("GBPJPY bias", dp.label(momentum * 0.6), f"{momentum:+.2f}")
        c2.metric("GBP funds net", f"{gbp['mm_net']:,}", f"{gbp['chg4']:+,} (4w)")
        c3.metric("JPY funds net", f"{jpy['mm_net']:,}", f"{jpy['chg4']:+,} (4w)")

        if crowd > 0.6:
            st.warning("Crowd heavily long GBPJPY (long GBP, short JPY): watch for a sharp "
                       "unwind on BoJ or risk-off news.")
        elif crowd < -0.6:
            st.warning("Crowd heavily short GBPJPY: squeeze risk to the upside.")
        else:
            st.info("Positioning not extreme. Trade price action and your levels.")

        st.line_chart(pd.DataFrame({
            "GBP": gbp_df.set_index("date")["mm_net"],
            "JPY": jpy_df.set_index("date")["mm_net"],
        }))
        st.caption("No public dealer options data for GBPJPY. Use as bias only.")
    except Exception as e:
        st.error(f"GBPJPY data failed: {e}")
