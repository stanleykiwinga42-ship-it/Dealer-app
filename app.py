"""
Dealer Positioning Dashboard (Streamlit app)

Setup:
    pip install streamlit requests pandas numpy yfinance
    Put this file in the same folder as dealer_positioning.py
Run:
    streamlit run app.py
"""

import numpy as np
import pandas as pd
import streamlit as st

import dealer_positioning as dp

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


def gold_score(cot, op, spot):
    parts = [("COT", cot["score"], 0.4)]
    parts.append(("Flow", float(np.clip(1.0 - op["pc_vol"], -1, 1)) * 0.5, 0.2))
    if op["skew"] is not None:
        parts.append(("Skew", float(np.clip(-op["skew"] * 5, -1, 1)), 0.2))
    if op["flip"]:
        parts.append(("Vs gamma flip", 0.3 if spot > op["flip"] else -0.3, 0.2))
    total = sum(s * w for _, s, w in parts) / sum(w for _, _, w in parts)
    return total, parts


# ---------------------------------------------------------------- sidebar
st.sidebar.title("Settings")
days = st.sidebar.slider("Options expiry window (days)", 7, 90, 45)
levels_text = st.sidebar.text_input("Your QuikStrike gold levels (comma-separated)", "")
if st.sidebar.button("Refresh data"):
    st.cache_data.clear()
    st.rerun()
levels = [float(x) for x in levels_text.split(",") if x.strip()]

st.title("Dealer Positioning Dashboard")
st.caption("Free data only: CFTC COT + GLD options. Decision support, not financial advice.")

tab_gold, tab_gj = st.tabs(["XAUUSD (Gold)", "GBPJPY"])

# ---------------------------------------------------------------- gold
with tab_gold:
    try:
        gc = load_price("GC=F")
        spot, chain = load_gld(days)
        ratio = gc / spot
        op = dp.analyze_options(spot, chain)
        cot_df = load_cot(dp.COT_CODES["GOLD"])
        cot = dp.analyze_cot(cot_df)
        total, parts = gold_score(cot, op, spot)
        conv = lambda x: None if x is None else round(x * ratio)

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Gold (GC)", f"{gc:,.1f}")
        c2.metric("Bias", dp.label(total), f"{total:+.2f}")
        c3.metric("Gamma regime", "Positive" if op["net_gex"] > 0 else "Negative")
        c4.metric("Put wall (support)", f"{conv(op['put_wall']):,}" if op["put_wall"] else "n/a")
        c5.metric("Call wall (resistance)", f"{conv(op['call_wall']):,}" if op["call_wall"] else "n/a")

        if op["flip"]:
            st.info(f"Gamma flip near gold {conv(op['flip']):,}. "
                    f"Above it: calmer, mean-reverting. Below it: faster, trending.")
        if op["net_gex"] > 0:
            st.success("Positive gamma: fade the walls, expect ranges.")
        else:
            st.warning("Negative gamma: moves accelerate, favor breakouts.")

        left, right = st.columns(2)
        with left:
            st.subheader("Open interest by strike (gold price)")
            near = chain[(chain.strike > spot * 0.9) & (chain.strike < spot * 1.1)].copy()
            near["gold"] = (near.strike * ratio).round().astype(int)
            oi = near.pivot_table(index="gold", columns="side",
                                  values="openInterest", aggfunc="sum").fillna(0)
            st.bar_chart(oi)
        with right:
            st.subheader("Managed money net (COT)")
            st.line_chart(cot_df.set_index("date")[["mm_net", "swap_net"]])
            st.caption(f"{cot['state']} | percentile {cot['pct']:.0f} | "
                       f"4w change {cot['chg4']:+,} | report {cot['date']}")

        st.subheader("Score breakdown")
        st.dataframe(pd.DataFrame(parts, columns=["Signal", "Score", "Weight"]),
                     hide_index=True, use_container_width=True)

        m1, m2, m3 = st.columns(3)
        m1.metric("Put/Call OI", f"{op['pc_oi']:.2f}")
        m2.metric("Put/Call volume", f"{op['pc_vol']:.2f}")
        m3.metric("Skew (put IV - call IV)", f"{op['skew']:+.3f}" if op["skew"] is not None else "n/a")

        if not op["unusual"].empty:
            st.subheader("Unusual volume (volume > 3x OI)")
            st.dataframe(op["unusual"], hide_index=True, use_container_width=True)

        if levels:
            st.subheader("Your levels")
            rows = [{"Level": lv, "Distance (pts)": round(lv - gc, 1),
                     "Role": "Resistance" if lv > gc else "Support"} for lv in sorted(levels)]
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

        st.caption("Confirm at levels with order flow (absorption, icebergs) before entry. "
                   "GLD-based GEX is a proxy for COMEX gold and assumes customers are long calls.")
    except Exception as e:
        st.error(f"Gold data failed: {e}")

# ---------------------------------------------------------------- GBPJPY
with tab_gj:
    try:
        gbp_df = load_cot(dp.COT_CODES["GBP"])
        jpy_df = load_cot(dp.COT_CODES["JPY"])
        gbp, jpy = dp.analyze_cot(gbp_df), dp.analyze_cot(jpy_df)
        momentum = (gbp["trend"] - jpy["trend"]) / 2
        crowd = (gbp["pct"] - jpy["pct"]) / 100

        c1, c2, c3 = st.columns(3)
        c1.metric("GBPJPY momentum bias", dp.label(momentum * 0.6), f"{momentum:+.2f}")
        c2.metric("GBP funds net", f"{gbp['mm_net']:,}", f"{gbp['chg4']:+,} (4w)")
        c3.metric("JPY funds net", f"{jpy['mm_net']:,}", f"{jpy['chg4']:+,} (4w)")

        if crowd > 0.6:
            st.warning("Crowd heavily long GBPJPY (long GBP, short JPY): watch for a sharp unwind "
                       "on BoJ or risk-off news.")
        elif crowd < -0.6:
            st.warning("Crowd heavily short GBPJPY: squeeze risk to the upside.")
        else:
            st.info("Positioning not extreme. Trade price action and your levels.")

        st.subheader("Managed money net")
        both = pd.DataFrame({
            "GBP": gbp_df.set_index("date")["mm_net"],
            "JPY": jpy_df.set_index("date")["mm_net"],
        })
        st.line_chart(both)
        st.caption("No public dealer options data for GBPJPY. Use as bias only.")
    except Exception as e:
        st.error(f"GBPJPY data failed: {e}")
