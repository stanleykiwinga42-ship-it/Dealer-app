"""
Dealer Positioning Dashboard (Streamlit app)
Keep this file in the same folder as dealer_positioning.py
Run locally: streamlit run app.py
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
    parts = []  # (name, score, weight)
    cot, cot_df, op, spot, ratio, gc = None, None, None, None, None, None

    # COT (independent of options)
    try:
        cot_df = load_cot(dp.COT_CODES["GOLD"])
        cot = dp.analyze_cot(cot_df)
        parts.append(("COT", cot["score"], 0.4))
    except Exception as e:
        st.error(f"COT data failed: {e}")

    # Options (independent of COT)
    try:
        spot, chain = load_gld(days)
        op = dp.analyze_options(spot, chain)
        parts.append(("Flow", float(np.clip(1.0 - op["pc_vol"], -1, 1)) * 0.5, 0.2))
        if op["skew"] is not None:
            parts.append(("Skew", float(np.clip(-op["skew"] * 5, -1, 1)), 0.2))
        if op["flip"]:
            parts.append(("Vs gamma flip", 0.3 if spot > op["flip"] else -0.3, 0.2))
        try:
            gc = load_price("GC=F")
            ratio = gc / spot
        except Exception:
            gc, ratio = None, None
    except Exception as e:
        st.warning(f"Options data unavailable, showing COT only. Reason: {e}")

    # Verdict
    if parts:
        total = sum(s * w for _, s, w in parts) / sum(w for _, _, w in parts)
        c1, c2, c3 = st.columns(3)
        c1.metric("Gold (GC)", f"{gc:,.1f}" if gc else "n/a")
        c2.metric("Bias", dp.label(total), f"{total:+.2f}")
        if op:
            c3.metric("Gamma regime", "Positive" if op["net_gex"] > 0 else "Negative")
        else:
            c3.metric("Gamma regime", "n/a")

    # COT section
    if cot is not None:
        st.subheader("Funds positioning (COT)")
        st.write(f"**{cot['state']}** | net {cot['mm_net']:,} | percentile {cot['pct']:.0f} | "
                 f"4w change {cot['chg4']:+,} | report {cot['date']}")
        st.line_chart(cot_df.set_index("date")[["mm_net", "swap_net"]])

    # Options section
    if op is not None:
        st.subheader("Options levels")
        conv = (lambda x: round(x * ratio)) if ratio else (lambda x: x)
        unit = "gold price" if ratio else "GLD price"

        m1, m2, m3 = st.columns(3)
        m1.metric(f"Put wall ({unit})", f"{conv(op['put_wall']):,}" if op["put_wall"] else "n/a")
        m2.metric(f"Call wall ({unit})", f"{conv(op['call_wall']):,}" if op["call_wall"] else "n/a")
        m3.metric(f"Gamma flip ({unit})", f"{conv(op['flip']):,}" if op["flip"] else "n/a")

        if op["net_gex"] > 0:
            st.success("Positive gamma: fade the walls, expect ranges.")
        else:
            st.warning("Negative gamma: moves accelerate, favor breakouts.")

        near = chain[(chain.strike > spot * 0.9) & (chain.strike < spot * 1.1)].copy()
        near["level"] = (near.strike * (ratio or 1)).round().astype(int)
        st.caption(f"Open interest by strike ({unit})")
        st.bar_chart(near.pivot_table(index="level", columns="side",
                                      values="openInterest", aggfunc="sum").fillna(0))

        k1, k2, k3 = st.columns(3)
        k1.metric("Put/Call OI", f"{op['pc_oi']:.2f}")
        k2.metric("Put/Call volume", f"{op['pc_vol']:.2f}")
        k3.metric("Skew", f"{op['skew']:+.3f}" if op["skew"] is not None else "n/a")

        if not op["unusual"].empty:
            st.caption("Unusual volume (volume > 3x OI)")
            st.dataframe(op["unusual"], hide_index=True, use_container_width=True)

    if parts:
        st.subheader("Score breakdown")
        st.dataframe(pd.DataFrame(parts, columns=["Signal", "Score", "Weight"]),
                     hide_index=True, use_container_width=True)

    if levels and gc:
        st.subheader("Your levels")
        rows = [{"Level": lv, "Distance (pts)": round(lv - gc, 1),
                 "Role": "Resistance" if lv > gc else "Support"} for lv in sorted(levels)]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    st.caption("Confirm at levels with order flow before entry. GLD-based GEX is a proxy "
               "for COMEX gold and assumes customers are long calls.")

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
