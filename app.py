"""
Dealer Positioning Dashboard (Streamlit app) - intraday version
Files needed in the same folder: dealer_positioning.py, trade_plan.py, intraday.py
Run locally: streamlit run app.py
"""

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st

import dealer_positioning as dp
import intraday as idy
import trade_plan as tp

st.set_page_config(page_title="Dealer Positioning", page_icon="🥇", layout="wide")


@st.cache_data(ttl=900)
def load_cot(code: str) -> pd.DataFrame:
    return dp.fetch_cot(code)


@st.cache_data(ttl=900)
def load_gld(days: int):
    return dp.fetch_gld_chain(days)


@st.cache_data(ttl=30)
def load_live(symbol: str):
    return idy.live_price(symbol)


@st.cache_data(ttl=300)
def load_bars(symbol: str, interval: str, period: str):
    return idy.get_bars(symbol, interval, period)


# ---------------------------------------------------------------- sidebar
ss = st.session_state
st.sidebar.title("Settings")
tv_input = st.sidebar.number_input(
    "TradingView XAUUSD price now (0 = auto)", min_value=0.0, value=0.0, step=0.1,
    help="Type the live price from your TradingView chart. The app locks the gap between "
         "TradingView and futures at that moment, then tracks the live price from it. "
         "Type a new price any time to re-sync.")
risk_usd = st.sidebar.number_input("Risk per trade ($)", min_value=1.0, value=50.0, step=10.0)
days = st.sidebar.slider("Options expiry window (days)", 7, 90, 30)
levels_text = st.sidebar.text_input(
    "Extra levels (futures price, comma-separated)", "",
    help="Levels you read from CME QuikStrike or elsewhere, in futures price. "
         "The app shifts them to TradingView prices.")
with st.sidebar.expander("Paste CME QuikStrike OI table"):
    st.caption("One line per strike: strike call_oi put_oi (plain numbers, in futures strikes). "
               "Example: 4200 1800 100")
    paste_text = st.text_area("OI table", "", height=150)
if st.sidebar.button("Refresh everything"):
    st.cache_data.clear()
    st.rerun()
manual = [float(x) for x in levels_text.split(",") if x.strip()]

# ---------------------------------------------------------------- live price + offset
now_utc = datetime.now(timezone.utc)
gc_now = load_live("GC=F")

if tv_input > 0:
    if ss.get("tv_typed") != tv_input and gc_now:
        ss["tv_typed"] = tv_input
        ss["offset"] = tv_input - gc_now
    offset_src = "locked to your typed TradingView price"
else:
    ss["tv_typed"] = None
    auto = load_live("XAUUSD=X")
    ss["offset"] = (auto - gc_now) if (auto and gc_now) else 0.0
    offset_src = "auto from Yahoo spot gold (type your TradingView price to be exact)"
offset = float(ss.get("offset", 0.0))


@st.fragment(run_every=15)
def price_strip():
    g = idy.live_price("GC=F")
    off = float(st.session_state.get("offset", 0.0))
    a, b = st.columns(2)
    a.metric("XAUUSD (TradingView est.)", f"{g + off:,.2f}" if g else "n/a")
    b.metric("GC futures", f"{g:,.2f}" if g else "n/a")
    st.caption(f"Live price updates every 15s | {datetime.now(timezone.utc):%H:%M:%S} UTC | "
               f"Yahoo can lag 1-2 min")


st.title("Dealer Positioning Dashboard")
price_strip()
st.caption(f"Offset (TradingView minus futures): {offset:+.2f} ({offset_src}). "
           f"Today (UTC): {now_utc:%Y-%m-%d %H:%M}.")

tab_gold, tab_gj = st.tabs(["XAUUSD (Gold)", "GBPJPY"])

# ---------------------------------------------------------------- gold
with tab_gold:
    parts = []
    cot = cot_df = op = chain = lv = exp = None
    spot = ratio = None

    try:
        cot_df = load_cot(dp.COT_CODES["GOLD"])
        cot = dp.analyze_cot(cot_df)
        parts.append(("COT", cot["score"], 0.4))
    except Exception as e:
        st.error(f"COT data failed: {e}")

    opts_ok = False
    try:
        spot, chain = load_gld(days)
        op = dp.analyze_options(spot, chain)
        lv = tp.clean_levels(spot, chain)
        if gc_now:
            ratio = gc_now / spot
        opts_ok = bool(lv["ok"] and ratio)
        if opts_ok:
            exp = idy.expiry_info(chain, spot)
    except Exception as e:
        st.warning(f"Options data unavailable. Reason: {e}")

    if opts_ok:
        parts.append(("Flow", float(np.clip(1.0 - op["pc_vol"], -1, 1)) * 0.5, 0.2))
        if op["skew"] is not None:
            parts.append(("Skew", float(np.clip(-op["skew"] * 5, -1, 1)), 0.2))
        if op["flip"]:
            parts.append(("Vs gamma flip", 0.3 if spot > op["flip"] else -0.3, 0.2))
    elif lv is not None:
        st.warning(f"GLD options data looks unreliable (only {lv['share'] * 100:.0f}% of strikes "
                   f"have open interest), so it is ignored. Paste CME levels in the sidebar for "
                   f"better results.")

    price = (gc_now + offset) if gc_now else (tv_input or None)
    if price is None:
        st.error("Could not get a gold price. Type the TradingView price in the sidebar.")
        st.stop()

    total = sum(s * w for _, s, w in parts) / sum(w for _, _, w in parts) if parts else 0.0

    # Levels in TradingView prices
    levels = []
    if opts_ok:
        if lv["put_wall"]:
            levels.append(("GLD put wall", tp.to_tv(lv["put_wall"], ratio, offset)))
        if lv["call_wall"]:
            levels.append(("GLD call wall", tp.to_tv(lv["call_wall"], ratio, offset)))
        if op["flip"]:
            levels.append(("Gamma flip", tp.to_tv(op["flip"], ratio, offset)))
    cme_df = idy.parse_oi_table(paste_text)
    if cme_df is not None and gc_now:
        cme = idy.cme_levels(cme_df, gc_now)
        for name, key in (("CME put wall", "put_wall"), ("CME call wall", "call_wall"),
                          ("CME max pain", "max_pain")):
            if cme[key]:
                levels.append((name, cme[key] + offset))
    for m in manual:
        levels.append((f"Level {m:g}", m + offset))

    pin = None
    if exp and exp["max_pain"] and ratio:
        pin = tp.to_tv(exp["max_pain"], ratio, offset)

    # Intraday inputs
    m15 = load_bars("GC=F", "15m", "5d")
    h1 = load_bars("GC=F", "1h", "1mo")
    trend = idy.trend_snapshot(m15, h1)
    atr = trend["atr15"] or price * 0.0025
    session = idy.session_info(now_utc)

    plan = idy.build_plan(price, total, atr, levels, trend,
                          (op["net_gex"] > 0) if opts_ok else None, session, pin,
                          exp["dte"] if exp else None, cot["pct"] if cot else None)

    # ---- display
    c1, c2 = st.columns(2)
    c1.metric("Bias (COT + options)", dp.label(total), f"{total:+.2f}")
    c2.metric("Gamma regime",
              ("Positive" if op["net_gex"] > 0 else "Negative") if opts_ok else "unknown")

    st.subheader("Trade plan (intraday)")
    if plan["side"] == "WAIT":
        st.info("WAIT. " + plan["reasons"][0])
    else:
        msg = f"{plan['side']} | {plan['setup']} | Grade {plan['grade']} ({plan['score']}/100)"
        (st.success if plan["score"] >= 70 else st.warning if plan["score"] >= 50
         else st.error)(msg)
        st.write("**What to expect:** " + plan["expect"])
        p1, p2 = st.columns(2)
        p1.metric("Entry", f"{plan['entry']:,.2f}", plan["entry_type"], delta_color="off")
        p2.metric("Stop loss", f"{plan['sl']:,.2f}", f"risk {plan['risk_pts']:.1f} pts",
                  delta_color="off")
        p3, p4 = st.columns(2)
        p3.metric("TP1", f"{plan['tp1']:,.2f}", f"{plan['rr1']:.1f}R", delta_color="off")
        p4.metric("TP2", f"{plan['tp2']:,.2f}", f"{plan['rr2']:.1f}R", delta_color="off")
        lots = risk_usd / (plan["risk_pts"] * 100) if plan["risk_pts"] else 0
        st.write(f"**Size:** about {lots:.2f} lots for ${risk_usd:,.0f} risk "
                 f"(assumes 100 oz per lot; check your broker).")
        for r in plan["reasons"]:
            st.write("• " + r)
        st.caption(plan["time_stop"])
    for w in plan.get("warnings", []):
        st.warning(w)

    st.subheader("Market conditions")
    st.write(f"**Session:** {session['name']}")
    if exp:
        st.write(f"**Next options expiry:** {exp['date']} ({exp['dte']} day(s)). "
                 + (f"Max-pain price ~{pin:,.0f}." if pin else ""))
    t_txt = {1: "up", -1: "down", 0: "mixed"}
    st.write(f"**Trend:** 15m {t_txt[trend['dir15']]}, 1h {t_txt[trend['dir1h']]} | "
             f"RSI(15m) {trend['rsi15']:.0f} | ATR(15m) {atr:.1f} pts"
             if trend["rsi15"] is not None else "**Trend:** price bars unavailable.")

    if levels:
        st.subheader("Key levels (TradingView prices)")
        rows = [{"Level": n, "Price": round(p, 1), "Distance": round(p - price, 1),
                 "Role": "Resistance" if p > price else "Support"}
                for n, p in sorted(levels, key=lambda x: x[1], reverse=True)]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    if cot is not None:
        st.subheader("Funds positioning (COT, weekly)")
        st.write(f"**{cot['state']}** | net {cot['mm_net']:,} | percentile {cot['pct']:.0f} | "
                 f"4w change {cot['chg4']:+,}")
        st.caption(f"Latest COT report date: {cot['date']}. The chart dates are weekly report "
                   f"dates from past years, not today's date.")
        st.line_chart(cot_df.set_index("date")[["mm_net", "swap_net"]])

    if opts_ok:
        st.subheader("Options detail (GLD proxy)")
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
        st.subheader("Bias breakdown")
        st.dataframe(pd.DataFrame(parts, columns=["Signal", "Score", "Weight"]),
                     hide_index=True, use_container_width=True)

    st.caption("Rule-based framework, not financial advice. Confirm at levels with order flow "
               "(absorption, icebergs) before entry. Re-type the TradingView price now and "
               "then, because the futures-to-spot gap drifts.")

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
        st.caption("No public dealer options data for GBPJPY. Bias only; the intraday plan "
                   "is built for gold.")
    except Exception as e:
        st.error(f"GBPJPY data failed: {e}")
