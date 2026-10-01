"""
cboe.py
Fetch an options chain from Cboe's public delayed-quotes feed (about 15 minutes
delayed; open interest updates once a day). Returns the same table layout as
dealer_positioning.fetch_gld_chain so the rest of the app works unchanged.

This is an unofficial public feed: it can change or block servers at any time.
"""

import re

import pandas as pd
import requests

URL = "https://cdn.cboe.com/api/global/delayed_quotes/options/{sym}.json"
HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                         "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
           "Accept": "application/json"}
OCC = re.compile(r"^(.+?)(\d{6})([CP])(\d{8})$")


def fetch_chain(symbol="GLD", max_days=45):
    r = requests.get(URL.format(sym=symbol), headers=HEADERS, timeout=30)
    r.raise_for_status()
    data = r.json().get("data", {})
    spot = data.get("current_price")
    opts = data.get("options", [])
    if not opts or not spot:
        raise ValueError("Cboe returned no options or no price")
    spot = float(spot)

    today = pd.Timestamp.today().normalize()
    rows = []
    for o in opts:
        m = OCC.match(str(o.get("option", "")))
        if not m:
            continue
        expiry = pd.to_datetime(m.group(2), format="%y%m%d", errors="coerce")
        if pd.isna(expiry):
            continue
        strike = int(m.group(4)) / 1000.0
        if strike < spot * 0.75 or strike > spot * 1.25:
            continue
        rows.append({
            "strike": strike,
            "side": "call" if m.group(3) == "C" else "put",
            "expiry": expiry,
            "dte": max(int((expiry - today).days), 1),
            "openInterest": o.get("open_interest"),
            "volume": o.get("volume"),
            "impliedVolatility": o.get("iv"),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("Cboe chain had no usable rows")
    for c in ("strike", "openInterest", "volume", "impliedVolatility"):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    if df["impliedVolatility"].median() > 3:  # given in percent, convert to decimal
        df["impliedVolatility"] = df["impliedVolatility"] / 100.0

    df = df[df.expiry >= today]
    chosen = df[df.dte <= max_days]
    if chosen.empty:
        first4 = sorted(df.expiry.unique())[:4]
        chosen = df[df.expiry.isin(first4)]
    if chosen.empty:
        raise ValueError("No Cboe expiries in range")
    return spot, chosen.reset_index(drop=True)
