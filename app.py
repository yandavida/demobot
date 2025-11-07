import numpy as np
import pandas as pd
import streamlit as st

from bs import bs_greeks
from signals import build_iron_condor, payoff_iron_condor

st.set_page_config(page_title="Iron Condor Desk", layout="wide")

st.title("🦅 Iron Condor Desk")
st.caption("Black–Scholes pricing, greeks, and payoff explorer")

with st.sidebar:
    st.header("Market Inputs")
    spot = st.number_input("Underlying Spot", min_value=1.0, value=420.0, step=1.0)
    risk_free = st.number_input(
        "Risk-free rate (annual)", min_value=0.0, value=0.03, step=0.005, format="%.3f"
    )
    dividend = st.number_input(
        "Dividend yield (annual)", min_value=0.0, value=0.00, step=0.005, format="%.3f"
    )
    iv = st.number_input(
        "Implied volatility (annual σ)", min_value=0.01, value=0.18, step=0.01, format="%.2f"
    )
    dte = st.number_input("Days to expiration", min_value=1, value=30, step=1)
    time_to_expiry = dte / 365.0

    st.header("Structure")
    target_delta = st.slider("Target short delta (abs)", 0.05, 0.35, 0.12, 0.01)
    wing_width = st.number_input(
        "Wing width (same units as spot)", min_value=0.5, value=5.0, step=0.5
    )
    multiplier = st.number_input("Contract multiplier", min_value=1, value=100, step=1)
    quantity = st.number_input("Quantity", min_value=1, value=1, step=1)

condor = build_iron_condor(
    spot,
    risk_free,
    dividend,
    iv,
    time_to_expiry,
    target_delta=target_delta,
    width=wing_width,
)

legs = list(condor.legs)
legs_table = []
for leg in legs:
    greeks = bs_greeks(spot, leg.strike, risk_free, dividend, iv, time_to_expiry, leg.typ)
    sign = leg.sign
    legs_table.append(
        {
            "Action": leg.action,
            "Type": leg.typ,
            "Strike": round(leg.strike, 2),
            "Theo Price": round(leg.price, 4),
            "Delta": round(sign * greeks.delta, 4),
            "Theta / day": round(sign * greeks.theta / 365.0, 6),
            "Vega / 1%": round(sign * greeks.vega * 0.01, 6),
        }
    )

st.subheader("Leg detail (per contract)")
st.dataframe(pd.DataFrame(legs_table), use_container_width=True)

credit_points = condor.credit
max_risk_points = max(condor.width - credit_points, 0.0)
rr = credit_points / max_risk_points if max_risk_points else 0.0

st.subheader("P&L metrics")
col_a, col_b, col_c = st.columns(3)
with col_a:
    st.metric("Net credit (points)", f"{credit_points:.3f}")
    st.metric("Net credit (currency)", f"{credit_points * multiplier * quantity:,.2f}")
with col_b:
    st.metric("Max risk (points)", f"{max_risk_points:.3f}")
    st.metric("Max risk (currency)", f"{max_risk_points * multiplier * quantity:,.2f}")
with col_c:
    st.metric("Reward / Risk", f"{rr:.2f}")
    st.metric(
        "Breakeven cushion (points)",
        f"{condor.width - max_risk_points if condor.width else 0.0:.3f}",
    )

st.subheader("Strike placement")
st.markdown(
    (
        f"Short call **{condor.short_call.strike:.2f}**, long call **{condor.long_call.strike:.2f}**  "
        f"Short put **{condor.short_put.strike:.2f}**, long put **{condor.long_put.strike:.2f}**"
    )
)

price_grid = np.linspace(spot * 0.92, spot * 1.08, 200)
payoff = payoff_iron_condor(condor, price_grid)

st.subheader("Expiry payoff (per contract)")
payoff_df = pd.DataFrame({"Underlying": price_grid, "Payoff": payoff}).set_index("Underlying")
st.line_chart(payoff_df)

st.caption("All greeks shown are per contract. Multiply by contract multiplier and quantity for desk totals.")
