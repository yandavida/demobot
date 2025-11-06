import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from bs import bs_greeks
from signals import build_iron_condor, payoff_iron_condor

st.set_page_config(page_title='Option Bot – Iron Condor', layout='wide')

st.title('🦅 Option Bot – Iron Condor (Skeleton)')
st.caption('Educational prototype · BS pricing · Greeks · Payoff')

with st.sidebar:
    st.header('Market Inputs')
    S   = st.number_input('Spot (S)', min_value=1.0, value=500.0, step=1.0)
    r   = st.number_input('Risk‑free (annual, r)', min_value=0.0, value=0.03, step=0.005, format='%.3f')
    q   = st.number_input('Dividend yield (annual, q)', min_value=0.0, value=0.00, step=0.005, format='%.3f')
    iv  = st.number_input('IV (annual σ)', min_value=0.01, value=0.16, step=0.01, format='%.2f')
    dte = st.number_input('Days to expiry (DTE)', min_value=1, value=21, step=1)
    t_years = dte/365.0

    st.header('Strategy')
    target_delta = st.slider('Target Delta (abs.)', 0.05, 0.35, 0.12, 0.01)
    width = st.number_input('Wing width (same units as S)', min_value=0.5, value=5.0, step=0.5)
    multiplier = st.number_input('Multiplier ($/pt)', min_value=1, value=100, step=1)
    qty = st.number_input('Quantity', min_value=1, value=1, step=1)

condor = build_iron_condor(S, r, q, iv, t_years, target_delta=target_delta, width=width)

col1, col2, col3 = st.columns(3)
with col1:
    st.subheader('Legs & Greeks (per unit)')
    rows = []
    for leg in [condor.short_call, condor.long_call, condor.short_put, condor.long_put]:
        g = bs_greeks(S, leg.strike, r, q, iv, t_years, leg.typ)
        rows.append([leg.action, leg.typ, round(leg.strike,2), round(leg.price,3), round(leg.delta,3),
                     round(g.theta/365.0, 6), round(g.vega/100.0, 6)])
    st.table({'Action':[r[0] for r in rows],
              'Type':[r[1] for r in rows],
              'Strike':[r[2] for r in rows],
              'Theo Price':[r[3] for r in rows],
              'Delta':[r[4] for r in rows],
              'Theta/day':[r[5] for r in rows],
              'Vega/1%':[r[6] for r in rows]})

with col2:
    st.subheader('Credit & Risk (per 1 unit)')
    credit = condor.credit
    max_risk = max(condor.width, 1e-9) - credit
    rr = credit / max_risk if max_risk > 0 else 0.0
    st.metric('Net Credit (points)', f'{credit:.3f}')
    st.metric('Max Risk (points)', f'{max_risk:.3f}')
    st.metric('R/R', f'{rr:.2f}')
    st.write(f'Notional × multiplier × qty → Credit: **{credit*multiplier*qty:.2f}**, MaxRisk: **{max_risk*multiplier*qty:.2f}**')

with col3:
    st.subheader('Strikes')
    st.write(f'Short Call: **{condor.short_call.strike:.2f}**  |  Long Call: **{condor.long_call.strike:.2f}**')
    st.write(f'Short Put:  **{condor.short_put.strike:.2f}**   |  Long Put:  **{condor.long_put.strike:.2f}**')

st.subheader('Payoff at Expiry (per 1 unit)')
lo = S*(1-0.08); hi = S*(1+0.08)
grid = np.linspace(lo, hi, 200)
payoff = payoff_iron_condor(condor, grid)

fig, ax = plt.subplots()
ax.plot(grid, payoff)
ax.axhline(0, linewidth=0.8)
ax.axvline(S, linewidth=0.8)
ax.set_xlabel('Underlying Price at Expiry')
ax.set_ylabel('P/L (points)')
st.pyplot(fig)

st.divider()
st.caption('To enable IBKR orders, open ibkr.py and wire a button that calls place_condor_IBKR().')
