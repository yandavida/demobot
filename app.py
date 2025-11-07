import math
from dataclasses import dataclass
from typing import Tuple
import numpy as np
import pandas as pd
import streamlit as st

# ===== Black–Scholes =====
def _phi(x: float) -> float:
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

def _norm_pdf(x: float) -> float:
    return math.exp(-0.5*x*x) / math.sqrt(2*math.pi)

def d1(S, K, r, q, sigma, t):
    if sigma <= 0 or t <= 0: return 0.0
    return (math.log(S/K) + (r - q + 0.5*sigma*sigma)*t) / (sigma*math.sqrt(t))

def d2(S, K, r, q, sigma, t):
    return d1(S,K,r,q,sigma,t) - sigma*math.sqrt(t)

def bs_price(S, K, r, q, sigma, t, typ='C'):
    if t <= 0 or sigma <= 0:
        return max(0.0, (S-K) if typ.upper()=='C' else (K-S))
    D1, D2 = d1(S,K,r,q,sigma,t), d2(S,K,r,q,sigma,t)
    if typ.upper()=='C':
        return S*math.exp(-q*t)*_phi(D1) - K*math.exp(-r*t)*_phi(D2)
    return K*math.exp(-r*t)*_phi(-D2) - S*math.exp(-q*t)*_phi(-D1)

@dataclass
class Greeks:
    delta: float
    gamma: float
    theta: float
    vega: float

def bs_greeks(S, K, r, q, sigma, t, typ='C') -> Greeks:
    if t <= 0 or sigma <= 0: return Greeks(0.0,0.0,0.0,0.0)
    D1, D2 = d1(S,K,r,q,sigma,t), d2(S,K,r,q,sigma,t)
    pdf = _norm_pdf(D1)
    delta = math.exp(-q*t)*(_phi(D1) if typ.upper()=='C' else (_phi(D1)-1.0))
    gamma = math.exp(-q*t)*pdf/(S*sigma*math.sqrt(t))
    first = -(S*math.exp(-q*t)*pdf*sigma)/(2.0*math.sqrt(t))
    if typ.upper()=='C':
        theta = first - r*K*math.exp(-r*t)*_phi(D2) + q*S*math.exp(-q*t)*_phi(D1)
    else:
        theta = first + r*K*math.exp(-r*t)*_phi(-D2) - q*S*math.exp(-q*t)*_phi(-D1)
    vega = S*math.exp(-q*t)*pdf*math.sqrt(t)
    return Greeks(delta, gamma, theta, vega)

# ===== Strategy (single-file) =====
@dataclass
class Leg:
    typ: str
    strike: float
    action: str
    price: float
    delta: float

@dataclass
class Condor:
    short_call: Leg
    long_call: Leg
    short_put: Leg
    long_put: Leg
    @property
    def credit(self) -> float:
        return (self.short_call.price - self.long_call.price +
                self.short_put.price - self.long_put.price)
    @property
    def width(self) -> float:
        return max(self.long_call.strike - self.short_call.strike,
                   self.short_put.strike - self.long_put.strike)

def _strike_by_target_delta(S, r, q, sigma, t_years, target_delta_abs, typ, tol=1e-4) -> Tuple[float,float]:
    vol = max(sigma, 0.05)
    w = 3.0 * vol * math.sqrt(max(t_years, 1e-6))
    k_low, k_high = S*math.exp(-w)*0.8, S*math.exp(w)*1.2
    def f(K): return abs(bs_greeks(S,K,r,q,sigma,t_years,typ).delta) - target_delta_abs
    lo, hi = k_low, k_high
    flo, fhi = f(lo), f(hi)
    for _ in range(50):
        mid = 0.5*(lo+hi); val = f(mid)
        if abs(val) < tol:
            return mid, bs_greeks(S, mid, r, q, sigma, t_years, typ).delta
        if (flo>0 and val>0) or (flo<0 and val<0):
            lo, flo = mid, val
        else:
            hi, fhi = mid, val
    K = 0.5*(lo+hi)
    return K, bs_greeks(S, K, r, q, sigma, t_years, typ).delta

def build_iron_condor(S, r, q, sigma, t_years, target_delta=0.12, width=5.0) -> Condor:
    k_sc, d_sc = _strike_by_target_delta(S,r,q,sigma,t_years,target_delta,'C')
    k_sp, d_sp = _strike_by_target_delta(S,r,q,sigma,t_years,target_delta,'P')
    k_lc, k_lp = k_sc+width, k_sp-width
    sc_p = bs_price(S,k_sc,r,q,sigma,t_years,'C')
    lc_p = bs_price(S,k_lc,r,q,sigma,t_years,'C')
    sp_p = bs_price(S,k_sp,r,q,sigma,t_years,'P')
    lp_p = bs_price(S,k_lp,r,q,sigma,t_years,'P')
    return Condor(
        short_call=Leg('C',k_sc,'SELL',sc_p,d_sc),
        long_call =Leg('C',k_lc,'BUY', lc_p,0.0),
        short_put =Leg('P',k_sp,'SELL',sp_p,d_sp),
        long_put  =Leg('P',k_lp,'BUY', lp_p,0.0),
    )

def payoff_iron_condor(condor: Condor, S_grid: np.ndarray) -> np.ndarray:
    call = lambda S,K: np.maximum(S-K,0.0)
    put  = lambda S,K: np.maximum(K-S,0.0)
    credit = condor.credit
    return (credit
            - call(S_grid, condor.short_call.strike)
            + call(S_grid, condor.long_call.strike)
            - put (S_grid, condor.short_put.strike)
            + put (S_grid, condor.long_put.strike))

# ===== UI =====
st.set_page_config(page_title="Iron Condor Streamlit Desk", layout="wide")
st.title("🦅 Iron Condor – Streamlit Desk (Single-File)")

with st.sidebar:
    st.header("Market")
    S   = st.number_input("Spot (S)", min_value=1.0, value=3317.09, step=1.0)
    r   = st.number_input("Risk-free r (annual)", 0.0, 0.03, 0.005, format="%.3f")
    q   = st.number_input("Dividend yield q (annual)", 0.0, 0.00, 0.005, format="%.3f")
    iv  = st.number_input("IV σ (annual)", 0.01, 0.16, 0.01, format="%.2f")
    dte = st.number_input("Days to expiry (DTE)", 1, 21, 1)
    t   = dte/365.0
    st.header("Strategy")
    target_delta = st.slider("Target Delta (abs.)", 0.05, 0.35, 0.12, 0.01)
    width        = st.number_input("Wing width (same units as S)", 0.5, 20.0, 0.5)
    multiplier   = st.number_input("Multiplier (₪/$ per point)", 1, 50, 1)
    qty          = st.number_input("Quantity", 1, 1, 1)

condor = build_iron_condor(S, r, q, iv, t, target_delta, width)

# Legs table
rows = []
for leg in [condor.short_call, condor.long_call, condor.short_put, condor.long_put]:
    g = bs_greeks(S, leg.strike, r, q, iv, t, leg.typ)
    rows.append([leg.action, leg.typ, round(leg.strike,2), round(leg.price,3),
                 round(leg.delta,3), round(g.theta/365.0,6), round(g.vega/100.0,6)])
st.subheader("Legs (per unit)")
st.table(pd.DataFrame(rows, columns=["Action","Type","Strike","Theo Price","Delta","Theta/day","Vega/1%"]))

# Metrics
st.subheader("Credit & Risk (per unit)")
credit = condor.credit
max_risk = max(condor.width - credit, 0.0)
rr = (credit/max_risk) if max_risk>0 else 0.0
st.write(f"**Net Credit (pts):** {credit:.3f}  |  **Max Risk (pts):** {max_risk:.3f}  |  **R/R:** {rr:.2f}")
st.write(f"Amounts × multiplier × qty → Credit: **{credit*multiplier*qty:.2f}**, MaxRisk: **{max_risk*multiplier*qty:.2f}**")
st.write(f"Δ(short call) ≈ {condor.short_call.delta:.3f}  |  Δ(short put) ≈ {abs(condor.short_put.delta):.3f}")

# Payoff chart (built-in Streamlit)
st.subheader("Payoff at Expiry (per unit)")
grid = np.linspace(S*0.92, S*1.08, 200)
payoff = payoff_iron_condor(condor, grid)
df = pd.DataFrame({"Underlying": grid, "Payoff": payoff}).set_index("Underlying")
st.line_chart(df)
st.caption("Zero line = break-even. Adjust inputs on the left.")
