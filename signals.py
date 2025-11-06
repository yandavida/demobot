from dataclasses import dataclass
import numpy as np
from bs import bs_price, bs_greeks

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
    def credit(self):
        return (self.short_call.price - self.long_call.price + self.short_put.price - self.long_put.price)
    @property
    def width(self):
        return max(self.long_call.strike-self.short_call.strike, self.short_put.strike-self.long_put.strike)

def build_iron_condor(S,r,q,sigma,t_years,target_delta=0.12,width=5.0):
    call_k = S*(1.0+0.02)
    put_k  = S*(1.0-0.02)
    sc = bs_price(S, call_k, r, q, sigma, t_years, 'C')
    lc = bs_price(S, call_k+width, r, q, sigma, t_years, 'C')
    sp = bs_price(S, put_k,  r, q, sigma, t_years, 'P')
    lp = bs_price(S, put_k-width,  r, q, sigma, t_years, 'P')
    d_sc = bs_greeks(S, call_k, r, q, sigma, t_years, 'C').delta
    d_sp = bs_greeks(S, put_k,  r, q, sigma, t_years, 'P').delta
    return Condor(
        Leg('C', call_k, 'SELL', sc, d_sc),
        Leg('C', call_k+width, 'BUY',  lc, 0.0),
        Leg('P', put_k,  'SELL', sp, d_sp),
        Leg('P', put_k-width, 'BUY',  lp, 0.0),
    )

def payoff_iron_condor(condor, S_grid):
    def call_payoff(S,K): return np.maximum(S-K,0.0)
    def put_payoff(S,K):  return np.maximum(K-S,0.0)
    credit = condor.credit
    sc = -call_payoff(S_grid, condor.short_call.strike)
    lc =  call_payoff(S_grid, condor.long_call.strike)
    sp = -put_payoff(S_grid,  condor.short_put.strike)
    lp =  put_payoff(S_grid,  condor.long_put.strike)
    return credit + sc + lc + sp + lp
