"""Helpers for constructing and valuing iron condors."""
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from bs import bs_greeks, bs_price


@dataclass
class Leg:
    typ: str
    strike: float
    action: str
    price: float
    delta: float

    @property
    def sign(self) -> int:
        return 1 if self.action.upper() == "BUY" else -1


@dataclass
class Condor:
    short_call: Leg
    long_call: Leg
    short_put: Leg
    long_put: Leg

    @property
    def legs(self) -> Iterable[Leg]:
        return (self.short_call, self.long_call, self.short_put, self.long_put)

    @property
    def credit(self) -> float:
        return (
            self.short_call.price
            - self.long_call.price
            + self.short_put.price
            - self.long_put.price
        )

    @property
    def width(self) -> float:
        call_width = abs(self.long_call.strike - self.short_call.strike)
        put_width = abs(self.short_put.strike - self.long_put.strike)
        return max(call_width, put_width)


def build_iron_condor(
    S: float,
    r: float,
    q: float,
    sigma: float,
    t_years: float,
    target_delta: float = 0.12,
    width: float = 5.0,
) -> Condor:
    """Construct a simplified iron condor around ±2% strikes."""
    # Short strikes anchored ±2% from the underlying; target_delta is a placeholder for future logic.
    short_call_strike = S * (1.0 + 0.02)
    short_put_strike = S * (1.0 - 0.02)

    long_call_strike = short_call_strike + width
    long_put_strike = short_put_strike - width

    short_call_price = bs_price(S, short_call_strike, r, q, sigma, t_years, "C")
    long_call_price = bs_price(S, long_call_strike, r, q, sigma, t_years, "C")
    short_put_price = bs_price(S, short_put_strike, r, q, sigma, t_years, "P")
    long_put_price = bs_price(S, long_put_strike, r, q, sigma, t_years, "P")

    short_call_delta = -bs_greeks(S, short_call_strike, r, q, sigma, t_years, "C").delta
    long_call_delta = bs_greeks(S, long_call_strike, r, q, sigma, t_years, "C").delta
    short_put_delta = -bs_greeks(S, short_put_strike, r, q, sigma, t_years, "P").delta
    long_put_delta = bs_greeks(S, long_put_strike, r, q, sigma, t_years, "P").delta

    return Condor(
        short_call=Leg("C", short_call_strike, "SELL", short_call_price, short_call_delta),
        long_call=Leg("C", long_call_strike, "BUY", long_call_price, long_call_delta),
        short_put=Leg("P", short_put_strike, "SELL", short_put_price, short_put_delta),
        long_put=Leg("P", long_put_strike, "BUY", long_put_price, long_put_delta),
    )


def payoff_iron_condor(condor: Condor, S_grid: np.ndarray) -> np.ndarray:
    """Return payoff of the iron condor at expiry (per contract)."""
    def call_payoff(S: np.ndarray, K: float) -> np.ndarray:
        return np.maximum(S - K, 0.0)

    def put_payoff(S: np.ndarray, K: float) -> np.ndarray:
        return np.maximum(K - S, 0.0)

    credit = condor.credit
    short_call = -call_payoff(S_grid, condor.short_call.strike)
    long_call = call_payoff(S_grid, condor.long_call.strike)
    short_put = -put_payoff(S_grid, condor.short_put.strike)
    long_put = put_payoff(S_grid, condor.long_put.strike)
    return credit + short_call + long_call + short_put + long_put


__all__ = [
    "Leg",
    "Condor",
    "build_iron_condor",
    "payoff_iron_condor",
]
