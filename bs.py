"""Black-Scholes pricing utilities for the Streamlit Iron Condor desk."""
from dataclasses import dataclass
from math import erf, exp, log, pi, sqrt


@dataclass(frozen=True)
class Greeks:
    """Container for the primary option greeks."""
    delta: float
    gamma: float
    theta: float
    vega: float


def _norm_pdf(x: float) -> float:
    return exp(-0.5 * x * x) / sqrt(2.0 * pi)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def _intrinsic_value(S: float, K: float, typ: str) -> float:
    if typ.upper() == "C":
        return max(S - K, 0.0)
    if typ.upper() == "P":
        return max(K - S, 0.0)
    raise ValueError("Option type must be 'C' for call or 'P' for put")


def bs_price(S: float, K: float, r: float, q: float, sigma: float, t: float, typ: str = "C") -> float:
    """Return the Black-Scholes price for a European call or put."""
    typ_u = typ.upper()
    if t <= 0.0 or sigma <= 0.0:
        return _intrinsic_value(S, K, typ_u)

    sqrt_t = sqrt(t)
    d1 = (log(S / K) + (r - q + 0.5 * sigma * sigma) * t) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    disc_r = exp(-r * t)
    disc_q = exp(-q * t)

    if typ_u == "C":
        return S * disc_q * _norm_cdf(d1) - K * disc_r * _norm_cdf(d2)
    if typ_u == "P":
        return K * disc_r * _norm_cdf(-d2) - S * disc_q * _norm_cdf(-d1)
    raise ValueError("Option type must be 'C' for call or 'P' for put")


def bs_greeks(S: float, K: float, r: float, q: float, sigma: float, t: float, typ: str) -> Greeks:
    """Return the Black-Scholes greeks (delta, gamma, theta, vega)."""
    typ_u = typ.upper()
    if t <= 0.0 or sigma <= 0.0:
        intrinsic_delta = 0.0
        if typ_u == "C":
            intrinsic_delta = 1.0 if S > K else 0.0
        elif typ_u == "P":
            intrinsic_delta = -1.0 if S < K else 0.0
        else:
            raise ValueError("Option type must be 'C' for call or 'P' for put")
        return Greeks(delta=intrinsic_delta, gamma=0.0, theta=0.0, vega=0.0)

    sqrt_t = sqrt(t)
    d1 = (log(S / K) + (r - q + 0.5 * sigma * sigma) * t) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    disc_r = exp(-r * t)
    disc_q = exp(-q * t)
    pdf = _norm_pdf(d1)

    gamma = disc_q * pdf / (S * sigma * sqrt_t)
    vega = S * disc_q * pdf * sqrt_t
    theta_common = -S * disc_q * pdf * sigma / (2.0 * sqrt_t)

    if typ_u == "C":
        delta = disc_q * _norm_cdf(d1)
        theta = theta_common + q * S * disc_q * _norm_cdf(d1) - r * K * disc_r * _norm_cdf(d2)
    elif typ_u == "P":
        delta = disc_q * (_norm_cdf(d1) - 1.0)
        theta = theta_common - q * S * disc_q * _norm_cdf(-d1) + r * K * disc_r * _norm_cdf(-d2)
    else:
        raise ValueError("Option type must be 'C' for call or 'P' for put")

    return Greeks(delta=delta, gamma=gamma, theta=theta, vega=vega)


__all__ = ["bs_price", "bs_greeks", "Greeks"]
