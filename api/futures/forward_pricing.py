import numpy as np
from typing import List, Tuple


def price_known_income_forward(
    spot: float,
    risk_free_rate: float,
    tte: float,
    cash_flows: List[Tuple[float, float, float]],
):
    """
    Prices a forward contract when there are known income cash flows.
    Each cash flow is a triplet of amount, zero rate, and time.
    I = sum(c_i * e^-(r_i * t_i))
    F_0 = (S_0 - I) * e^(r * t)
    """
    known_income = sum(
        cash_flow_income * np.e ** (-cash_flow_rate * cash_flow_time)
        for cash_flow_income, cash_flow_rate, cash_flow_time in cash_flows
    )
    forward_price = (spot - known_income) * np.e ** (risk_free_rate * tte)
    return forward_price


def price_known_yield_forward(spot, income, risk_free_rate, known_yield, tte):
    """
    Prices a forward contract when there is known yield income.
    Ie when there is income that is a percentage of the spot asset.
    Yield is expected to be the yield per annum with continuous compounding.
    """
    forward_price = (spot - income) * np.e ** ((risk_free_rate - known_yield) * tte)
    return forward_price
