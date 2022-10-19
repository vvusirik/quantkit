from typing import List, Tuple, Dict, Callable
from functools import partial
import numpy as np
import pandas as pd


def compute_bond_price(
    yield_to_maturity: float,
    face_value: float,
    annual_coupon: float,
    periods_per_annum: float,
    term: float,
):
    term_step = 1 / periods_per_annum
    coupon_terms = np.arange(
        start=term_step,
        stop=term + term_step,
        step=term_step,
    )
    price = sum(
        annual_coupon / periods_per_annum * np.e ** -(yield_to_maturity * t)
        for t in coupon_terms
    )
    price += face_value * np.e ** -(yield_to_maturity * term)
    return price


def newton_solver(
    f: Callable,
    x0: float,
    target: float,
    secant_eps: float = 1e-6,
    max_iters: int = 1000,
    threshold: float = 1e-10,
):
    # Solve for f(x) == target, ie construct g(x) = f(x) - target
    # so we can solve for g(x) = 0
    g = lambda x: f(x) - target
    x = x0
    for i in range(1000):
        g_x = g(x)
        g_prime_x = (g(x + secant_eps) - g(x - secant_eps)) / (2 * secant_eps)
        x -= g_x / g_prime_x

        if -threshold <= g_x <= threshold:
            return x

    raise Exception(f"Unable to converge within max iterations ({max_iters})")


def compute_yield_to_maturity(
    bond_price: float,
    face_value: float,
    annual_coupon: float,
    periods_per_annum: float,
    term: float,
) -> float:
    """
    YTM is the anticipated annualized continuous return on a bond if you hold it until maturity.
    B = sum[(c / m) * e * -(y * t_i)] + F * e * -(y * t)
    """
    price_bond = partial(
        compute_bond_price,
        face_value=face_value,
        annual_coupon=annual_coupon,
        periods_per_annum=periods_per_annum,
        term=term,
    )

    # Use current yield as an initial approximation
    guess = compute_current_yield(bond_price=bond_price, annual_coupon=annual_coupon)

    yield_to_maturity = newton_solver(
        price_bond,
        x0=guess,
        target=bond_price,
        secant_eps=1e-6,
        max_iters=1000,
    )

    term_step = 1 / periods_per_annum
    coupon_terms = np.arange(
        start=term_step,
        stop=term + term_step,
        step=term_step,
    )
    inferred_price = sum(
        (annual_coupon / periods_per_annum) * np.e ** -(yield_to_maturity * t)
        for t in coupon_terms
    ) + face_value * np.e ** -(yield_to_maturity * term)

    assert round(bond_price, 2) == round(inferred_price, 2)
    return yield_to_maturity


def compute_current_yield(bond_price: float, annual_coupon: float):
    """
    Current yid represents expected return if the bond owner purchased and held the bond for a year
    y = annual cash inflows / market price
    """
    return annual_coupon / bond_price


def compute_par_yield(
    face_value: float,
    term: int,
    coupons_per_annum: int,
    zero_rates: Dict[float, float],
) -> float:
    """
    Compute the continuous coupon rate required to make the bond sell for face value.
    """
    term_step = 1 / coupons_per_annum
    coupon_terms = np.arange(
        start=term_step,
        stop=term + term_step,
        step=term_step,
    )

    # The difference between the face value and the final face value payment of the bond
    # discounted to present value is the present value of the coupon payments
    coupon_pv = face_value - (face_value * np.e ** (-zero_rates[term] * term))

    # Convert the coupon present value to future value terms by dividing by the cumulative discount factor
    # Annualize the coupon amount by multiplying by coupons per annum
    coupon_discount_factor = sum(np.e ** (-zero_rates[t] * t) for t in coupon_terms)
    coupon_amount = coupons_per_annum * coupon_pv / coupon_discount_factor

    # Check that the coupon amount corresponds to the expected face value
    check_face_value = sum(
        coupon_amount / coupons_per_annum * np.e ** -(zero_rates[t] * t)
        for t in coupon_terms
    ) + (face_value * np.e ** -(zero_rates[term] * term))
    assert face_value == check_face_value

    # Compute and convert the discrete coupon rate to a continuous coupon rate
    coupon_rate = coupon_amount / face_value
    continuous_coupon_rate = compute_continuous_rate(coupon_rate, coupons_per_annum)

    return continuous_coupon_rate


def compute_continuous_rate(discrete_rate: float, periods: int):
    return periods * np.log(1 + discrete_rate / periods)


def compute_discrete_rate(continuous_rate: float, periods: int):
    return periods * (np.e ** (continuous_rate / periods) - 1)


def compute_zero_rates(bond_data: pd.DataFrame) -> pd.DataFrame:
    """
    bond_data["face_value"]
    bond_data["ttm"]
    bond_data["present_value"]
    bond_data["annual_coupon"]
    bond_data["coupons_per_annum"]
    bond_data["zero_rate"] -> not populated
    """

    # Bond data must be sorted with ascending ttm since we incrementally build up the zero rate curve
    assert is_sorted(bond_data["ttm"])

    zero_rates = pd.DataFrame(columns=["zero_rate"], index=bond_data.ttm)

    # Incrementally compute the zero rates
    for idx, row in bond_data.iterrows():

        # If coupon rate == 0, this is already a zero coupon bond, so we can directly compute the zero rate
        if row["annual_coupon"] == 0:
            ttm = row["ttm"]
            zero_rates.loc[ttm, "zero_rate"] = (
                np.log(row["face_value"] / row["present_value"]) / row["ttm"]
            )

        else:
            coupon_amount = row["annual_coupon"] / bond_data["coupons_per_annum"]

            # Discount each coupon cash flow according to the previously computed zero rate
            coupon_cumulative_pv = sum(
                coupon_amount * np.e ** (-zero_rates.loc[t, "zero_rate"] * t)
                for t in np.arange(
                    start=1 / bond_data["coupons_per_annum"],
                    stop=bond_data["t"],
                    step=1 / bond_data["coupons_per_annum"],
                )
            )

            # Zero rate for current bond maturity is determined by rate on its final cash flow (face value + coupon)
            # Since all the other zero rates have been solved for now
            zero_rates.loc[idx, "zero_rate"] = (
                np.log(
                    (row["present_value"] - coupon_cumulative_pv)
                    / (bond_data["face_value"] + coupon_amount)
                )
                / row["ttm"]
            )

    return zero_rates


def is_sorted(sequence):
    return sorted(sequence) == sequence


def compute_forward_rates(zero_rates: Dict[float, float]) -> Dict[float, float]:
    forward_rates = {}
    zero_rates_list = sorted(list(zero_rates.items()), key=lambda x: x[0])
    for ((t1, r1), (t2, r2)) in zip(zero_rates_list[:-1], zero_rates_list[1:]):
        forward_rates[t1] = (t2 * r2 - t1 * r1) / (t2 - t1)
    return forward_rates


if __name__ == "__main__":
    par_yield = compute_par_yield(
        face_value=100,
        term=2,
        coupons_per_annum=2,
        zero_rates={0.5: 0.05, 1: 0.058, 1.5: 0.064, 2: 0.068},
    )
    print(par_yield)

    ytm = compute_yield_to_maturity(
        bond_price=98.39,
        face_value=100,
        annual_coupon=6,
        periods_per_annum=2,
        term=2,
    )
    print(ytm)

    # YTM for a money market bond
    mm_ytm = compute_yield_to_maturity(
        bond_price=100-1.61,
        face_value=100,
        annual_coupon=0,
        periods_per_annum=1,
        term=.25,
    )
    print(mm_ytm)

    price = compute_bond_price(
        yield_to_maturity=.0280,
        face_value=100,
        annual_coupon=0,
        periods_per_annum=1,
        term=1,
    )
    print(price)
