from scipy.interpolate import CubicSpline
from typing import Sequence, Tuple, Optional, Union
from dataclasses import dataclass
import datetime as dt
from .rate_solvers import compute_yield_to_maturity

Term = float
Yield = float

def _xor(a: bool, b: bool) -> bool:
    return a != b


@dataclass
class BondPriceData:
    price: float
    maturity: Union[Term, dt.date]
    coupon: float
    face_value: float = 100
    periods_per_annum: float = 2


class YieldCurve(object):
    def __init__(
        self,
        yields: Optional[Sequence[Tuple[Term, Yield]]] = None,
        prices: Optional[Sequence[Tuple[Term, BondPriceData]]] = None,
    ):
        assert (yields is not None) != (prices is not None)

        if yields:
            term, yields = zip(*yields)

        if prices:
            term, prices = zip(*prices)
            yields = [
                compute_yield_to_maturity(
                    bond_price=pdata.price,
                    face_value=pdata.face_value,
                    annual_coupon=pdata.coupon,
                    periods_per_annum=pdata.periods_per_annum,
                    term=pdata.maturity,
                )
                for pdata in prices
            ]
        self.spline = CubicSpline(term, yields)

    def __call__(self, term: Term):
        return self.spline(term)
