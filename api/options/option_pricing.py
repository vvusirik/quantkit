import QuantLib as ql
import time
import pandas as pd

# 1. create the instrument
option = ql.VanillaOption(
    ql.PlainVanillaPayoff(ql.Option.Call, strike=100),
    ql.EuropeanExercise(date=ql.Date(30, 8, 2022)),
)

# 2. create handles for the inputs
today = ql.Date(22, 8, 2022)
riskFreeTS = ql.YieldTermStructureHandle(
    ql.FlatForward(today, 0.05, ql.Actual365Fixed())
)
dividendTS = ql.YieldTermStructureHandle(
    ql.FlatForward(today, 0.01, ql.Actual365Fixed())
)
volatility = ql.BlackVolTermStructureHandle(
    ql.BlackConstantVol(today, ql.NullCalendar(), 0.1, ql.Actual365Fixed())
)

und = ql.SimpleQuote(100)

# 3. set up the pricing engine
process = ql.BlackScholesMertonProcess(
    ql.QuoteHandle(und), dividendTS, riskFreeTS, volatility
)
engine = ql.AnalyticEuropeanEngine(process)

option.setPricingEngine(engine)
print(option.NPV())


def price_options(und, option):
    def price_for_und(und_price: float):
        und.setValue(und_price)
        return option.NPV()

    res = pd.Series(range(1, 1_000_000)).apply(price_for_und)
    return res


s = time.time()
res = price_options(und, option)
e = time.time()
print(e - s)
