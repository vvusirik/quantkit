import QuantLib as ql
import matplotlib.pyplot as plt
import time

# Short end
# Deposit rates
depo_maturities = [ql.Period(6, ql.Months), ql.Period(12, ql.Months)]
depo_rates = [5.25, 5.5]

# Bond rates
bond_maturities = [ql.Period(6 * i, ql.Months) for i in range(3, 21)]
bond_rates = [
    5.75,
    6.0,
    6.25,
    6.5,
    6.75,
    6.80,
    7.00,
    7.1,
    7.15,
    7.2,
    7.3,
    7.35,
    7.4,
    7.5,
    7.6,
    7.6,
    7.7,
    7.8,
]


def bootstrap_curve():
    calc_date = ql.Date(15, 1, 2015)
    ql.Settings.instance().evaluationDate = calc_date

    calendar = ql.UnitedStates()
    business_convention = ql.Unadjusted
    day_count = ql.Thirty360()
    end_of_month = True
    settlement_days = 0
    face_amount = 100
    coupon_frequency = ql.Period(ql.Semiannual)
    settlement_days = 0

    depo_helpers = [
        ql.DepositRateHelper(
            ql.QuoteHandle(ql.SimpleQuote(rate / 100.0)),
            mat,
            settlement_days,
            calendar,
            business_convention,
            end_of_month,
            day_count,
        )
        for rate, mat in zip(depo_rates, depo_maturities)
    ]

    bond_helpers = []
    for rate, mat in zip(bond_rates, bond_maturities):
        termination_date = calc_date + mat
        schedule = ql.Schedule(
            calc_date,
            termination_date,
            coupon_frequency,
            calendar,
            business_convention,
            business_convention,
            ql.DateGeneration.Backward,
            end_of_month,
        )
        bond_helpers.append(
            ql.FixedRateBondHelper(
                ql.QuoteHandle(ql.SimpleQuote(face_amount)),
                settlement_days,
                face_amount,
                schedule,
                [rate / 100.0],
                day_count,
                business_convention,
            )
        )

    rate_helpers = depo_helpers + bond_helpers
    yield_curve = ql.PiecewiseCubicZero(calc_date, rate_helpers, day_count)

    # 3. get the spot rates out of the yield curve
    tenors = []
    spot_rates = []
    for d in yield_curve.dates()[1:]:
        yrs = day_count.yearFraction(calc_date, d)
        compounding = ql.Compounded
        freq = ql.Semiannual
        zero_rate: ql.InterestRate = yield_curve.zeroRate(yrs, compounding, freq)
        tenors.append(yrs)

        eq_rate: ql.InterestRate = zero_rate.equivalentRate(
            day_count, compounding, freq, calc_date, d
        )
        spot_rates.append(100 * eq_rate.rate())

    plot_curve(tenors, spot_rates)


def plot_curve(maturities, rates):
    plt.plot(maturities, rates)
    plt.savefig("./plots/yield_curve.png")


if __name__ == "__main__":
    bootstrap_curve()
