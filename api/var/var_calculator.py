from flask import Flask
import time
from urllib import request
import pandas as pd
from typing import Optional, Tuple, Dict
import datetime as dt
import pandas_market_calendars as mcal
import numpy as np

app = Flask(__name__)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0


@app.route("/time")
def currtime():
    return {"time": time.time()}


@app.route("/yield_curves")
def fetch_yield_curves():
    yields = []
    for year in range(2000, 2023):
        url = f"https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv/{year}/all?type=daily_treasury_yield_curve&field_tdr_date_value=2022&page&_format=csv"
        with request.urlopen(url) as f:
            yields.append(pd.read_csv(f))
    yield_df = pd.concat(yields)
    return yield_df.to_html(index=False)


# async def _fetch_annual_yield_curve_data(year: int) -> pd.DataFrame:
#     url = f"https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv/{year}/all?type=daily_treasury_yield_curve&field_tdr_date_value=2022&page&_format=csv"
#     with request.urlopen(url) as f:
#         yields.append(pd.read_csv(f))


def generate_data(date_range: Tuple[dt.date, dt.date]):
    """
    Generate pricing data
    """
    # 1. Generate the bdate range
    nyse = mcal.get_calendar("NYSE")
    start, end = date_range
    bdates = nyse.schedule(start_date=start, end_date=end)

    # 2. Randomly generate the price variable data


def compute_var(
    positions: Dict[str, float],
    price_data: pd.DataFrame,
    horizon_days: float = 10,
    confidence: float = 0.99,
):
    portfolio_value_initial: float = sum(positions.values())
    price_df: pd.DataFrame = price_data[list(positions.keys())]
    returns_df: pd.DataFrame = (price_df / price_df.shift()).dropna()
    portfolio_scenarios: pd.DataFrame = pd.DataFrame(
        data={
            symbol: (dollar_position * returns_df[symbol])
            for symbol, dollar_position in positions.items()
        }
    ).sum(axis=1)
    print(portfolio_scenarios)
    return_scenarios: pd.Series = portfolio_scenarios / portfolio_value_initial
    loss_scenarios: pd.Series = portfolio_value_initial - portfolio_scenarios
    print(loss_scenarios)
    var = loss_scenarios.quantile(1 - confidence)
    n_day_var = var * np.sqrt(horizon_days)
    return n_day_var


if __name__ == "__main__":
    price_data = pd.read_csv("~/Downloads/var_price_data.csv")
    positions = {"SP500": 4000, "FTSE500": 3000, "CAC40": 1000, "NIKKEI": 2000}
    var = compute_var(positions=positions, price_data=price_data)
    print(var)
