# http://www.cboe.com/delayedquote/quote-table-download
import numpy as np
import pandas as pd
import re
from datetime import datetime


def read_data(file_name):
    month = {
        "A": 1,
        "B": 2,
        "C": 3,
        "D": 4,
        "E": 5,
        "F": 6,
        "G": 7,
        "H": 8,
        "I": 9,
        "J": 10,
        "K": 11,
        "L": 12,
    }

    with open(file_name) as file_obj:
        f = file_obj.readline()
        symbol, spot_price = f.split()[0], float(f.split()[3].split(",")[1])
        print(symbol, spot_price)

    df = pd.read_csv("quotedata.dat", skiprows=[0, 1])
    ndf = pd.DataFrame(np.zeros((len(df), 3)), columns=["tau", "price", "strike"])
    today = datetime.now()
    for i in range(len(df)):

        ticker = re.findall("\d{4}\w\d{3,4}", df.iloc[i][0])[0]
        yy = int(ticker[:2]) + 2000
        dd = int(ticker[2:4])
        m = month[ticker[4]]
        K = int(ticker[5:])

        expiry = (datetime(yy, m, dd) - today).days
        if expiry in [-1, 0]:
            expiry = 1
        tau = expiry / 365
        if re.search("SPXW.*E", df.iloc[i][0]):
            ndf.loc[i] = [tau, (df["Bid"][i] + df["Ask"][i]) / 2, K]

    drop_index = []
    for i in range(len(ndf)):
        if (ndf["strike"][i] - spot_price) ** 2 > (0.02 * spot_price) ** 2:
            drop_index.append(i)

    ndf = ndf.drop(drop_index).reset_index(drop=True)
    ndf = ndf.drop(ndf[ndf["price"] == 0].index)

    # ndf = ndf.iloc[322:]
    return symbol, spot_price, ndf


# symbol, spot_price, ndf = read_data('quotedata.dat')
# print(ndf.head())
