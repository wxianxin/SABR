"""
SABR model 
"""

__author__ = "Steven Wang"
__email__ = "github.com/wxianixn"

import math
import numpy as np
import pandas as pd
from scipy import optimize
from scipy.stats import norm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import read_data

################################################################################
# import data
def import_data(file_name):
    symbol, spot_price, df = read_data.read_data(file_name)
    return symbol, spot_price, df


################################################################################
# BS IV
def BS_call(S, K, tau, r, sigma):
    """Calculate Call option price based BS.
    Args:
        S (float): Spot price
        K (float): Strike price
        tau (float): Time to maturiry
        r (float): Risk-free rate
        sigma (float): Volatitlity

    Returns:
        float: call option price
    """
    d_1 = (math.log(S / K) + (r + sigma ** 2 / 2) * tau) / (sigma * tau ** 0.5)
    d_2 = d_1 - sigma * tau ** 0.5
    call_price = norm.cdf(d_1) * S - norm.cdf(d_2) * K * math.exp(-r * tau)
    return call_price


def IV_objective_func(sigma, S, K, tau, r, price):
    """Objective function used for optimization in getting Implied Volatility.
    """
    result = (price - BS_call(S, K, tau, r, sigma)) ** 2
    return result


def BS_IV(sigma_0, S, K, tau, r, price):
    bnds = ((0, None),)
    # brent is the better method
    opt_result = optimize.brent(IV_objective_func, args=(S, K, tau, r, price))

    return opt_result


################################################################################
# Get Implied Volatility Surface
# !!!!!!!!!!!!!!!!! IV not correct
def get_IV(S, r, df):
    """
    Args:
        S (float): Spot price
        r (float): Risk-free rate
        df (float): Option data with 3 columns: tau, option trading price, strike price
    Returns:
        list: A list of implied volatility
    """
    mkt_vol_list = []
    for i in range(len(df)):
        tau = df.iloc[i]["tau"]
        K = df.iloc[i]["strike"]
        call_price = df.iloc[i]["price"]
        # It is important to use a good initial guess
        initial_guess = (2 * math.pi / tau) ** 0.5 * call_price / S
        mkt_vol_list.append(BS_IV(initial_guess, S, K, tau, r, call_price))

    return mkt_vol_list


################################################################################
# sabr
def sabr_vol(alpha, beta, rho, nu, spot_price, K, tau):
    """
    Args:
        alpha (float):
        beta (float):
        rho (float):
        nu (float):
        spot_price (float):
    """
    mid = (spot_price * K) ** 0.5
    # epsilon ???
    epsilon = tau
    zeta = alpha / (nu * (1 - beta)) * (spot_price ** (1 - beta) - K ** (1 - beta))
    gamma_1 = beta / mid
    gamma_2 = beta * (beta - 1) / mid ** 2
    D = math.log(((1 - 2 * rho * zeta + zeta ** 2) ** 0.5 + zeta - rho) / (1 - rho))
    if D == 0:
        print(
            "!!!!!!D is 0; it could be that the spot price is too close to strike price!!!"
        )
        print(spot_price ** (1 - beta) - K ** (1 - beta))
        print(zeta)
        D = 0.000001
    A = alpha * math.log(spot_price / K) / D
    B = (
        (2 * gamma_2 - gamma_1 ** 2 + mid ** (-2))
        / 24
        * (nu * mid ** beta / alpha) ** 2
        + rho * gamma_1 / 4 * nu * mid ** beta / alpha
        + (2 - 3 * rho ** 2) / 24
    )

    sigma = A * (1 + B * epsilon)

    if mid < 0.001:
        print("mid: ", mid)

    return sigma


def sabr_obj_func(params, S, df, IV_list):
    alpha, beta, rho, nu = params
    summ = 0
    for i in range(len(IV_list)):
        spot_price = S
        tau = df.iloc[i]["tau"]
        K = df.iloc[i]["strike"]
        call_price = df.iloc[i]["price"]

        e = (sabr_vol(alpha, beta, rho, nu, spot_price, K, tau) - IV_list[i]) ** 2
        # if math.isnan(e):
        # why 100 not 0? After compare the 0 and 100, 100 'feels' better
        #   e = 100
        summ += e

    print("sum = ", summ)
    return summ


def get_sabr_params(S, df, IV_list):
    # improve initial Guess
    initial_guess = [0.0001, 0.5, 0, 0.0001]
    bnds = (0.0001, None), (0, 1), (-0.9999, 0.9999), (0.0001, 0.9999)
    sabr_params = optimize.minimize(
        sabr_obj_func,
        x0=initial_guess,
        args=(S, df, IV_list),
        bounds=bnds,
        method="SLSQP",
        options={"eps": 0.001},
    )

    return sabr_params.x


#################################################################################
## Get sabr volatility
#################################################################################
def get_sabr_vol_surface(params, spot_price, tt, kk):
    """Get surface vol of given tau and K
    """
    alpha, beta, rho, nu = params
    spot_price = spot_price
    sabr_vol_list = []
    for tau in tt:
        for K in kk:
            sabr_vol_list.append(sabr_vol(alpha, beta, rho, nu, spot_price, K, tau))

    sabr_vol_df = pd.DataFrame(np.array(sabr_vol_list).reshape(len(tt), len(kk)))

    return sabr_vol_list, sabr_vol_df


#################################################################################
## Get sabr volatility 1
#################################################################################
def get_sabr_vol_surface_1(params, spot_price, df):
    """Get a list of vol of original data
    """
    alpha, beta, rho, nu = params
    spot_price = spot_price
    sabr_vol_list = []
    for i in range(len(df)):
        tau = df.iloc[i]["tau"]
        K = df.iloc[i]["strike"]
        sabr_vol_list.append(sabr_vol(alpha, beta, rho, nu, spot_price, K, tau))

    return sabr_vol_list


################################################################################
# Plot volatility surface
################################################################################
def volatility_surface(spot_price, kk, tt, z):
    # kk = [x/spot_price for x in kk]
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    tt, kk = np.meshgrid(tt, kk)
    surf = ax.plot_surface(kk, tt, z, cmap=cm.seismic, linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()


################################################################################
# Plot comparison of volatility surface
################################################################################
def volatility_surface_comparison(kk, tt, z_1, z_2):
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    tt, kk = np.meshgrid(tt, kk)
    surf = ax.plot_surface(kk, tt, z_1, cmap=cm.seismic, linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)

    # TODO
    # blahblah

    plt.show()
