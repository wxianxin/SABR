# Steven Wang
# 2017

import read_data

import math
import numpy as np
import pandas as pd
import re
from scipy.optimize import minimize
from scipy.stats import norm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

################################################################################
# import data 
################################################################################
def import_data(file_name):
    df = pd.read_csv(file_name, header=None)
    return df

df = import_data('option.csv')
#df = read_data('quotedata.dat')
################################################################################
# BS IV
################################################################################
def BS_call(S, K, tau, r, sigma):
    d_1 = (sigma * tau**.5)**(-1) * (math.log(S/K) + (r + sigma**2/2)*tau)
    d_2 = d_1 - sigma*tau**.5
    call = norm.cdf(d_1)*S - norm.cdf(d_2)*K*math.exp(-r*tau)
    return call

def IV_obj_func(sigma, S, K, tau, r, price):
    result = (price - BS_call(S, K, tau, r, sigma))**2
    return result

def BS_IV(sigma_0, S, K, tau, r, price):
    bnds = (0,None),
    opt_result = minimize(IV_obj_func, x0=sigma_0, args=(S, K, tau, r, price), bounds=bnds, method='SLSQP', options={'eps':0.001})
    return opt_result.x[0]

################################################################################
# Get Implied Volatility Surface
################################################################################
def market_vol(S, df):
    mkt_vol = pd.DataFrame(np.zeros((5,5)))
    c=-1
    d=0
    mkt_vol_list = []
    for i in range(len(df)):
        c+=1
        if c==5:
            c=0
            d=d+1
        tau = df.iloc[i][0]
        K = df.iloc[i][1]
        call_price = df.iloc[i][2]
        # It is important to use a good initial guess
        initial_guess = (2*math.pi / tau)**.5 * call_price / S
        mkt_vol.iloc[d][c] = (BS_IV(initial_guess, S, K, tau, 0.01, call_price))
        mkt_vol_list.append(BS_IV(initial_guess, S, K, tau, 0.01, call_price))

    return mkt_vol, mkt_vol_list

################################################################################
# SABR
################################################################################
def SABR_vol(alpha, beta, rho, nu, F, K, tau):
    F_mid = (F*K)**.5
    gamma_1 = beta / F_mid
    gamma_2 = beta * (beta - 1) / F_mid**2
    zeta = alpha / (nu*(1-beta)) * (F**(1-beta) - K**(1-beta))
    D = math.log(((1 - 2*rho*zeta + zeta**2)**.5 + zeta - rho) / (1 - rho))
    A = math.log(F/K) / D
    B = (1 + ((2*gamma_2 - gamma_1**2 + F_mid**(-.5))/24 * (nu*F_mid**beta / alpha)**2 + \
              rho*gamma_1 / 4 * nu * F_mid**beta / alpha + \
              (2 - 3*rho**2) / 24 \
             ) * tau)
    sigma_sabr = alpha * A * B
    return sigma_sabr

def SABR_obj_func(params, S, df, IV_list):
    alpha, beta, rho, nu = params
    summ = 0
    for i in range(len(IV_list)):
        F = S
        tau = df.iloc[i][0]
        K = df.iloc[i][1]
        call_price = df.iloc[i][2]

        e = (SABR_vol(alpha, beta, rho, nu, F, K, tau) - IV_list[i])**2
        if math.isnan(e):
            e = 1
        summ += e
    print('sum = ', summ)

    return summ

def Get_SABR_params(S, df, IV_list):
    # improve initial Guess
    initial_guess = [0.0001, 0.5, 0, 0.0001]
    bnds = (0.0001, None), (0, 1), (-.9999, .9999), (0.0001, None)
    #bnds = (0, None), (0, 1), (None, None), (None, None)
    SABR_params = minimize(SABR_obj_func, x0=initial_guess, args=(S, df, IV_list), bounds=bnds, method='SLSQP', options={'eps':0.001})

    return SABR_params.x

################################################################################
# Plot volatility surface
################################################################################
def volatility_surface(kk, tt, z):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    tt, kk = np.meshgrid(tt, kk)
    surf = ax.plot_surface(kk,tt,z,cmap=cm.coolwarm, linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

################################################################################
# Plot comparison of volatility surface
################################################################################
def volatility_surface_comparison(kk, tt, z_1, z_2):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    tt, kk = np.meshgrid(tt, kk)
    surf = ax.plot_surface(kk,tt,z_1,cmap=cm.coolwarm, linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    surf_2 = ax.plot_surface(kk,tt,z_2,cmap=cm.coolwarm, linewidth=0, antialiased=False)
    plt.show()
################################################################################
# Testing
################################################################################
def get_SABR_vol_surface(params, S, df):
    alpha, beta, rho, nu = params
    F = S
    s_vol = pd.DataFrame(np.zeros((5,5)))
    s_vol_list = []
    c=-1
    d=0
    for i in range(len(df)):
        c+=1
        if c==5:
            c=0
            d=d+1
        tau = df.iloc[i][0]
        K = df.iloc[i][1]
        s_vol.iloc[d][c] = (SABR_vol(alpha, beta, rho, nu, F, K, tau))
        s_vol_list.append(SABR_vol(alpha, beta, rho, nu, F, K, tau))

    return s_vol, s_vol_list
# Testing
################################################################################
# Testing
################################################################################
z,IV_list = market_vol(2548.18, df)
params = Get_SABR_params(2548.18, df, IV_list)
print('SABR params', params)
a, b = get_SABR_vol_surface(params, 2548.18, df)
print(z)
print(a)
kk = [2530, 2540, 2550, 2560, 2570]
tt = [0.00274, 0.021918, 0.054795, 0.093151, 0.180822]
#volatility_surface(kk,tt,z)

volatility_surface_comparison(kk,tt,z,a)

