################################################################################
# Testing
################################################################################
import matplotlib.pyplot as plt
import sabr

symbol, spot_price, df = sabr.import_data("quotedata.dat")
IV_list = sabr.get_IV(spot_price, r=0.013, df=df)
params = sabr.get_sabr_params(spot_price, df, IV_list)

tt = df["tau"].unique()
kk = df["strike"].unique()
# tt = [i/100 for i in range(150)]
# kk = [i for i in range(int(spot_price * 0.8), int(spot_price * 1.2))]
print(tt)
print(kk)

sabr_vol_list, sabr_vol_df = sabr.get_sabr_vol_surface(params, spot_price, tt, kk)
sabr_vol_list_1 = sabr.get_sabr_vol_surface_1(params, spot_price, df)

print(symbol, spot_price, "sabr params: ", params)
sabr.volatility_surface(spot_price, tt, kk, sabr_vol_df)
plt.plot(IV_list, label="IV")
plt.xlim((0, len(df)))
plt.plot(sabr_vol_list_1, label="old")
plt.plot(sabr_vol_list, label="new")
plt.legend()
plt.show()
# volatility_surface_comparison(kk,tt,z,a)

print(sabr.sabr_vol(params[0], params[1], params[2], params[3], 1200, 1300, 0.5))
