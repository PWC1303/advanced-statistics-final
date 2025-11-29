
import pandas as pd 
import statsmodels.api as sm
import numpy as np 
import matplotlib.pyplot as plt


df = pd.read_csv("data/INDPRO.csv")    
start = np.where(df["observation_date"] =="1970-01-01")[0][0]
end = np.where(df["observation_date"] =="2015-12-01")[0][0]
df = df.iloc[start:end]

df["observation_date"] = pd.to_datetime(df["observation_date"])
df = df.set_index("observation_date")
df = df.asfreq("MS")


X = df["INDPRO"]
DeltaX = X.diff(1).dropna()

arma11 = sm.tsa.ARIMA(DeltaX, order = (1,0,1)).fit(method = "innovations_mle")

Xhat = arma11.forecast(steps = 4)

Xhat.index = pd.date_range(start=DeltaX.index[-1] + pd.offsets.MonthBegin(1),
                           periods=4, freq="MS")



Xhat = pd.concat((DeltaX[-1:], Xhat))


fig, ax = plt.subplots(figsize = (12, 4))
ax.plot(DeltaX,color = "cornflowerblue")


ax.plot(Xhat, color = "magenta",linestyle ="dashed")
plt.savefig("figs/q3_vii_Delta_INDPRO_forecast.png")




