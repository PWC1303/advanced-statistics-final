import pandas as pd 
import statsmodels.api as sm
import numpy as np 
import matplotlib.pyplot as plt
from q3_i_ii import adf_printer

df = pd.read_csv("data/INDPRO.csv")    
start = np.where(df["observation_date"] =="1970-01-01")[0][0]
end = np.where(df["observation_date"] =="2015-12-01")[0][0]
df = df.iloc[start:end]

X = df["INDPRO"]
adf = sm.tsa.adfuller(X)
adf_printer(adf)

# Difference
DeltaX = np.diff(X, n=1)
fig, ax = plt.subplots(figsize = (12, 4))
ax.plot(DeltaX,color ="cornflowerblue")

plt.ylabel("Delta_X")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.savefig("figs/q3_iii_plot")
plt.close()
adf = sm.tsa.adfuller(DeltaX)
adf_printer(adf)




