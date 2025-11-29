import pandas as pd 
import statsmodels.api as sm
import numpy as np 
import matplotlib.pyplot as plt
df = pd.read_csv("data/INDPRO.csv")    
start = np.where(df["observation_date"] =="1970-01-01")[0][0]
end = np.where(df["observation_date"] =="2015-12-01")[0][0]
df = df.iloc[start:end]
fig, ax =   plt.subplots()



plt.plot(df["observation_date"],df["INDPRO"], color  = "cornflowerblue")
plt.ylabel("INDPRO")
plt.xlabel("Observation Date")
plt.xticks(plt.xticks()[0][::100], fontsize=8)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.savefig("figs/q3_i_plot")
plt.close()


def adf_printer(adf):
    adf_stat, pvalue, lags, nobs, crit, icbest = adf
    print(f"Adf statistic{adf_stat}")
    print(f"p-value {pvalue}")
    print(f"Lags used {lags}")
    print(f"Number of obs. {nobs}")


    
Y =df["INDPRO"]
adf = sm.tsa.adfuller(Y)

adf_printer(adf)


