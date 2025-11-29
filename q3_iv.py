
import pandas as pd 
import statsmodels.api as sm
import numpy as np 
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf


df = pd.read_csv("data/INDPRO.csv")    
start = np.where(df["observation_date"] =="1970-01-01")[0][0]
end = np.where(df["observation_date"] =="2015-12-01")[0][0]
df = df.iloc[start:end]

X = df["INDPRO"]
DeltaX = np.diff(X, n=1)
plt.figure()
plot_acf(DeltaX)
plt.savefig("figs/q3_iv_acf_plot")
plt.figure()
plot_pacf(DeltaX)
plt.savefig("figs/q3_iv_pacf_plot")



