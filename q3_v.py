
import pandas as pd 
import statsmodels.api as sm
import numpy as np 
import matplotlib.pyplot as plt


df = pd.read_csv("data/INDPRO.csv")    
start = np.where(df["observation_date"] =="1970-01-01")[0][0]
end = np.where(df["observation_date"] =="2015-12-01")[0][0]
df = df.iloc[start:end]

X = df["INDPRO"]
DeltaX = np.diff(X,n=1)

selecter = sm.tsa.arma_order_select_ic(DeltaX, ic = ["aicc", "bic"],
                             max_ar = 4, max_ma = 4,
                               fit_kw = {"method" : "innovations_mle"}
                               )

print(selecter)