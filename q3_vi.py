
import pandas as pd 
import statsmodels.api as sm
import numpy as np 

df = pd.read_csv("data/INDPRO.csv")    
start = np.where(df["observation_date"] =="1970-01-01")[0][0]
end = np.where(df["observation_date"] =="2015-12-01")[0][0]
df = df.iloc[start:end]

X = df["INDPRO"]
DeltaX = np.diff(X,n=1)


arma11 = sm.tsa.ARIMA(DeltaX, order = (1,0,1)).fit(method = "innovations_mle")
print(arma11.summary())