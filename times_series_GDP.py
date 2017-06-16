import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm

#import data
data= pd.read_csv("US-GDPC1.csv")
#data preprocessing
data.index = sm.tsa.datetools.dates_from_range('1947Q1', "2016Q4")
data = data[1]
data["GDPC1"] = np.log(data["GDPC1"])

#visualise the correlation
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(data.values.squeeze(), lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(data, lags=40, ax=ax2)
fig.show()

#performing adf test
adfuller(data.values.ravel(),autolag ="AIC")

#fixing the series
diff_s = (data.values - data.shift(1))
diff_s = diff_s.dropna(axis=0)
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(diff_s.squeeze(), lags=20, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(diff_s.squeeze(), lags=20, ax=ax2)
fig.show()

#perform adf test
adfuller(diff_s.values.ravel(),autolag ="AIC")

#Building the model using an AR4 and MA1 with difference 1
arma_mod20 = sm.tsa.ARIMA(data, (4,1,1)).fit()
arma_mod20.summary()

#predicting the forecast
fig, ax = plt.subplots(figsize=(12, 8))
ax = data.plot(ax=ax)
fig = arma_mod20.plot_predict('2017Q1', '2029Q1', dynamic=True, ax=ax, plot_insample=False)
fig.show()

#evaluating the model
resid = arma_mod20.resid
stats.normaltest(resid)
#visualising the forecast
fig, ax = plt.subplots(figsize=(12, 8))
plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(resid.values.squeeze(), lags=10, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(resid, lags=10, ax=ax2)
fig.show()



