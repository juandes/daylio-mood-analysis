"""
This script fits a time series model using my Fitbit steps data.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from fbprophet import Prophet

# setting the Seaborn aesthetics.
sns.set()

df = pd.read_csv('../data/prophet_timeseries_data.csv')

# the trend line is a bit underfit, so I'll increase changepoint_prior_scale
# to 0.06 (from 0.05).
m = Prophet(changepoint_prior_scale=0.8)
m.fit(df)
forecast = m.predict(df)
# forecast.to_csv("forecast.csv")
fig = m.plot_components(forecast, figsize=(20, 16))
# this plot shows the daily, weekly and trend seasonality.
plt.show()
plt.savefig('../plots/prophet_timeseries_plot.png')
