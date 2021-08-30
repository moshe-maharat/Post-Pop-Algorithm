# -*- coding: utf-8 -*-
"""
Created on Tue May  4 13:35:54 2021

@authors: Moshe Maharat & Avraham Pakada
"""
#install programs
#run line 9 and then line 10
conda install -c conda-forge pystan
conda install -c conda-forge fbprophet

import pandas as pd
from pandas import read_csv

#prediction for 2 hours after

# load the dataset
url1 = 'https://drive.google.com/file/d/1FNNVIFv4FCZSUdKVIim4CTkIKzQLZoBf/view?usp=sharing'
df1 = pd.read_csv('https://drive.google.com/uc?export=download&id='+url1.split('/')[-2])
df1.head()

#set date format & drop other columns
df1['Hour'] = df1['Time'].apply(lambda x: str(x)[:2])
df1['Mintue'] = df1['Time'].apply(lambda x: str(x)[-2:])
df1['Time & Date'] = pd.DatetimeIndex(df1['Hour']+':'+df1['Mintue'])
df1 = df1.drop(['Date','Time','Hour','Mintue'],axis = 1)
df1.head()

#name new columns
df1.columns = ['y', 'ds']
df1.head()


from fbprophet import Prophet
# define the model
m1 = Prophet(daily_seasonality=True,changepoint_prior_scale=0.001, seasonality_mode='multiplicative', seasonality_prior_scale=0.1, interval_width=0.95)
# fit prophet model on the dataset
m1.fit(df1)


future1 = m1.make_future_dataframe(periods=120,freq='min')
# use the model to make a forecast
forecast1 = m1.predict(future1)
forecast1.head()
#show relevant columns
#run next line separately
forecast1[['ds','yhat']]

#show forecast graphs
plot_two_hours_1 = m1.plot(forecast1, xlabel='Time & Date', ylabel='Popularity Percentage', uncertainty=False)
plot_two_hours_1B = m1.plot_components(forecast1, uncertainty=False)

############################################################################################################################################################################

#prediction for one day after

# load the dataset
url2 = 'https://drive.google.com/file/d/1Kef1d61YOU4IG2gtiZW6Wi8Vrr5EMum7/view?usp=sharing'
df2 = pd.read_csv('https://drive.google.com/uc?export=download&id='+url2.split('/')[-2])
df2.head()

#set date format & drop other columns
df2['Hour'] = df2['Time'].apply(lambda x: str(x)[:2])
df2['Mintue'] = df2['Time'].apply(lambda x: str(x)[-2:])
df2['Time & Date'] = pd.DatetimeIndex(df2['Hour']+':'+df2['Mintue'])
df2 = df2.drop(['Date','Time','Hour','Mintue'],axis = 1)
df2.head()

#name new columns
df2.columns = ['y', 'ds']
df2.head()


from fbprophet import Prophet
# define the model
m2 = Prophet(daily_seasonality=True, changepoint_prior_scale=0.006, seasonality_mode='multiplicative', seasonality_prior_scale=0.90, interval_width=0.95)
# fit prophet model on the dataset
m2.fit(df2)


future2 = m2.make_future_dataframe(periods=1500,freq='min')
# use the model to make a forecast
forecast2 = m2.predict(future2)
forecast2.head()
#show relevant columns
#run next line separately
forecast2[['ds','yhat']]

#show forecast graphs
plot_one_day_2 = m2.plot(forecast2, xlabel='Time & Date', ylabel='Popularity Percentage', uncertainty=False)
plot_one_day_2B = m2.plot_components(forecast2, uncertainty=False)


############################################################################################################################################################################
#prediction for one week after

# load the dataset
url3 = 'https://drive.google.com/file/d/1Yn3hn16tcKl38UDNjc8m8kw08AtIGz6C/view?usp=sharing'
df3 = pd.read_csv('https://drive.google.com/uc?export=download&id='+url3.split('/')[-2])
df3.head()

#set date format & drop other columns
df3['Hour'] = df3['Time'].apply(lambda x: str(x)[:2])
df3['Mintue'] = df3['Time'].apply(lambda x: str(x)[-2:])
df3['Time & Date'] = pd.DatetimeIndex(df3['Hour']+':'+df3['Mintue'])
df3 = df3.drop(['Date','Time','Hour','Mintue'],axis = 1)
df3.head()

#name new columns
df3.columns = ['y', 'ds']
df3.head()


from fbprophet import Prophet
# define the model
m3 = Prophet(daily_seasonality=True, changepoint_prior_scale=0.006, seasonality_mode='multiplicative', seasonality_prior_scale=0.90, interval_width=0.95)
# fit prophet model on the dataset
m3.fit(df3)


future3 = m3.make_future_dataframe(periods=10080,freq='min')
# use the model to make a forecast
forecast3 = m3.predict(future3)
forecast3.head()
#show relevant columns
#run next line separately
forecast3[['ds','yhat']]

#show forecast graphs
plot_one_day_3 = m3.plot(forecast3, xlabel='Time & Date', ylabel='Popularity Percentage', uncertainty=False)
plot_one_day_3B = m3.plot_components(forecast3, uncertainty=False)

############################################################################################################################################################################
#creating df4 of one week only for MSE

# load the dataset
url4 = 'https://drive.google.com/file/d/1TMwaPncCSaJxN-IMXr_QeeWOmf7d6owV/view?usp=sharing'
df4 = pd.read_csv('https://drive.google.com/uc?export=download&id='+url4.split('/')[-2])
df4.head()

#set date format & drop other columns
df4['Hour'] = df4['Time'].apply(lambda x: str(x)[:2])
df4['Mintue'] = df4['Time'].apply(lambda x: str(x)[-2:])
df4['Time & Date'] = pd.DatetimeIndex(df4['Hour']+':'+df4['Mintue'])
df4 = df4.drop(['Date','Time','Hour','Mintue'],axis = 1)
df4.head()

#name new columns
df4.columns = ['y', 'ds']
df4.head()


from fbprophet import Prophet
# define the model
m4 = Prophet()
# fit prophet model on the dataset
m4.fit(df4)
######################################################################################################################################################################

#MSE for fisrt prediction(2 hours after)
#making cross validation
from fbprophet.diagnostics import cross_validation
df_cv1 = cross_validation(m2, initial = pd.to_timedelta(0,unit="Min"),horizon = pd.to_timedelta(118,unit="Min"))
df_cv1.head()

#creating MSE array
from fbprophet.diagnostics import performance_metrics
df_p1 = performance_metrics(df=df_cv1, rolling_window=0.1)
df_p1.head()
#run separately
df_p1[['horizon','mse']]

#show MSE graphs
from fbprophet.plot import plot_cross_validation_metric
fig1 = plot_cross_validation_metric(df_cv=df_cv1, metric='mse', rolling_window=0.1)

######################################

#MSE for second prediction(1 day)
#making cross validation
df_cv2 = cross_validation(m3, initial = pd.to_timedelta(1,unit="Min"), horizon = pd.to_timedelta(720,unit="Min"))
df_cv2.head()

#creating MSE array
df_p2 = performance_metrics(df=df_cv2, rolling_window=0.1)
df_p2.head()
#run separately
df_p2[['horizon','mse']]

#show MSE graphs
fig2 = plot_cross_validation_metric(df_cv=df_cv2, metric='mse', rolling_window=0.1)
######################################

#MSE for third prediction(1 week)
#making cross validation
df_cv3 = cross_validation(m4, initial = pd.to_timedelta(0,unit="Min"), horizon= pd.to_timedelta(1385,unit="Min"))
df_cv3.head()

#creating MSE array
df_p3 = performance_metrics(df=df_cv3, rolling_window=0.1)
df_p3.head()
#run separately
df_p3[['horizon','mse']]

#show MSE graphs
fig3 = plot_cross_validation_metric(df_cv=df_cv3, metric='mse', rolling_window=0.1)

