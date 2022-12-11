from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import pmdarima as pm
import pandas as pd



conso_elec_2016 = pd.read_csv("eCO2mix_RTE_Annuel-Definitif_2016_selec_ts.csv", index_col="Timestamp", parse_dates=True)

conso_elec_2017 = pd.read_csv("eCO2mix_RTE_Annuel-Definitif_2017_selec_ts.csv", index_col="Timestamp", parse_dates=True)

conso_elec = pd.concat([conso_elec_2016, conso_elec_2017])

conso_elec.index = [pd.Timestamp(i, unit='s') for i in conso_elec.index]

df_forecast = pd.DataFrame()
df_forecast.index = conso_elec.index
df_forecast["Prevision J-1"] = conso_elec["Prevision J-1"]
df_forecast["Prevision J"] = conso_elec["Prevision J"]
df_forecast["Consommation"] = conso_elec["Consommation"]


df_forecast = df_forecast.resample(rule="D").mean()

print(df_forecast.index[-1])
"""abscisse = df_forecast.index
ordonnee = df_forecast["Consommation"]
plt.plot(abscisse, ordonnee)
plt.xlabel("Date")
plt.ylabel("Consommation électrique (MW)")
plt.title("Consommation électrique française sur les années 2016 et 2017 (moyenne par jour)")
plt.show()"""

training_data = list(df_forecast["Consommation"][:-14])


arima_pred = pm.auto_arima(y=training_data,
                      start_p=1, 
                      start_q=1,
                      test='adf', # use adftest to find optimal 'd'
                      max_p=3, max_q=3, # maximum p and q
                      m=7, # frequency of series (if m==1, seasonal is set to FALSE automatically)
                      d=None,# let model determine 'd'
                      seasonal=True, # No Seasonality for standard ARIMA
                      trace=False, #logs 
                      error_action='warn', #shows errors ('ignore' silences these)
                      suppress_warnings=True,
                      stepwise=True)

"""
#arima_pred.plot_diagnostics()
#plt.show()

"""

nb_periods = 14

real_values = pd.read_csv("eCO2mix_RTE_Annuel-Definitif_2018_selec_ts.csv", index_col="Timestamp", parse_dates=True)
real_values.index = [pd.Timestamp(i, unit='s') for i in real_values.index]
real_values = real_values.resample(rule="D").mean()

print(real_values.head())

def forecast(ARIMA_model, periods=nb_periods):
    # Forecast
    n_periods = periods
    fitted, confint = ARIMA_model.predict(n_periods=n_periods, return_conf_int=True)
    index_of_fc = pd.date_range(df_forecast.index[-1] + pd.DateOffset(months=0), periods = n_periods, freq='D')
    print(index_of_fc)
    # make series for plotting purpose
    fitted_series = pd.Series(fitted, index=index_of_fc)
    lower_series = pd.Series(confint[:, 0], index=index_of_fc)
    upper_series = pd.Series(confint[:, 1], index=index_of_fc)

    # Plot
    plt.figure(figsize=(15,7))
    plt.plot(df_forecast["Consommation"], color='#1f76b4')
    plt.plot(fitted_series, color='darkgreen')
    plt.plot(real_values["Consommation"][:nb_periods], color="red")

    plt.fill_between(lower_series.index, 
                    lower_series, 
                    upper_series, 
                    color='k', alpha=.15)

    plt.title("ARIMA/SARIMA - Forecast of Power consumption in France")
    plt.legend(["Données d'entraînement", "Prédiction", "Consommation effective"])
    plt.xlabel("Date")
    plt.ylabel("Consommation électrique (MW)")
    mape = "MAPE = " + str(pm.metrics.smape(real_values["Consommation"][:nb_periods], fitted_series))
    plt.text(x=16822, y=90000, s=mape)
    print(mape)

    plt.show()

forecast(arima_pred)