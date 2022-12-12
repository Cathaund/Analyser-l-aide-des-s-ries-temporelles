import random
import matplotlib.pyplot as plt
import pmdarima as pm
import pandas as pd
print(help(random.random))

df = pd.DataFrame()
df["gaussian"] = [random.gauss(3, 4) for i in range(1000)]

training_data = df["gaussian"]

arima_pred = pm.auto_arima(y=training_data,
                      start_p=1, 
                      start_q=1,
                      test='adf', # use adftest to find optimal 'd'
                      max_p=3, max_q=3, # maximum p and q
                      m=0, # frequency of series (if m==1, seasonal is set to FALSE automatically)
                      d=None,# let model determine 'd'
                      seasonal=False, # No Seasonality for standard ARIMA
                      trace=False, #logs 
                      error_action='warn', #shows errors ('ignore' silences these)
                      suppress_warnings=True,
                      stepwise=True)


arima_pred.plot_diagnostics()
plt.show()



df["aleatoire"] = [random.random() for i in range(1000)]

training_data = df["aleatoire"]

arima_pred = pm.auto_arima(y=training_data,
                      start_p=1, 
                      start_q=1,
                      test='adf', # use adftest to find optimal 'd'
                      max_p=3, max_q=3, # maximum p and q
                      m=0, # frequency of series (if m==1, seasonal is set to FALSE automatically)
                      d=None,# let model determine 'd'
                      seasonal=False, # No Seasonality for standard ARIMA
                      trace=False, #logs 
                      error_action='warn', #shows errors ('ignore' silences these)
                      suppress_warnings=True,
                      stepwise=True)


arima_pred.plot_diagnostics()
plt.show()
print(max(list(df["aleatoire"])))