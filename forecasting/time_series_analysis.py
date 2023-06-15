from models.arima import ArimaModel

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import STL
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class TimeSeriesAnalysis:
    def __init__(self):
        self.models = [ArimaModel()]

    def check_stationarity(self, data):
        # Perform the Augmented Dickey-Fuller test
        result = adfuller(data)
        # If p-value is less than 0.05, the series is stationary
        print('ADF Statistic: %f' % result[0])
        return result[1] < 0.05

    def make_stationary(self, data):
        # Apply differencing to make the series stationary
        return data.diff().dropna()
    
    def remove_seasonality(self, data, period=12):
        stl = STL(data, period=period)
        res = stl.fit()
        seasonal = res.seasonal
        adjusted = data - seasonal
        return pd.DataFrame(adjusted)
    
    def split_data(self, data, train_percent=.7, validate_percent=.15):
        m = len(data)
        train_end = int(train_percent * m)
        validate_end = int(validate_percent * m) + train_end
        train = data[:train_end]
        validate = data[train_end:validate_end]
        test = data[validate_end:]
        return train, validate, test
    
    def plot_forecast(self, actual, in_sample_predictions, future_predictions):
        # Concatenate the actual and predicted values
        df_pred = pd.DataFrame(index=actual.index, columns=['invoiced_amount', 'y_train_pred'])
        df_pred['invoiced_amount'] = actual
        df_pred['y_train_pred'] = in_sample_predictions
        
        # Concatenate with future predictions
        future_dates = pd.date_range(start=actual.index[-1], periods=len(future_predictions), closed='right')
        df_future = pd.DataFrame(index=future_dates, columns=['invoiced_amount', 'y_train_pred'])
        df_future['invoiced_amount'] = np.nan
        df_future['y_train_pred'] = future_predictions
        df_pred = pd.concat([df_pred, df_future])

        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(df_pred['invoiced_amount'], label='Actual')
        plt.plot(df_pred['y_train_pred'], label='ARIMA Prediction')
        plt.plot(df_pred.iloc[-len(future_predictions):]['y_train_pred'], label='Forecast')
        plt.xlabel('Date')
        plt.ylabel('Invoiced Amount')
        plt.legend(facecolor='white')
        plt.grid()
        plt.show()

    def forecast_timeseries(self, data, column_to_forecast='invoiced_amount'):
        # Check if data is stationary, if not, make it stationary
        if not self.check_stationarity(data[column_to_forecast]):
            season_adjusted = self.remove_seasonality(data[column_to_forecast])
        else:
            season_adjusted = data

        # Split the data into training, validation, and test sets
        train, validate, test = self.split_data(season_adjusted[column_to_forecast])

        # Fit each of the models to the sales data and evaluate performance        
        best_model = None
        best_performance = float('inf')
        for model in self.models:
            model.fit(train)
            performance = model.evaluate(validate)
            if performance < best_performance:
                best_performance = performance
                best_model = model
        
        # Generate sales forecast using the best model
        # Generate sales forecast using the best model
        in_sample_predictions = best_model.model.predict_in_sample(dynamic=False)
        future_predictions = best_model.predict(len(test))
        
        # Plot the forecast
        self.plot_forecast(season_adjusted[column_to_forecast], in_sample_predictions, future_predictions)
        

        
        # Optional: return or plot forecast
        return future_predictions
