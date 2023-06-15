import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pmdarima as pm
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score


class ArimaModel:
    
    def __init__(self, seasonality_period=12):
        self.seasonality_period = seasonality_period
        self.model = None
    
    def fit(self, data):
        self.model = auto_arima(data, test='adf',
                                   max_p=2, max_d=2, max_q=2,
                                   seasonal=True, m=self.seasonality_period,
                                   max_P=3, max_D=2, max_Q=3,
                                   D=1,
                                   trace=True,
                                   error_action='ignore',
                                   suppress_warnings=True,
                                   stepwise=True)
        
    def predict(self, future_periods):
        return self.model.predict(n_periods=future_periods, dynamic=False)
    
    def evaluate(self, validate_data):
        pred_validate = self.model.predict(n_periods=len(validate_data), dynamic=False)
        mae, mse, rmse, mape, _ = self.calculate_metrics(validate_data, pred_validate)
        print(f'MAE: {mae}, MSE: {mse}, RMSE: {rmse}, MAPE: {mape}')
        return mae  # or another metric that you consider for model evaluation
        
    @staticmethod
    def calculate_metrics(actual, predictions):
        mae = mean_absolute_error(actual, predictions)
        mse = mean_squared_error(actual, predictions)
        rmse = np.sqrt(mse)
        mape = mean_absolute_percentage_error(actual, predictions) * 100
        r2 = r2_score(actual, predictions)
        return mae, mse, rmse, mape, r2



# Example usage:
# arima = ArimaModel()
# arima.fit(df_sales['invoiced-amount'])
# df_sales = arima.in_sample_forecast(df_sales)
# arima.plot_forecast(df_sales, future_periods=24)
