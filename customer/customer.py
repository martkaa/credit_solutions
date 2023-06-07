from statsmodels.tsa.stattools import adfuller
from models.arima import ARIMA
from models.xgboost import XGBoost
from models.random_forest import RandomForest
from models.linear_regression import LinearRegression

class Customer:
    def __init__(self, customer_id, dataset):
        self.customer_id = customer_id
        self.data = self.fetch_data(dataset)
        self.models = [ARIMA(), XGBoost(), RandomForest(), LinearRegression()]

    def fetch_data(self, dataset):
        # Fetch data specific to the customer from the dataset
        return dataset[dataset['customer_id'] == self.customer_id]

    def check_stationarity(self):
        # Perform the Augmented Dickey-Fuller test
        result = adfuller(self.data)
        # If p-value is less than 0.05, the series is stationary
        return result[1] < 0.05

    def make_stationary(self):
        # Apply differencing to make the series stationary
        self.data = self.data.diff().dropna()

    def forecast_sales(self):
        # Initialize models
        models = [ARIMA(), XGBoost(), RandomForest(), LinearRegression()]
        
        # For each model, fit the model to sales data and evaluate performance
        best_model = None
        best_performance = float('inf')
        for model in models:
            model.fit(self.data['sales'])
            performance = model.evaluate(self.data['sales_validation'])
            if performance < best_performance:
                best_performance = performance
                best_model = model
        
        # Generate sales forecast using the best model
        self.sales_forecast = best_model.predict()
        
        # Optional: plot forecast
        best_model.plot_forecast(self.data['sales'])

    def forecast_profitability(self):
        # Forecast profitability using each of the models
        for model in self.models:
            model.fit(self.data['profitability'])
            self.profitability_forecast = model.predict()

    def run_all(self):
        if not self.check_stationarity():
            self.make_stationary()
        self.forecast_sales()
        self.forecast_profitability()
        # Additional methods for forecasting net exposure, customer duration, etc.
        # Also, method to calculate CLV
