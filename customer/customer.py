
from forecasting.time_series_analysis import TimeSeriesAnalysis

class Customer:
    def __init__(self, customer_id, dataset):
        self.customer_id = customer_id
        self.data = self.fetch_data(dataset)
        self.time_series_analysis = TimeSeriesAnalysis()

    def fetch_data(self, dataset):
        # Fetch data specific to the customer from the dataset
        # find all rows with the customer_id, print error if there are none
        dataset = dataset[dataset['customer_id'] == self.customer_id]
        if len(dataset) == 0:
            print(f'No data found for customer {self.customer_id}')
        
        # set the index to the period
        dataset.set_index('period', inplace=True)
        return dataset
    
    def forecast_timeseries_customer(self, column_to_forecast='invoiced_amount'):
        # Forecast sales using the time_series_analysis object
        forecast = self.time_series_analysis.forecast_timeseries(
            data=self.data, 
            column_to_forecast=column_to_forecast
        )
        # Here forecast will contain the sales forecast
        return forecast
    
    

    def run_all(self):
        # For example, to forecast sales, you can call:
        sales_forecast = self.forecast_sales()
        # You can include additional methods for forecasting other metrics
        # and processing the results.
        # ...
