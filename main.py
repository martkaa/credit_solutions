import pandas as pd
from data_processing.preprocessing import DataPreprocessor
from customer.customer import Customer


pd.set_option('display.max_columns', None)

def main():
    

    # Pass the DataFrame to the preprocessor
    preprocessor = DataPreprocessor('tmp_stitched.csv')
    
    # Preprocess the data
    processed_data, aggregated_data = preprocessor.preprocess()

    print(processed_data.head())

    # Assume dataset is your DataFrame containing customer data
    customer = Customer(customer_id=16563, dataset=processed_data)
    print(customer.data.head())
    sales_forecast = customer.forecast_timeseries_customer()


    
    # Split the data
    #train_data, validate_data, test_data = split_data(processed_data)

if __name__ == "__main__":
    main()

