import pandas as pd
from data_processing.preprocessing import DataPreprocessor, split_data

def main():
    # Read the data first

    
    # Pass the DataFrame to the preprocessor
    preprocessor = DataPreprocessor('tmp_stitched.csv')
    
    # Preprocess the data
    processed_data = preprocessor.preprocess()

    print(processed_data.head())
    
    # Split the data
    #train_data, validate_data, test_data = split_data(processed_data)

if __name__ == "__main__":
    main()
