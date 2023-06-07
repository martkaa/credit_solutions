import pandas as pd
from data_processing.preprocessing import DataPreprocessor, split_data

def main():
    # Read the data first
    data = pd.read_csv('tmp_stitched.csv')
    
    # Pass the DataFrame to the preprocessor
    preprocessor = DataPreprocessor(data)
    
    # Preprocess the data
    processed_data = preprocessor.preprocess()
    
    # Split the data
    train_data, validate_data, test_data = split_data(processed_data)

if __name__ == "__main__":
    main()
