import pandas as pd
import numpy as np

class DataPreprocessor:
    def __init__(self, filename):
        self.df = self._read_data(filename, encoding='ISO-8859-1', sep=';')
        self.df = self._format_columns()
        self.df = self._process_dates()
        self.df = self._process_numeric_cols()
        self.df = self._process_categorical_cols()
        self.df = self._remove_zero_invoiced_customers()

    def _read_data(self, filename):
        data = pd.read_csv(filename, encoding='ISO-8859-1', sep=';')
        return data

    def _format_columns(self):
        self.df.columns = [col.replace('-', '_').replace(',', '.') for col in self.df.columns]
        return self.df

    def _process_dates(self):
        for col in ['period', 'start_date', 'end_date']:
            self.df.loc[:, col] = pd.to_datetime(self.df[col], format='%Y-%m', errors='coerce')
        return self.df

    def _process_numeric_cols(self):
        for col in ['invoiced_amount', 'cost', 'other_costs']:
            self.df.loc[:, col] = self.df[col].str.replace(',', '.').astype(float)
        return self.df

    def _process_categorical_cols(self):
        self.df.replace('-', np.nan, inplace=True)
        self.df['risk_segment'] = self.df['risk_segment'].astype(pd.Int64Dtype())
        return self.df

    def _remove_zero_invoiced_customers(self):
        self.df = self.df[~self.df.groupby('customer_id')['invoiced_amount'].transform('sum').eq(0)]
        return self.df

    def preprocess(self):
        self.df = self._fill_na_values()
        self.df = self._aggregate_data()
        self.df = self._calculate_duration()
        self.df = self._random_generate_credit_score()
        return self.df

    def _fill_na_values(self):
        self.df['invoice_reminder_degree'].fillna(0, inplace=True)
        self.df['invoice_reminders'].fillna(0, inplace=True)
        return self.df

    def _aggregate_data(self):
        end_date_max = pd.to_datetime('2023-05-11')
        # Continue with your code to aggregate data...
        return self.df

    def _calculate_duration(self):
        # Continue with your code to calculate duration...
        return self.df

    def _random_generate_credit_score(self):
        # Continue with your code to generate random credit scores...
        return self.df


# And for splitting your data, you can have a standalone function:
def split_data(data, train_percent=.7, validate_percent=.15):
    m = len(data)
    train_end = int(train_percent * m)
    validate_end = int(validate_percent * m) + train_end
    train = data[:train_end]
    validate = data[train_end:validate_end]
    test = data[validate_end:]
    return train, validate, test
