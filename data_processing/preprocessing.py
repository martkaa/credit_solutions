import pandas as pd
import numpy as np

class DataPreprocessor:
    def __init__(self, filename):
        self.df = self._read_data(filename)
        self.df = self._format_columns()
        self.df = self._process_dates()
        self.df = self._process_numeric_cols()
        self.df = self._process_categorical_cols()
        self.df = self._remove_zero_invoiced_customers()

    def _read_data(self, filename):
        data = pd.read_csv(filename, encoding='ISO-8859-1', sep=';', low_memory=False)
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
            # set invoiced amount to its absolute value
            self.df.loc[:, col] = self.df[col].abs()
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
        self.df = self._calculate_duration()
        self.df = self._random_generate_credit_score()
        self.df = self._aggregate_data()
        return self.df

    def _fill_na_values(self):
        self.df['invoice_reminder_degree'].fillna(0, inplace=True)
        self.df['invoice_reminders'].fillna(0, inplace=True)
        return self.df
    
    def _calculate_duration(self):
        self.df['duration'] = (self.df['end_date'] - self.df['start_date']).dt.days
        self.df = self.df.drop(columns=['start_date', 'end_date'])
        return self.df

    def _aggregate_data(self):
        end_date_max = pd.to_datetime('2023-05-11')
        aggregated_df = self.df.groupby(['customer_id']).agg({
            'postcode': 'first', 
            'customer_group': 'first', 
            'end_date': 'max', 
            'start_date': 'first', 
            'duration': 'max',
            'event': 'first',
            'invoice_frequency': 'sum',
            'invoiced_amount': 'mean', 
            'cost': 'mean', 
            'other_costs': 'mean', 
            'invoice_reminders': 'sum',
            'invoice_reminder_degree': 'max', 
            'credit_score': 'mean', 
            'calculated_credit_limit': 'first',
            'contribution_margin': 'mean'
        }).reset_index()
        aggregated_df.loc[aggregated_df['end_date'].isnull(), 'end_date'] = end_date_max
        return aggregated_df

    def _calculate_duration(self):
        # Continue with your code to calculate duration...
        return self.df

    def _random_generate_credit_score(self):
        self.df['contribution_margin'] = (self.df['invoiced_amount'] - self.df['cost']) / self.df['invoiced_amount']

        score_ranges = {1: (0, 0.25), 2: (0.25, 0.5), 3: (0.5, 0.75), 4: (0.75, 1)}
        score_ranges_estimated = {1: (0, 0.2), 2: (0.2, 0.4), 3: (0.4, 0.6), 4: (0.6, 0.8), 5: (0.8, 1)}

        self.df['credit_score'] = self.df.apply(lambda row: np.random.uniform(*score_ranges[row['risk_segment']]) if pd.notnull(row['risk_segment']) else np.nan, axis=1)

        mask = self.df['risk_segment'].isnull()

        self.df.loc[mask, 'credit_score'] = self.df[mask]['contribution_margin'].apply(lambda x: np.random.uniform(*score_ranges_estimated[1]) if 0 <= x < 0.2 else
                                                                                           np.random.uniform(*score_ranges_estimated[2]) if 0.2 <= x < 0.4 else
                                                                                           np.random.uniform(*score_ranges_estimated[3]) if 0.4 <= x < 0.6 else
                                                                                           np.random.uniform(*score_ranges_estimated[4]) if 0.6 <= x < 0.8 else
                                                                                           np.random.uniform(*score_ranges_estimated[5]) if 0.8 <= x <= 1 else np.nan)
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
