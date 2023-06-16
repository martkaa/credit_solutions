import pandas as pd
import numpy as np

class DataPreprocessor:
    def __init__(self, filename):
        self.df = self._read_data(filename)
        self.df = self._process_dates()
        self.df = self._format_columns()
        
        self.df = self._process_numeric_cols()
        self.df = self._process_categorical_cols()
        self.df = self._remove_zero_invoiced_customers()

    def _read_data(self, filename):
        data = pd.read_csv(filename, encoding='ISO-8859-1', sep=';', low_memory=False)
        return data
    
    

    def _format_columns(self):
        self.df.columns = self.df.columns.str.replace('-', '_')
        for col in ['calculated_credit_limit']:
            self.df.loc[:, col] = self.df[col].str.replace(',', '.').astype(float)
            self.df.loc[:, col].fillna(100000, inplace=True)
        
        return self.df
    
    def _process_dates(self):
        for col in ['start-date', 'end-date']:
            self.df.loc[:, col] = pd.to_datetime(self.df[col], format='%d.%m.%Y', errors='coerce')
        self.df.loc[:, 'period'] = pd.to_datetime(self.df['period'], format='%Y-%m', errors='coerce')
        # set the index to the period
        return self.df

    

    def _process_numeric_cols(self):
        # remove all rows where cutsomer id is not a number and convert to int
        print(self.df.head())
        self.df = pd.to_numeric(self.df['customer_id'], errors='coerce')
        self.df = self.df.dropna()
        
        self.df['customer_id'] = self.df['customer_id'].astype(int)


        for col in ['invoiced_amount', 'cost', 'other_costs']:
            self.df.loc[:, col] = self.df[col].str.replace(',', '.').astype(float)
            self.df.loc[:, col] = self.df[col].abs()
            self.df.loc[:, col].fillna(0, inplace=True)
        
        #self.df['customer_id'] = self.df['customer_id'].astype(int)
        return self.df

    def _process_categorical_cols(self):
        self.df.replace('-', np.nan, inplace=True)
        self.df['risk_segment'] = self.df['risk_segment'].astype(pd.Int64Dtype())

        # set the values in ´country´ to ´other´ if there is no country
        self.df.loc[self.df['country'].isnull(), 'country'] = 'unknown_country'
        # set industry to other if there is no industry
        self.df.loc[self.df['industry'].isnull(), 'industry'] = 'Unknown'
        return self.df
    
    # to make sure the predictions are based on a sufficient amount of data
    def _remove_customers_with_less_than_37_periods(self, df):
        # Filter out customers with fewer than 37 periods
        filtered_df = df[df.groupby('customer_id')['period'].transform('count').ge(37)]
        return filtered_df


    def _remove_zero_invoiced_customers(self):
        self.df = self.df[~self.df.groupby('customer_id')['invoiced_amount'].transform('sum').eq(0)]
        # remove cuatomers with 
        return self.df

    def preprocess(self):
        self.df = self._remove_customers_with_less_than_37_periods(self.df)
        self.df = self._fill_na_values()
        self.df = self._random_generate_credit_score()
        self.df = self._fill_missing_credit_scores()

        aggregated_df = self._aggregate_data()
        aggregated_df = self._calculate_duration_and_event(aggregated_df)
        return self.df, aggregated_df

    def _fill_na_values(self):
        self.df['invoice_reminder_degree'].fillna(0, inplace=True)
        self.df['invoice_reminders'].fillna(0, inplace=True)
        # drop rows with null in customer name
        self.df = self.df.dropna(subset=['customer_name'])
        # drop rows with null in postcode
        self.df = self.df.dropna(subset=['postcode'])
        return self.df
    
    def _calculate_duration_and_event(self, df):
        end_date_max = pd.to_datetime('2023-05-11')
        df['event'] = df['end_date'].notnull().astype(int)
        df.loc[df['end_date'].isnull(), 'end_date'] = end_date_max
        df['duration'] = (df['end_date'] - df['start_date']).dt.days
        #self.df = self.df.drop(columns=['start_date', 'end_date'])
        return df
    
    def _random_generate_credit_score(self):
        self.df['contribution_margin'] = (self.df['invoiced_amount'] - self.df['cost']) / self.df['invoiced_amount']
        # set the values in contribution_margin to 0 if the value is NaN
        

        score_ranges = {1: (0, 0.25), 2: (0.25, 0.5), 3: (0.5, 0.75), 4: (0.75, 1)}
        score_ranges_estimated = {1: (0, 0.2), 2: (0.2, 0.4), 3: (0.4, 0.6), 4: (0.6, 0.8), 5: (0.8, 1)}

        self.df['credit_score'] = self.df.apply(lambda row: np.random.uniform(*score_ranges[row['risk_segment']]) if pd.notnull(row['risk_segment']) else np.nan, axis=1)

        mask = self.df['risk_segment'].isnull()

        self.df.loc[mask, 'credit_score'] = self.df[mask]['contribution_margin'].apply(lambda x: np.random.uniform(*score_ranges_estimated[1]) if 0 <= x < 0.2 else
                                                                                           np.random.uniform(*score_ranges_estimated[2]) if 0.2 <= x < 0.4 else
                                                                                           np.random.uniform(*score_ranges_estimated[3]) if 0.4 <= x < 0.6 else
                                                                                           np.random.uniform(*score_ranges_estimated[4]) if 0.6 <= x < 0.8 else
                                                                                           np.random.uniform(*score_ranges_estimated[5]) if 0.8 <= x <= 1 else np.nan)
        self.df['contribution_margin'] = self.df['contribution_margin'].fillna(0)
        # fill missing values in risk_segment with 0
        self.df['risk_segment'] = self.df['risk_segment'].fillna(0)
        return self.df
    
    def _fill_missing_credit_scores(self):
        self.df['credit_score'] = self.df.groupby('customer_id')['credit_score'].transform(lambda x: x.fillna(x.mean()))
        # remove customers with no credit score
        self.df = self.df[~self.df['credit_score'].isnull()]
        return self.df

    

    def _aggregate_data(self):
        aggregated_df = self.df.groupby(['customer_id']).agg({
            'start_date': 'min',
            'end_date': 'max',
            'postcode': 'first', 
            'customer_group': 'first', 
            'end_date': 'max', 
            'start_date': 'first', 
            'invoice_frequency': 'sum',
            'invoiced_amount': 'mean', 
            'cost': 'mean', 
            'other_costs': 'mean', 
            'invoice_reminders': 'sum',
            'invoice_reminder_degree': 'max', 
            'credit_score': 'mean', 
            'calculated_credit_limit': 'first',
            'contribution_margin': 'median'
        }).reset_index()
        return aggregated_df

