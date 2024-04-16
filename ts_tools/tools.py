import pandas as pd
import matplotlib.pyplot as plt

class tools:
    def __init__(self):
        pass
    
    def ts_prepartion(self, ts:pd.DataFrame, date_column:str, target_column:str)->pd.DataFrame:
        """
        The function `ts_preparation` renames columns in a DataFrame and returns a subset of columns.
        
        :param ts: The `ts_preparation` function takes three parameters:
        :param date_column: The `date_column` parameter in the `ts_preparation` function refers to the
        column in the input time series data (`ts`) that contains the dates or timestamps associated with
        each data point
        :param target_column: The `target_column` parameter in the `ts_preparation` function refers to
        the column in the input time series data that contains the target variable or the variable you
        are trying to predict or analyze. This column is typically the one that you want to forecast or
        model based on historical data
        :return: a pandas DataFrame with columns 'unique_id', 'ds', and 'y' after renaming the columns
        'ticker' to 'unique_id', 'date_column' to 'ds', and 'target_column' to 'y'.
        """
        ts = ts.rename(columns={'ticker':'unique_id', date_column: 'ds', target_column: 'y'})
        ts = ts[['unique_id','ds', 'y']]
        return ts
    
    def plot_ts(self, ts:pd.DataFrame):
        """
        The function plots time series data for up to 5 unique IDs, handling cases where there are more than
        5 unique IDs.
        
        :param ts: It looks like the code you provided is a method for plotting time series data for
        different unique IDs. The `ts` parameter seems to be a DataFrame containing time series data with
        columns 'unique_id', 'ds', and 'y'
        """

        unique_ids = ts['unique_id'].unique()

        if len(unique_ids) > 5:
            print(f'Too many unique ids to plot: {len(unique_ids)}. We will plot the first 5.') 
            unique_ids = unique_ids[:5]

        fig, axs = plt.subplots(len(unique_ids), 1, figsize=(10, 6), sharex=True)
    
        if len(unique_ids) == 1:
            axs = [axs]
        
        for i, unique_id in enumerate(unique_ids):
            subset = ts[ts['unique_id'] == unique_id]  
            axs[i].plot(subset['ds'], subset['y'], label=f"ID: {unique_id}")  
            axs[i].legend(loc="best")  
            axs[i].set_ylabel('y')  
        
        plt.xlabel('ds')  
        plt.tight_layout()  
        plt.show() 

    def plot_forecast(self, ts: pd.DataFrame):

        unique_ids = ts['unique_id'].unique()

        fig, axs = plt.subplots(len(unique_ids), 1, figsize=(10, 6), sharex=True)
    
        if len(unique_ids) == 1:
            axs = [axs]
        
        for i, unique_id in enumerate(unique_ids):
            subset = ts[ts['unique_id'] == unique_id]  
            axs[i].plot(subset['ds'], subset['yhat'], label=f"ID: {unique_id}")
            if 'y' in subset.columns:
                axs[i].plot(subset['ds'], subset['y'], label=f"Actual ID: {unique_id}")
            if 'yhat_lower' in subset.columns and 'yhat_upper' in subset.columns:
                axs[i].fill_between(subset['ds'], subset['yhat_lower'], subset['yhat_upper'], alpha=0.2)
            axs[i].legend(loc="best")
            axs[i].set_xticklabels(subset['ds'], rotation=45)

        plt.xlabel('ds') 
        plt.ylabel('Forecasted y') 
        plt.tight_layout()  
        plt.show() 
