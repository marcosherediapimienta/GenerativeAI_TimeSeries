import pandas as pd
import matplotlib.pyplot as plt

class tools:
    def __init__(self):
        pass
    def ts_prepartion(self, ts, date_column, target_column)->pd.DataFrame:
        ts = ts.rename(columns={'ticker':'unique_id', date_column: 'ds', target_column: 'y'})
        ts = ts[['unique_id','ds', 'y']]
        return ts
    def plot(self, ts):
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
            axs[i].set_ylabel('Valor')  
        
        plt.xlabel('Fecha')  
        plt.tight_layout()  
        plt.show() 