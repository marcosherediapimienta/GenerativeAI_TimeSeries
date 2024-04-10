import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from chronos import ChronosPipeline
import warnings

warnings.filterwarnings("ignore")

class Chronos:
    def __init__(self, ts_data: pd.DataFrame, train_ratio: float = 0.7, validation_ratio: float = 0.2, column_num: str='y',
                  pretrained_model: str = 'small', freq: str = 'D', num_samples: int = 20, temperature: float = 1.0, 
                  top_k: int = 50, top_p: float = 1.0):
        """
        Initializes the Chronos class with the time series data.
        :param ts_data: A Pandas DataFrame containing the time series data with a 'unique_id' column.
        :param train_ratio: The proportion of data to be used for training.
        :param validation_ratio: The proportion of data to be used for validation.
        """
        self.ts_data = ts_data
        self.train_ratio = train_ratio
        self.validation_ratio = validation_ratio
        self.num_column = column_num
        self.freq = freq
        self.num_samples = num_samples
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p

        self.models = {}  
        self.best_params = {} 
        self.results = {}

        if pretrained_model == 'small':
            self.pretrained_model = 'amazon/chronos-t5-small'
        elif pretrained_model == 'tiny':
            self.pretrained_model = 'amazon/chronos-t5-tiny'
        elif pretrained_model == 'mini':
            self.pretrained_model = 'amazon/chronos-t5-mini'
        elif pretrained_model == 'base':
            self.pretrained_model = 'amazon/chronos-t5-base'
        elif pretrained_model == 'large':
            self.pretrained_model = 'amazon/chronos-t5-large'

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def __split_data__(self, data: pd.DataFrame):
        """
        Splits the data into train, validation, and test sets.
        """
        max_index = len(data)
        train_index = int(max_index * self.train_ratio)
        validation_index = train_index + int(max_index * self.validation_ratio)
        return {
            'train_data': data[:train_index],
            'validation_data': data[train_index:validation_index],
            'test_data': data[validation_index:]
        }
    
    def predict(self, evaluation: bool = False, horizon: int = 12)->pd.DataFrame:

        self.evaluation = evaluation

        pipeline = ChronosPipeline.from_pretrained(self.pretrained_model, device_map = self.device, 
                                                   torch_dtype = torch.float16)

        unique_ids = self.ts_data['unique_id'].unique()

        ts_forecast = []

        for unique_id in unique_ids:

            print(f"Processing {unique_id}...")

            ts_data_individual = self.ts_data[self.ts_data['unique_id'] == unique_id]
            ts_data_individual = ts_data_individual.sort_values(by='ds', ascending=True)
            split_data = self.__split_data__(ts_data_individual)
            
            if not evaluation:
                ts_chronos = pd.concat([split_data['train_data'], split_data['validation_data'], split_data['test_data']])
                self.prediction_length = horizon

                last_date = split_data['test_data']['ds'].max()
                start_date = last_date + pd.Timedelta(days=1)
                self.forecast_dates = pd.date_range(start=start_date, periods=self.prediction_length, freq=self.freq)

            else:
                ts_chronos = pd.concat([split_data['train_data'], split_data['validation_data']])
                self.prediction_length = len(split_data['test_data'])

                last_date = split_data['validation_data']['ds'].max()
                start_date = last_date + pd.Timedelta(days=1)
                self.forecast_dates = pd.date_range(start=start_date, periods=self.prediction_length, freq=self.freq)

                self.mape_scores = {}
                self.smape_scores = {}

            context = torch.tensor(ts_chronos[self.num_column].values)

            self.forecast = pipeline.predict(
                context,
                self.prediction_length,
                num_samples=self.num_samples,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                limit_prediction_length=False
            )

            forecast_np = self.forecast.numpy().mean(axis=1).flatten()
            
            if self.evaluation:
                real_series = split_data['test_data'][self.num_column]
                mape_score = self.__mape__(real_series, forecast_np)
                smape_score = self.__smape__(real_series, forecast_np)

                self.mape_scores[unique_id] = mape_score
                self.smape_scores[unique_id] = smape_score

                print(f'MAPE: {mape_score:.2f}%, SMAPE: {smape_score:.2f}% for {unique_id}.')

                ts_results = pd.DataFrame({'unique_id':unique_id,'ds': self.forecast_dates, 'y': real_series, 'yhat':forecast_np})
            else:
                ts_results = pd.DataFrame({'unique_id':unique_id,'ds': self.forecast_dates, 'yhat':forecast_np})
            
            ts_forecast.append(ts_results)
        
        ts_foreast_client = pd.concat(ts_forecast, ignore_index=True)

        return ts_foreast_client
    
    def __mape__(self, real:pd.Series, forecast:pd.Series) -> float:

        real, forecast = np.array(real), np.array(forecast)
        mape = np.mean(np.abs((real - forecast) / real)) * 100

        return mape
    
    def __smape__(self, real:pd.Series, forecast:pd.Series) -> float:

        real, forecast = np.array(real), np.array(forecast)
        smape = 100/len(real) * np.sum(2 * np.abs(forecast - real) / (np.abs(real) + np.abs(forecast)))

        return smape

    def get_scores(self)->pd.DataFrame:

        mape_scores = pd.DataFrame(self.mape_scores.items(), columns=['unique_id', 'mape'])
        smape_scores = pd.DataFrame(self.smape_scores.items(), columns=['unique_id', 'smape'])

        df_scores = pd.merge(mape_scores, smape_scores, on='unique_id')  
    
        return df_scores   