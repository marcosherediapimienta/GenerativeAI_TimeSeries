import pandas as pd
import numpy as np
import itertools
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import logging

logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").disabled = True

class ProphetMeta:
    def __init__(self, ts_data: pd.DataFrame, train_ratio: float = 0.7, validation_ratio: float = 0.2):
        """
        Initializes the ProphetMeta class with the time series data.
        :param ts_data: A Pandas DataFrame containing the time series data with a 'unique_id' column.
        :param train_ratio: The proportion of data to be used for training.
        :param validation_ratio: The proportion of data to be used for validation.
        """
        self.ts_data = ts_data
        self.train_ratio = train_ratio
        self.validation_ratio = validation_ratio
        self.models = {}  
        self.best_params = {} 
        self.results = {}

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

    def train_and_evaluate(self, yearly_seasonality: bool = True, weekly_seasonality: bool = True, daily_seasonality: bool = False,
                           seasonality_mode: str = 'additive', fourier_order: int = 2, country_holidays: str = 'US', freq_cutoff: str = 'W-MON'):
        """
        Trains the Prophet self.model and evaluates its performance for each unique_id.
        """
        unique_ids = self.ts_data['unique_id'].unique()

        for unique_id in unique_ids:
            print(f"Processing {unique_id}...")
            ts_data_individual = self.ts_data[self.ts_data['unique_id'] == unique_id]
            split_data = self.__split_data__(ts_data_individual)

            ts_prophet = pd.concat([split_data['train_data'], split_data['validation_data'], split_data['test_data']])

            horizon = pd.Timedelta(days=len(split_data['test_data']))
            training = pd.Timedelta(days=len(split_data['train_data']))
            cutoffs = pd.date_range(start=split_data['validation_data']['ds'].min(),\
                                     end=split_data['validation_data']['ds'].max(),\
                                     freq=freq_cutoff)

            param_grid = {
                'changepoint_prior_scale': [0.01, 0.1, 1.0],
                'seasonality_prior_scale': [0.01, 0.1, 1.0],
                'holidays_prior_scale': [0.01, 0.1, 1.0]
            }

            all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]

            smape_scores = []
            mape_scores = []

            for params in all_params:
                model = Prophet(**params, yearly_seasonality=yearly_seasonality, weekly_seasonality=weekly_seasonality,
                                daily_seasonality=daily_seasonality, seasonality_mode=seasonality_mode)
                model.add_country_holidays(country_name=country_holidays)
                model.add_seasonality(name='monthly', period=30.5, fourier_order=fourier_order)
                model.fit(ts_prophet)
                df_cv_model = cross_validation(model, initial=str(training.days) + ' days', horizon=str(horizon.days) + ' days', 
                                               disable_tqdm=True, parallel='threads', cutoffs=cutoffs)
                df_metrics = performance_metrics(df_cv_model)
                smape_scores.append(df_metrics['smape'].values[0])

                if 'mape' in df_metrics.columns:
                    mape_scores.append(df_metrics['mape'].values[0])

            best_params_index = np.argmin(mape_scores) if mape_scores else np.argmin(smape_scores)
            best_params = all_params[best_params_index]

            best_model = Prophet(**best_params, yearly_seasonality=yearly_seasonality, weekly_seasonality=weekly_seasonality,
                                 daily_seasonality=daily_seasonality, seasonality_mode=seasonality_mode)
            best_model.add_country_holidays(country_name=country_holidays)
            best_model.add_seasonality(name='monthly', period=30.5, fourier_order=fourier_order)
            best_model.fit(ts_prophet)

            self.models[unique_id] = best_model
            self.best_params[unique_id] = best_params

            df_cv_best = cross_validation(best_model, initial=str(training.days) + ' days', horizon=str(horizon.days) + ' days', 
                                          disable_tqdm=True, parallel='threads', cutoffs=cutoffs)
            df_metrics_best = performance_metrics(df_cv_best)

            self.results[unique_id] = {
                'SMAPE Error': df_metrics_best['smape'].mean(),
                'MAPE Error': df_metrics_best['mape'].mean() if 'mape' in df_metrics_best.columns else None,
                'Accuracy': (1 - df_metrics_best['smape'].mean()) * 100}
            
            print(f"Finished processing {unique_id}.")

    def predict(self, unique_id: str, horizon: int, freq: str = 'D')-> pd.DataFrame:
        """
        Generates future dates and predicts values for them using the trained Prophet self.model for a specific unique_id.
        """
        model = self.models.get(unique_id)
        if model:
            future = model.make_future_dataframe(periods=horizon, freq=freq)
            forecast = model.predict(future)
            return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        else:
            print(f"No model found for unique_id: {unique_id}")
            return None

    def get_results(self, unique_id: str):
        """
        Returns the evaluation results for a specific unique_id.
        """
        return self.results.get(unique_id, "No results found for this unique_id.")