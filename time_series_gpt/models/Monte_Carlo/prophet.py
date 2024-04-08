import pandas as pd
import numpy as np
import itertools
from prophet import Prophet, cross_validation, performance_metrics
import logging

logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").disabled = True

class ProphetMeta:
    def __init__(self, ts_data, train_ratio=0.7, validation_ratio=0.2):
        """
        Initializes the ProphetMeta class with the time series data.

        :param ts_data: A Pandas DataFrame containing the time series data.
        :param train_ratio: The proportion of data to be used for training.
        :param validation_ratio: The proportion of data to be used for validation.
        """
        self.ts_data = ts_data
        self.train_ratio = train_ratio
        self.validation_ratio = validation_ratio

        # Compute indices for splitting data
        max_index = len(ts_data)
        self.train_index = int(max_index * train_ratio)
        self.validation_index = self.train_index + int(max_index * validation_ratio)

        # Split the DataFrame
        self.train_data = pd.DataFrame(ts_data[:self.train_index])
        self.validation_data = pd.DataFrame(ts_data[self.train_index:self.validation_index])
        self.test_data = pd.DataFrame(ts_data[self.validation_index:])

        self.model = None
        self.best_params = None
        self.smape_error = None
        self.mape_error = None
        self.accuracy = None

    def train_and_evaluate(self):
        """
        Trains the Prophet model and evaluates its performance.
        """
        ts_prophet = pd.concat([self.train_data, self.validation_data, self.test_data]).reset_index()
        horizon = pd.Timedelta(days=len(self.test_data)*7)
        training = pd.Timedelta(days=len(self.train_data)*7)
        cutoffs = pd.date_range(start=self.validation_data['ds'].min(), end=self.validation_data['ds'].max(), freq='4W-MON')

        param_grid = {
            'changepoint_prior_scale': [0.01, 0.1, 1.0],
            'seasonality_prior_scale': [0.01, 0.1, 1.0],
            'holidays_prior_scale': [0.01, 0.1, 1.0]
        }

        all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]

        smape_scores = []
        mape_scores = []

        for params in all_params:
            model = Prophet(**params, yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False, seasonality_mode='multiplicative')
            model.add_country_holidays(country_name='ES')
            model.add_seasonality(name='monthly', period=30.5, fourier_order=2)
            model.fit(ts_prophet)
            df_cv_model = cross_validation(model, initial=training, horizon=horizon, disable_tqdm=True, parallel='processes', cutoffs=cutoffs)
            df_metrics = performance_metrics(df_cv_model)
            smape_scores.append(df_metrics['smape'].values[0])

            if 'mape' in df_metrics.columns:
                mape_scores.append(df_metrics['mape'].values[0])

        best_params_index = np.argmin(mape_scores) if mape_scores else np.argmin(smape_scores)
        self.best_params = all_params[best_params_index]
        self.model = Prophet(**self.best_params, yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False, seasonality_mode='multiplicative')
        self.model.add_country_holidays(country_name='ES')
        self.model.add_seasonality(name='monthly', period=30.5, fourier_order=2)
        self.model.fit(ts_prophet)

        df_cv_model = cross_validation(self.model, initial=training, horizon=horizon, disable_tqdm=True, parallel='processes', cutoffs=cutoffs)
        df_metrics = performance_metrics(df_cv_model)
        self.smape_error = df_metrics['smape'].mean()

        if 'mape' in df_metrics.columns:
            self.mape_error = df_metrics['mape'].mean()

        self.accuracy = (1 - self.smape_error) * 100
        print(f'The forecasting model accuracy is: {self.accuracy:.2f}%, with an SMAPE of {self.smape_error*100:.2f}%.')

    def predict(self, periods, freq='D'):
        """
        Generates future dates and predicts values for them using the trained Prophet model.
        
        :param periods: Number of periods to predict into the future.
        :param freq: Frequency of the prediction periods (default 'D' for days).
        :return: A Pandas DataFrame with the forecast.
        """
        future = self.model.make_future_dataframe(periods=periods, freq=freq)
        forecast = self.model.predict(future)
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]    

    def get_results(self):
        """
        Returns the evaluation results.

        :return: A dictionary containing the SMAPE error, MAPE error, and model accuracy.
        """
        return {
            'SMAPE Error': self.smape_error,
            'MAPE Error': self.mape_error,
            'Accuracy': self.accuracy
        }