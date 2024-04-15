from neuralforecast import NeuralForecast
from neuralforecast.models import PatchTST
from neuralforecast.auto import AutoPatchTST
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import smape
from ray import tune
from ray.tune.search.hyperopt import HyperOptSearch
from neuralforecast.losses.pytorch import MAE
import matplotlib.pyplot as plt
import pandas as pd

class patchtst_forecaster:
    def __init__(self, horizon:int=30, input_size:int=30, freq:str='D', auto:bool=True):
        self.horizon = horizon
        self.input_size = input_size
        self.freq = freq
        if auto:
            npatch_config = AutoPatchTST.get_default_config(h=horizon, backend='ray')
            model = AutoPatchTST(h=horizon, config= npatch_config, search_alg = HyperOptSearch(), backend ='ray', num_samples=3)
            self.nf = NeuralForecast(
                models=[model],
                freq= freq,
                local_scaler_type='robust-iqr'
            )
        else:  
            self.nf = NeuralForecast(
                models=[PatchTST(h=horizon, input_size=input_size)],
                freq= freq,
                local_scaler_type='robust-iqr'
            )

    def fit(self, ts:pd.DataFrame, evaluation:bool=False):
        self.ts = ts
        self.evaluation = evaluation
        if self.evaluation:
            ts_cv = self.nf.cross_validation(ts,n_windows=3, step_size=self.horizon, refit=False)
            ts_evaluation = evaluate(ts_cv.loc[:,ts_cv.columns !='cutoff'], metrics=[smape])
            ts_evaluation['best_model'] = ts_evaluation.drop(columns=['metric','unique_id']).idxmin(axis=1)
            return ts_evaluation
        else:
            self.nf.fit(ts)
        
    def predict(self) -> pd.DataFrame:
        if self.evaluation:
            print('We cannot predict with evaluation=True, please set evaluation=False in the fit method.')
            return None
        else:   
            result = self.nf.predict(self.horizon)
            result = result.reset_index()
            return result 


