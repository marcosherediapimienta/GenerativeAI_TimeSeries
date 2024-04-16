from neuralforecast import NeuralForecast
from neuralforecast.models import PatchTST
from neuralforecast.auto import AutoPatchTST
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import smape, mape
from ray.tune.search.hyperopt import HyperOptSearch
import pandas as pd

class patchtst_forecaster:
    def __init__(self, horizon:int=30, input_size:int=30, freq:str='D', auto:bool=True, evaluation:bool=False):
        self.horizon = horizon
        self.input_size = input_size
        self.freq = freq
        self.auto = auto
        self.evaluation = evaluation
        

    def fit(self, ts:pd.DataFrame):
        self.ts = ts

        if self.auto:
            npatch_config = AutoPatchTST.get_default_config(h=self.horizon, backend='ray')
            model = AutoPatchTST(h=self.horizon, config= npatch_config, search_alg = HyperOptSearch(),
                                  backend ='ray', num_samples=3)
            self.nf = NeuralForecast(
                models=[model],
                freq= self.freq,
                local_scaler_type='robust-iqr'
            )
        else:  
            self.nf = NeuralForecast(
                models=[PatchTST(h=self.horizon, input_size=self.input_size)],
                freq= self.freq,
                local_scaler_type='robust-iqr'
            )

        if self.evaluation:
            ts_cv = self.nf.cross_validation(ts, n_windows=3, step_size=self.horizon, refit=False)
            ts_cv = ts_cv.reset_index().rename(columns={'index':'unique_id'})
            self.ts_evaluation = evaluate(ts_cv.loc[:,ts_cv.columns !='cutoff'], metrics=[smape, mape])
            self.ts_evaluation['best_model'] = self.ts_evaluation.drop(columns=['metric','unique_id']).idxmin(axis=1)
            self.ts_evaluation['accuracy'] = ((1-self.ts_evaluation['sampe'])*100).round(2)
            print(self.ts_evaluation)
        
        self.nf.fit(ts)
        
    def predict(self) -> pd.DataFrame: 
            result = self.nf.predict(self.horizon)
            result = result.reset_index()
            return result 


