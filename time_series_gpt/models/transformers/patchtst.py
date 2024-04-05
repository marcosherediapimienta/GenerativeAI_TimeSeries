from neuralforecast import NeuralForecast
from neuralforecast.models import PatchTST
import matplotlib.pyplot as plt
import pandas as pd

class patchtst_model:
    def __init__(self, horizon, input_size, freq):
        self.horizon = horizon
        self.input_size = input_size
        self.freq = freq

    def fit(self, ts):
        self.ts = ts
        self.model = PatchTST(h=self.horizon, input_size=self.input_size, freq=self.freq)
        self.model.fit(self.ts)
        
    def predict(self):
        return self.model.predict(self.horizon)

    def plot(self):
        self.model.plot()

