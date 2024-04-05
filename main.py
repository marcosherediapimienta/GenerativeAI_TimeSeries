from time_series_gpt.models.patchtst import patchtst_model
from data_loading_finance.loading_data import LoadingData

# Load the data
loader = LoadingData(tickers=['AAPL'])
ts = loader.get_data()

# Create a PatchTST model
model = patchtst_model(horizon=7, input_size=1, freq='D')
model.fit(ts)
prediction = model.predict()
model.plot()





