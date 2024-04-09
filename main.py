from data_loading_finance.loading_data import LoadingData
import matplotlib.pyplot as plt

# Load the data
loader = LoadingData(bysector=True)
ts = loader.get_data()
info = loader.get_info_ticker()


ts['unique_id'] = 'BTC-USD'
ts = ts.rename(columns={'Date': 'ds', 'Adj Close': 'y'})
ts = ts[['unique_id','ds', 'y']]

print(info)
print(ts.head())

plt.plot(ts['ds'], ts['y'])
plt.show()







