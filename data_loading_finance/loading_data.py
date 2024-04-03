import yfinance as yf
import pandas as pd
from datetime import datetime

class LoadingData:
    def __init__(self, tickers=None):
        """
        The function initializes a list of stock tickers, defaulting to the S&P 500 if no tickers are
        provided.
        
        :param tickers: The `tickers` parameter in the `__init__` method is used to initialize the object
        with a list of stock tickers. If no tickers are provided, it defaults to the list of S&P 500 company
        tickers obtained from a Wikipedia page.
        """
        if tickers is None:
            ## Default to S&P 500
            tickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol'].tolist()
        elif isinstance(tickers, str):
            tickers = [tickers]
        self.tickers = tickers

    def get_data(self, start_date=None, end_date=None)->pd.DataFrame:
        """
        The `get_data` function downloads stock data for a list of tickers within a specified date range or
        defaults to the last 10 years if no dates are provided.
        
        :param start_date: The `start_date` parameter in the `get_data` method is used to specify the
        starting date for downloading data. If `start_date` is not provided when calling the method, it
        defaults to a date calculated based on the current date and a specified number of years (default is
        10 years
        :param end_date: The `end_date` parameter in the `get_data` function is used to specify the end date
        for downloading data. If `end_date` is not provided when calling the function, it defaults to the
        current date (today's date) in the format '%Y-%m-%d'
        :return: The `get_data` method returns a pandas DataFrame containing historical stock price data for
        the specified tickers within the specified date range. If data is successfully downloaded for the
        tickers, the method concatenates the data into a single DataFrame, sorts it by date and ticker, and
        returns the combined DataFrame. If no data is downloaded for any ticker, an empty DataFrame is
        returned.
        """
        all_data = []

        if end_date is None:
            end_date = datetime.today().strftime('%Y-%m-%d')
        if start_date is None:
            years= 10
            start_date = (datetime.today() - pd.Timedelta(days=365*years)).strftime('%Y-%m-%d')

        for ticker in self.tickers:
            print(f"Downloading data for {ticker}...")
            try:
                data = yf.download(ticker, start=start_date, end=end_date)
                data['ticker'] = ticker
                all_data.append(data)
            except Exception as e:
                print(f"Could not download data for {ticker}. Error: {e}")
        if all_data:
            combined_data = pd.concat(all_data)
            combined_data.reset_index(inplace=True)
            combined_data = combined_data.sort_values(['Date', 'ticker'], ascending=False).reset_index(drop=True)
            return combined_data
        else:
            return pd.DataFrame()
    
    def get_info_ticker(self, keys_of_interest=None)->pd.DataFrame:
        """
        This function retrieves information for one or multiple stock tickers based on specified keys of
        interest and returns a pandas DataFrame with the information.
        
        :param keys_of_interest: The `keys_of_interest` parameter in the `get_info_ticker` method is a list
        that contains the keys of interest for retrieving information about a stock ticker. By default, if
        no specific keys are provided, the method will use a predefined list of keys including 'shortName',
        'sector', 'inmdustry', and 'country'.
        :return: The function `get_info_ticker` returns a pandas DataFrame containing information for the
        specified tickers. If `keys_of_interest` is not provided, it defaults to a set of keys including
        'shortName', 'sector', 'industry', and 'country'. The function retrieves information for each ticker
        in `self.tickers`, filters the information based on the keys of interest, adding the ticker symbol. 
        """
        if keys_of_interest is None:
            keys_of_interest = ['shortName','sector','industry','country']
        try:
            if len(self.tickers) > 1:
                print(f"Getting info for {len(self.tickers)} tickers...")
                all_info = []
                for ticker in self.tickers:
                    info = yf.Ticker(ticker).info
                    filtered_info = {key: info.get(key, None) for key in keys_of_interest}
                    filtered_info['ticker'] = ticker
                    all_info.append(filtered_info)
                df_info = pd.DataFrame(all_info)
                return df_info
            print(f'Getting info for ticker {self.tickers[0]}...')
            info = yf.Ticker(self.tickers[0]).info
            filtered_info = {key: info.get(key, None) for key in keys_of_interest}
            filtered_info['ticker'] = self.tickers[0]
            df_info = pd.DataFrame([filtered_info])
            return df_info
        except Exception as e:
            print(f"Could not get info for {self.tickers}. Error: {e}")
            return pd.DataFrame(columns=keys_of_interest)