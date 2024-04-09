import yfinance as yf
import pandas as pd
from datetime import datetime
import requests
import warnings
import logging

logging.getLogger('yfinance').setLevel(logging.CRITICAL)

class LoadingData:
    def __init__(self, tickers=None, exchanges=None, regions=None, cryptos=None, sectors=None,\
                 byexchange=False, byregion=False, bycrypto=False, bysector=False):
        """
        The function initializes a list of stock tickers, defaulting to the S&P 500 if no tickers are
        provided.
        
        :param tickers: The `tickers` parameter in the `__init__` method is used to initialize the object
        with a list of stock tickers. If no tickers are provided, it defaults to the list of S&P 500 company
        tickers obtained from a Wikipedia page.
        """

        if (byexchange + byregion + bycrypto + bysector) > 1:
            raise ValueError("Only one of 'byexchange', 'byregion', 'bycrypto', or 'bysector' can be True at a time.")
        elif not all(isinstance(param, bool) for param in [byexchange, byregion, bycrypto, bysector]):
            raise ValueError("'byexchange', 'byregion', 'bycrypto', or 'bysector' parameters must be boolean values.")
        elif tickers is not None and (byexchange or byregion or bycrypto or bysector):
            raise ValueError("If 'tickers' are provided, 'byexchange', 'byregion', 'bycrypto', or 'bysector' must all be False.")
        elif tickers is None and exchanges is None and regions is None and cryptos is None and sectors is None \
            and not byexchange and not byregion and not bycrypto and not bysector:
            raise ValueError("No data source selected. Please provide tickers, exchanges, regions, cryptos, or sectors.")
        elif exchanges is not None and (regions is not None or cryptos is not None or sectors is not None):
            raise ValueError("Only one of 'exchanges' or 'regions' or 'cryptos' or 'sectors' can be provided.")
        elif regions is not None and (exchanges is not None or cryptos is not None or sectors is not None):
            raise ValueError("Only one of 'exchanges' or 'regions' or 'cryptos' or 'sectors' can be provided.")
        elif cryptos is not None and (exchanges is not None or regions is not None or sectors is not None):
            raise ValueError("Only one of 'exchanges' or 'regions' or 'cryptos' or 'sectors' can be provided.")
        elif sectors is not None and (exchanges is not None or regions is not None or cryptos is not None):
            raise ValueError("Only one of 'exchanges' or 'regions' or 'cryptos' or 'sectors' can be provided.")
        elif exchanges is not None and not byexchange:
            raise ValueError("If 'exchanges' is provided, 'byexchange' must be True.")
        elif regions is not None and not byregion:
            raise ValueError("If 'regions' is provided, 'byregion' must be True.")
        elif cryptos is not None and not bycrypto:
            raise ValueError("If 'cryptos' is provided, 'bycrypto' must be True.")
        elif sectors is not None and not bysector:
            raise ValueError("If 'sectors' is provided, 'bysector' must be True.")

        if tickers is None:
            tickers_list = []
            if byexchange is True:
                if exchanges is not None:
                    if isinstance(exchanges, list):
                        exchange_list = exchanges
                    elif isinstance(exchanges, str):
                        exchange_list = [exchanges]
                    else:
                        raise ValueError("The 'exchanges' parameter must be a list or a string.")
                else:
                    exchange_list = ['nyse', 'nasdaq', 'amex']
                for exchange in exchange_list:
                    df = self.__request_nasdaq__('byexchange',exchange)
                    if df.empty:
                        print(f'No data found for the byexchange {byexchange}.')
                    else:
                        df_filtered = df[~df['symbol'].str.contains("\.|\^")]
                        tickers_list.extend(df_filtered['symbol'].tolist())
                self.tickers = tickers_list
            elif byregion is True:
                if regions is not None:
                    if isinstance(regions, list):
                        region_list  = regions
                    elif isinstance(regions, str):
                        region_list = [regions]
                    else:
                        raise ValueError("The 'exchanges' parameter must be a list or a string.")
                else:
                    region_list = ['AFRICA','EUROPE','ASIA','SOUTH AMERICA','NORTH AMERICA']
                for region in region_list:
                    df = self.__request_nasdaq__('region',region)
                    if df.empty:
                        print(f'No data found for the region {region}.')
                    else:
                        df_filtered = df[~df['symbol'].str.contains("\.|\^")]
                        tickers_list.extend(df_filtered['symbol'].tolist())
                self.tickers = tickers_list
            elif bycrypto is True:
                if cryptos is not None:
                    if isinstance(cryptos, list):
                        crypto_list = cryptos
                    elif isinstance(cryptos, str):
                        crypto_list = [cryptos]
                    else:
                        raise ValueError("The 'cryptos' parameter must be a list or a string.")
                else:
                    crypto_list = self.__get_crypto_tickers__()
                self.tickers = crypto_list
            elif bysector is True:
                if sectors is not None:
                    if isinstance(sectors, list):
                        sector_list = sectors
                    elif isinstance(sectors, str):
                        sector_list = [sectors]
                    else:
                        raise ValueError("The 'sectors' parameter must be a list or a string.")
                else:
                    sector_list = ['Basic Materials','Consumer Discretionary', 'Energy','Industrials',
                                   'Real Estate','Technology','Utilities', 'Consumer Staples', 'Health Care', 'Telecommunication']
                for sector in sector_list:
                    df = self.__request_nasdaq__('sector',sector)
                    if df.empty:
                        print(f'No data found for the sector {sector}.')
                    else:
                        df_filtered = df[~df['symbol'].str.contains("\.|\^")]
                        tickers_list.extend(df_filtered['symbol'].tolist())
                self.tickers = tickers_list
        elif isinstance(tickers, list):
            self.tickers = tickers
        else:
            self.tickers = [tickers]
    
    def __request_nasdaq__(self, param, id_list)->pd.DataFrame:
            
        headers = {
                'authority': 'api.nasdaq.com',
                'accept': 'application/json, text/plain, */*',
                'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.116 Safari/537.36',
                'origin': 'https://www.nasdaq.com',
                'sec-fetch-site': 'same-site',
                'sec-fetch-mode': 'cors',
                'sec-fetch-dest': 'empty',
                'referer': 'https://www.nasdaq.com/',
                'accept-language': 'en-US,en;q=0.9',
        }

        r = requests.get('https://api.nasdaq.com/api/screener/stocks', headers=headers, params=self.__params__(f'{param}',id_list))
        data = r.json()['data']
        df = pd.DataFrame(data['rows'], columns=data['headers'])
        return df

    def __params__(self, param_type, value):
        """
        Returns parameters for API request based on the type of parameter needed.

        :param param_type: Type of parameter (ex: 'byexchange' or 'region').
        :param value: Value corresponding to the parameter type.
        :return: Parameters for the API request.
        """
        common_params = (
            ('letter', '0'),
            ('download', 'true')
        )
        specific_param = (param_type, value)
        params = common_params + (specific_param,)
        return params
    
    def __get_crypto_tickers__(self):
        url = 'https://api.coingecko.com/api/v3/coins/list'
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            ticker_list_with_usd = [crypto['symbol'].upper() for crypto in data]
            filtered_tickers = map(self.__filter_tickers__, ticker_list_with_usd)
            ticker_list_sorted = list(filter(None,sorted(set(filtered_tickers))))
            ticker_list = [ticker + '-USD' for ticker in ticker_list_sorted]
            return ticker_list
        else:
            print(f"Error: {response.status_code}")
            return []
        
    def __filter_tickers__(self,ticker):
        return ''.join(char if char not in '$.!\|^()@:° ͜ʖ ͡°' else '' for char in ticker)

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
                if data.empty:
                    print(f"No data found for {ticker} in YahooFinance.")
                    continue
                data['ticker'] = ticker
                all_data.append(data)
            except Exception as e:
                print(f"Could not download data for {ticker}. Error: {e}")
        if all_data:
            combined_data = pd.concat(all_data)
            combined_data.reset_index(inplace=True)
            combined_data = combined_data.sort_values(['ticker','Date'], ascending=False).reset_index(drop=True)
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