from python_bitvavo_api.bitvavo import Bitvavo
import json
import pandas as pd
import numpy as np
import time
import ta
from telegram import Bot
import asyncio
from datetime import datetime, timedelta
import os
import random

api_keys= json.loads(os.getenv('API_KEYS'))
api_key = api_keys['API_KEY']
api_secret = api_keys['API_SECRET']
token = Bot(token=api_keys['token'])
chat_id = api_keys["chat_id"]

bitvavo = Bitvavo({
    'APIKEY': api_keys['API_KEY'],
    'APISECRET': api_keys['API_SECRET'],
    'RESTURL': api_keys['RESTURL'],
    'WSURL': api_keys['WSURL'],

})


class apibot():
    def __init__(self, file_path, markets):
        self._markets = markets
        self._file_path = file_path

    def get_market_price(self, symbol):
        ticker = bitvavo.tickerPrice({'market': symbol})
        if 'error' in ticker:
            print(f"Fout bij ophalen marktprijs: {ticker['error']}")
        else:
            current_price = float(ticker['price'])
            
            return current_price

    
    async def place_market_order(self, symbol, amount, side):
        order = bitvavo.placeOrder(symbol, side, 'market', {'amount': amount})
        if 'error' in order:
            error_message = (f"Fout bij plaatsen kooporder: {order['error']}")
            print(error_message)
            await self.send_telegram_message(error_message)

        else:
            print(f"Kooporder succesvol geplaatst: {order}")
            buy_message = f"Gekocht:\nMarket: {symbol}"
            await self.send_telegram_message(buy_message)

            return order


    def place_stop_loss(self, symbol, amount, stop_loss_price):
        stop_loss_order = bitvavo.placeOrder(symbol, 'sell', 'stopLossLimit', {
            'amount': amount,
            'stopPrice': stop_loss_price,
            'price': stop_loss_price
        })
        if 'error' in stop_loss_order:
            print(f"Fout bij plaatsen stop-loss order: {stop_loss_order['error']}")
        else:
            print(f"Stop-loss order succesvol geplaatst: {stop_loss_order}")
            return stop_loss_order

    
    def place_take_profit(self, symbol, amount, take_profit_price):
        take_profit_order = bitvavo.placeOrder(symbol, 'sell', 'limit', {
            'amount': amount,
            'price': take_profit_price
        })
        if 'error' in take_profit_order:
            print(f"Fout bij plaatsen take-profit order: {take_profit_order['error']}")
        else:
            print(f"Take-profit order succesvol geplaatst: {take_profit_order}")
            return take_profit_order


    def place_long_position(self, symbol, amount, stop_loss_percentage, take_profit_percentage):

        order = self.place_market_order(self, symbol, amount, 'buy')
        if order:
            buy_price = float(order['fills'][0]['price'])
            stop_loss_price = round(buy_price * (1 - stop_loss_percentage / 100), 2)
            take_profit_price = round(buy_price * (1 + take_profit_percentage / 100), 2)

            self.place_stop_loss(symbol, amount, stop_loss_price)
            self.place_take_profit(symbol, amount, take_profit_price)


    def check_balance(self, asset):
        balance = bitvavo.balance({'symbol': asset})
        if 'error' in balance:
            print(f"Fout bij ophalen balans: {balance['error']}")
        else:
            for item in balance:
                if item['symbol'] == asset:
                    available_balance = float(item['available'])
                    print(f"Beschikbare {asset} balans: {available_balance}")
                    return available_balance
        return 0.0
        

    async def send_telegram_message(self, message):
        try:
            await token.send_message(chat_id=chat_id, text=message, read_timeout=20)
        except TimeoutError:
            print("Failed to send message due to timeout.")


    def get_bitvavo_data(self, market, interval, limit):

      response = bitvavo.candles(market, interval, {'limit': limit})
      if isinstance(response, dict):
          if response['errorCode'] == 205:
              print(f"Aandeel {market} niet gevonden")

              return None
      else:
          data = pd.DataFrame(response, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
          data = data.drop(data.index[500:])
          data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
          data[['open', 'high', 'low', 'close', 'volume']] = data[['open', 'high', 'low', 'close', 'volume']].apply(pd.to_numeric)
          data = data.set_index('timestamp')
          data = data.sort_index()
          data['market'] = market

          return data

    def generate_signals(self, df):

        if df is not None:
            for col in df.columns:
                if df[col].isnull().any():
                    print('Nog niet genoeg data')
                    break

                else:

                    # RSI Overbought / Oversold
                    df['RSI_Overbought'] = np.where(df['RSI'] >= 55, True, False)
                    df['RSI_Oversold'] = np.where(df['RSI'] <= 35, True, False)

                    # MACD Crossovers
                    df['MACD_Bullish'] = np.where(
                        (df['MACD'] > df['MACD_signal']) & (df['MACD'].shift(1) <= df['MACD_signal'].shift(1)), True, False)
                    df['MACD_Bearish'] = np.where(
                        (df['MACD'] < df['MACD_signal']) & (df['MACD'].shift(1) >= df['MACD_signal'].shift(1)), True, False)

                    # Bollinger Bands Cross
                    df['Bollinger_Breakout_High'] = np.where((df['close'] > df['Bollinger_High']), True, False)
                    df['Bollinger_Breakout_Low'] = np.where((df['close'] < df['Bollinger_Low']), True, False)

                    df['EMA_above'] = (df['EMA_8_above_EMA_13'] &
                                       df['EMA_13_above_EMA_21'] &
                                       df['EMA_21_above_EMA_55']).rolling(window=20).sum() == 20

                    df['EMA_below'] = (~df['EMA_8_above_EMA_13'] &
                                       ~df['EMA_13_above_EMA_21'] &
                                       ~df['EMA_21_above_EMA_55']).rolling(window=20).sum() == 20

                    return df

    def add_indicators(self, df):
        # Beweeglijke gemiddelden
        if df is not None:
            df['SMA_50'] = ta.trend.sma_indicator(df['close'], window=50)
            df['SMA_20'] = ta.trend.sma_indicator(df['close'], window=20)
            df['SMA_200'] = ta.trend.sma_indicator(df['close'], window=200)

            # Relatieve sterkte-index (RSI)
            df['RSI'] = ta.momentum.rsi(df['close'], window=14)

            # Moving Average Convergence Divergence (MACD)
            df['MACD'] = ta.trend.macd(df['close'])
            df['MACD_signal'] = ta.trend.macd_signal(df['close'])

            # Bollinger Bands
            bollinger = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
            df['Bollinger_High'] = bollinger.bollinger_hband()
            df['Bollinger_Low'] = bollinger.bollinger_lband()

            df['EMA_8'] = ta.trend.ema_indicator(df['close'], window=8)
            df['EMA_13'] = ta.trend.ema_indicator(df['close'], window=13)
            df['EMA_21'] = ta.trend.ema_indicator(df['close'], window=21)
            df['EMA_55'] = ta.trend.ema_indicator(df['close'], window=55)

            df['EMA_8_above_EMA_13'] = df['EMA_8'] > df['EMA_13']
            df['EMA_13_above_EMA_21'] = df['EMA_13'] > df['EMA_21']
            df['EMA_21_above_EMA_55'] = df['EMA_21'] > df['EMA_55']

            df['EMA_above'] = (df['EMA_8_above_EMA_13'] &
                               df['EMA_13_above_EMA_21'] &
                               df['EMA_21_above_EMA_55']).rolling(window=20).sum() == 20

            df['EMA_below'] = (~df['EMA_8_above_EMA_13'] &
                               ~df['EMA_13_above_EMA_21'] &
                               ~df['EMA_21_above_EMA_55']).rolling(window=20).sum() == 20

            df['volume_MA'] = df['volume'].rolling(window=20).mean()
            df['Bullish'] = (df['EMA_8'] > df['EMA_13']) & (df['EMA_13'] > df['EMA_21']) & (df['EMA_21'] > df['EMA_55'])
            df['Bearish'] = (df['EMA_8'] < df['EMA_13']) & (df['EMA_13'] < df['EMA_21']) & (df['EMA_21'] < df['EMA_55'])
            df['Buy Signal Long'] = df['EMA_above']
            df['Buy Signal Short'] = df['EMA_below']

            return df


    async def check_signals(self, df):
        if df is not None:
            last_index = df.index[-1]
            last_row = df.iloc[-1]
            
            print("Laatste data:")
            print(last_row, last_index)
            print('--------------------------------------------------------')
            
            # Going long
            if last_row['Buy Signal Long'] and last_row['RSI_Overbought'] != True:
                if self.check_balance('EUR'):
                    market = last_row['market']
                    amount = round(float(self.check_balance('EUR')) * 0.1 / self.get_market_price(market), 2)

                    self.place_long_position(symbol=market, amount=amount, stop_loss_percentage=4, take_profit_percentage=6)

                else:
                    error_message = "Ontoereikend cashsaldo in portfilio, laad geld op"
                    await self.send_telegram_message(error_message)


    async def main(self, bot):
        for i in self._markets:
            df = bot.get_bitvavo_data(market=i, interval='1h', limit=1440)
            df = bot.add_indicators(df)
            data_complete = bot.generate_signals(df)
            await bot.check_signals(data_complete)


if __name__ == '__main__':
    bot = apibot(file_path=file_path, markets=['AVAXEUR', 'BEAMEUR', 'ARBEUR', 'INJEUR', 'SOLEUR', 'ADAEUR', 'STXEUR', 'ADAEUR'])
    asyncio.run(bot.main(bot))


