
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
from scipy.signal import argrelextrema

api_keys= json.loads(os.getenv('API_KEYS'))

bitvavo = Bitvavo({
    'APIKEY': api_keys['API_KEY'],
    'APISECRET': api_keys['APISECRET'],
    'RESTURL': api_keys['RESTURL'],
    'WSURL': api_keys['WSURL']
})

token = Bot(token=api_keys['token'])
chat_id = api_keys["chat_id"]


class apibot():
    def __init__(self, file_path, markets):
        self._markets = markets
        self._file_path = file_path


    async def send_telegram_message(self, message):
      try:
          await token.send_message(chat_id=chat_id, text=message, read_timeout=20)
      except TimeoutError:
        print("Failed to send message due to timeout.")


    def load_data(self, file_path):
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)

                return data
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                print('Error loading json data: {e}')
                return []


    def update_file(self, file_path, order):
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    print(data)

                    if not isinstance(data, list):
                        data = []

            except json.JSONDecodeError:
                data = []
        else:
            data = []


        if order['type'] == "Sold":
            for i in data:
                if i['order'] == order['order']:
                    i.update(order)

        elif order['type'] == 'Stoploss':
            for i in data:
                if i['order'] == order['order']:
                    i.update(order)

        else:
            data.append(order)

        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)

    def identify_zones(self, levels, margin=1.5):
        levels = sorted(levels)
        zones = []
        current_zone = []
    
        for level in levels:
            if not current_zone:
                current_zone.append(level)
            elif level <= current_zone[-1] * (1 + margin / 100):
                current_zone.append(level)
            else:
                zones.append((min(current_zone), max(current_zone)))
                current_zone = [level]
    
        if current_zone:
            zones.append((min(current_zone), max(current_zone)))

        return zones

    def get_bitvavo_data(self, market, interval, limit):
      response = bitvavo.candles(market, interval, {'limit': limit})
      data = pd.DataFrame(response, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
      data = data.drop(data.index[500:])
      data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
      data[['open', 'high', 'low', 'close', 'volume']] = data[['open', 'high', 'low', 'close', 'volume']].apply(pd.to_numeric)
      data = data.set_index('timestamp')
      data = data.sort_index()
      data['market'] = market
      data['max'] = data.iloc[argrelextrema(data['close'].values, np.greater_equal, order=3)[0]]['close']
      data['min'] = data.iloc[argrelextrema(data['close'].values, np.less_equal, order=3)[0]]['close']

      return data

    def generate_signals(self, df):
      for col in df.columns:
        if df[col].isnull().any():
          print('Nog niet genoeg data')
          break

        else:
          # Golden Cross / Death Cross
          df['Golden_Cross'] = np.where((df['SMA_50'] > df['SMA_200']) & (df['SMA_50'].shift(1) <= df['SMA_200'].shift(1)), True, False)
          df['Death_Cross'] = np.where((df['SMA_50'] < df['SMA_200']) & (df['SMA_50'].shift(1) >= df['SMA_200'].shift(1)), True, False)

          df['up_trend'] = np.where(df['SMA_20'] > df['SMA_50'], True, False) #Op SOLANA sma20 over 50 anders evt #200, 50
          df['down_trend'] = np.where(df['SMA_50'] > df['SMA_20'], True, False) #50, 200

          # RSI Overbought / Oversold
          df['RSI_Overbought'] = np.where(df['RSI'] >= 60, True, False)
          df['RSI_Oversold'] = np.where(df['RSI'] <= 35, True, False)
          df['RSI_Overbought_MACD'] = np.where(df['RSI'] >= 65, True, False)
          df['RSI_Oversold_MACD'] = np.where(df['RSI'] <= 45, True, False)

          # MACD Crossovers
          df['MACD_Bullish'] = np.where((df['MACD'] > df['MACD_signal']) & (df['MACD'].shift(1) <= df['MACD_signal'].shift(1)), True, False)
          df['MACD_Bearish'] = np.where((df['MACD'] < df['MACD_signal']) & (df['MACD'].shift(1) >= df['MACD_signal'].shift(1)), True, False)


          # Bollinger Bands Cross
          df['Bollinger_Breakout_High'] = np.where((df['close'] > df['Bollinger_High']), True, False)
          df['Bollinger_Breakout_Low'] = np.where((df['close'] < df['Bollinger_Low']), True, False)


          return df

    def add_indicators(self, df):
        # Beweeglijke gemiddelden

        df['SMA_50'] = ta.trend.sma_indicator(df['close'], window=50)
        df['SMA_20'] = ta.trend.sma_indicator(df['close'], window=20)
        df['SMA_200'] = ta.trend.sma_indicator(df['close'], window=200)
        df['EMA_50'] = ta.trend.ema_indicator(df['close'], window=50)
        df['EMA_200'] = ta.trend.ema_indicator(df['close'], window=200)

        # Relatieve sterkte-index (RSI)
        df['RSI'] = ta.momentum.rsi(df['close'], window=14)

        # Moving Average Convergence Divergence (MACD)
        df['MACD'] = ta.trend.macd(df['close'])
        df['MACD_signal'] = ta.trend.macd_signal(df['close'])

        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
        df['Bollinger_High'] = bollinger.bollinger_hband()
        df['Bollinger_Low'] = bollinger.bollinger_lband()

        # Commodity Channel Index (CCI)
        df['CCI'] = ta.trend.cci(df['high'], df['low'], df['close'], window=20)

        # Stochastische Oscillator
        stoch = ta.momentum.stoch(df['high'], df['low'], df['close'], window=14, smooth_window=3)
        df['Stoch_K'] = stoch
        df['Stoch_D'] = ta.momentum.stoch_signal(df['high'], df['low'], df['close'], window=14, smooth_window=3)


        # On Balance Volume (OBV)
        df['OBV'] = ta.volume.on_balance_volume(df['close'], df['volume'])

        # Ichimoku Cloud
        ichimoku = ta.trend.IchimokuIndicator(df['high'], df['low'], window1=9, window2=26, window3=52)
        df['Ichimoku_Conversion'] = ichimoku.ichimoku_conversion_line()
        df['Ichimoku_Base'] = ichimoku.ichimoku_base_line()
        df['Ichimoku_A'] = ichimoku.ichimoku_a()
        df['Ichimoku_B'] = ichimoku.ichimoku_b()

        return df

    # Functie om signalen te controleren
    async def check_signals(self, df):
        resistance_levels = df['max'].dropna().tolist()
        support_levels = df['min'].dropna().tolist()
        resistance_zones = self.identify_zones(resistance_levels)
        support_zones = self.identify_zones(support_levels)

        last_index = df.index[-1]
        last_row = df.iloc[-1]

        print("Laatste data:")
        print(last_row, last_index)
        print('--------------------------------------------------------')

        indicators_buy = df.loc[last_index, ['RSI_Oversold', 'Bollinger_Breakout_Low']]
        indicators_MACD = df.loc[last_index, ['MACD_Bullish', 'RSI_Oversold_MACD']]
        indicators_sell = df.loc[last_index, ['RSI_Overbought']]

        order_number = random.randint(1000, 9999)
        is_within_support_zone = False
            for zone in support_zones:
                if zone[0] <= last_row['close'] <= zone[1]:
                    is_within_support_zone = True
                    break
                    
        if indicators_buy.all() and float(last_row['close']) is_within_support_zone:
            buy_message = f"Koop:\n {last_row['market']} {last_row['close']}"
            buy_order = {'type': 'Bought', 'symbol': last_row['market'],
                                                'time': str(last_index.to_pydatetime()),
                                                'closing_price': float(last_row['close']),
                                                'order': order_number, 'strategy': 'RSI_Oversold, up_trend'}

      
            print(buy_order)
            self.update_file(self._file_path, buy_order)
            await self.send_telegram_message(buy_message)

        elif indicators_MACD.all() and float(last_row['close']) is_within_support_zone:
            buy_message = f"Koop:\n {last_row['market']} {last_row['close']}"
            buy_order = {'type': 'Bought', 'symbol': last_row['market'],
                         'time': str(last_index.to_pydatetime()),
                         'closing_price': float(last_row['close']),
                         'order': order_number, 'strategy': 'MACD_Bullish, RSI_Oversold_MACD'}

            self.update_file(self._file_path, buy_order)
            print(buy_order)
            await self.send_telegram_message(buy_message)

        else:
          print('Geen koopsignalen gevonden')


        #take profit / Stop loss
        if self.load_data(self._file_path) is not None:
            for i in self.load_data(self._file_path):
                stop_limit = float(i['closing_price']) * 0.92
                if i['type'] == 'Bought' and i['symbol'] == last_row['market'] and \
                        float(last_row['close']) <= float(i['closing_price']) * 0.95 >= stop_limit:
                                   
                    percentage_loss = (float(i['closing_price']) - float(last_row['close'])) * 100 / float(i['closing_price'])
                    percentage_loss = format(percentage_loss, ".2f")

                    stoploss_message = f"Stoploss:\n {last_row['market']} prijs: {last_row['close']}\n" \
                                       f"percentage loss: {percentage_loss}"

                    stoploss_order = {'type': 'Stoploss', "symbol": last_row['market'], 'order': i['order'],
                                                           'time': str(last_index.to_pydatetime()),
                                                           'closing_price': float(last_row['close']),
                                                           'aankoopprijs': float(i['closing_price']),
                                                           'aankoopdatum': str(i['time']),
                                                           'percentage_loss': percentage_loss}
        
                    print(stoploss_order)
                    self.update_file(self._file_path, stoploss_order)
                    await self.send_telegram_message(stoploss_message)

                elif indicators_sell.all():                                                               
                    if i['type'] == 'Bought' and i['symbol'] == last_row['market'] and \
                            float(last_row['close']) >= float(i['closing_price']) * 1.15:
                                
                        percentage = (float(last_row['close']) - float(i['closing_price'])) / float(i['closing_price']) * 100
                        percentage = format(percentage, ".2f")
                        sell_order = {'type': 'Sold', 'symbol': last_row['market'],
                                                           'order': i['order'],
                                                           'time': str(last_index.to_pydatetime()),
                                                           'closing_price': float(last_row['close']),
                                                           'aankoopprijs': float(i['closing_price']),
                                                           'aankoopdatum': str(i['time']),
                                                           'percentage_gain': percentage}

                        sell_message = f"Verkoop:\n {last_row['market']} prijs: {last_row['close']} " \
                                       f"aankoopkoers: {float(i['closing_price'])}\n " \
                                       f"percentage gained: {percentage}"

                        print(sell_order)
                        self.update_file(self._file_path, sell_order)
                        await self.send_telegram_message(sell_message)

        else:
            print('Geen verkoopsignalen gevonden')


    async def main(self, bot):
        # start_time = datetime.now()
        # end_time = start_time + timedelta(hours=15)

        # while datetime.now() < end_time:
        for i in self._markets:
            df = bot.get_bitvavo_data(interval='1h', limit=1440, market=i) 
            df = bot.add_indicators(df)
            data_complete = bot.generate_signals(df)
            await bot.check_signals(data_complete)

            # time.sleep(5)

if __name__ == '__main__':
    # file_path = 'CryptoOrders.json'
    file_path = os.getenv('FILE_PATH')
    bot = apibot(file_path=file_path, markets=['SOL-EUR', 'ADA-EUR', 'AVAX-EUR', 'ICP-EUR',
                                               'COTI-EUR', 'JUP-EUR', 'MANTA-EUR', 'CFX-EUR'])
    asyncio.run(bot.main(bot))

