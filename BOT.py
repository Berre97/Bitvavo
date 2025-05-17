from python_bitvavo_api.bitvavo import Bitvavo
import json
import os
import pandas as pd
import numpy as np
import asyncio
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import ApplicationBuilder, CallbackQueryHandler, ContextTypes, Job, CallbackContext, MessageHandler, filters, ContextTypes
import asyncio
import sys
import ta
from datetime import datetime

api_keys = json.loads(os.getenv('API_KEYS'))
api_key = api_keys['API_KEY']
api_secret = api_keys['API_SECRET']
token = api_keys['token']
chat_id = api_keys["chat_id"]
app = ApplicationBuilder().token(token).build()

bitvavo = Bitvavo({
    'APIKEY': api_keys['API_KEY'],
    'APISECRET': api_keys['API_SECRET'],
    'RESTURL': api_keys['RESTURL'],
    'WSURL': api_keys['WSURL'],

})


class apibot():
    def __init__(self):
        self._bot = None
        self._buy_signals = {}
        self._order = None
        self._index = 0
        self._chat_id = -4717875969
        self._msg_id = None

    async def timeout_sessie(self, chat_id, bot):
        try:
            await asyncio.sleep(900)  # 15 minuten
            await bot.send_message(chat_id=chat_id, text="⏰ Tijd is verstreken. Order niet meer uitvoerbaar.")
            sys.exit()  # Hele programma stoppen
        except asyncio.CancelledError:
            pass


    def maak_knoppen(self):
        return InlineKeyboardMarkup([
            [InlineKeyboardButton("Ja", callback_data="ja")],
            [InlineKeyboardButton("Neen", callback_data="nee")]
        ])

    async def stuur_order(self, application):
        self._bot = application.bot
        if self._buy_signals and self._index < len(self._buy_signals):
            key, value = list(self._buy_signals.items())[self._index]
            prijs_per_eenheid = value['huidige_marktprijs']
            markt = key

            buy_message = f"Koopsignaal gedetecteerd:\nValuta: {markt}\nPrijs per eenheid: €{round(prijs_per_eenheid,2)}\n\n " \
                          f"Totaalbedrag: €{value['orderprijs']}\n" \
                          f"Je hebt €{self.check_balance('EUR')} beschikbaar, wil je deze koop bevestigen?"

            print(key, value['hoeveelheid'], prijs_per_eenheid, value['stop_loss'], value['take_profit'], value['orderprijs'])

            msg = await self._bot.send_message(chat_id=self._chat_id, text=buy_message, reply_markup=self.maak_knoppen())
            self._msg_id = msg.message_id
            asyncio.create_task(self.timeout_sessie(self._chat_id, bot))

        else:
            await self._bot.send_message(chat_id=self._chat_id, text="Er zijn geen koopsignalen meer.")
            sys.exit()

    async def tekst_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        antwoord = update.message.text.lower()

        if antwoord == "ja":
            key, value = list(self._buy_signals.items())[self._index]
            market = key
            amount = value['hoeveelheid']
            side = 'buy'
            stop_loss_price = value['stop_loss']
            stop_loss_limit = value['stop_limit']
            take_profit_price = value['take_profit']

            await context.bot.send_message(text="Bezig met plaatsen van order...", chat_id=self._chat_id, message_id=self._msg_id)
            await self.place_market_order(market, amount, side, stop_loss_price, stop_loss_limit, take_profit_price)


        if antwoord == "nee":
            self._index += 1
            await self.stuur_order(app)


    async def knop_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        await query.answer()
        keuze = query.data

        if keuze == "ja":
            key, value = list(self._buy_signals.items())[self._index]
            market = key
            amount = value['hoeveelheid']
            side = 'buy'
            stop_loss_price = value['stop_loss']
            take_profit_price = value['take_profit']
            stop_loss_limit = value["stop_limit"]

            await self._bot.send_message(chat_id=self._chat_id, text=f"Bezig met plaatsen van order.....")
            await self.place_market_order(market, amount, side, stop_loss_price, stop_loss_limit, take_profit_price)
            sys.exit()

        elif keuze == "nee":
            self._index += 1
            await self.stuur_order(app)


    async def place_stop_loss(self, symbol, amount, stop_loss_price, stop_loss_limit):
        print(stop_loss_limit, stop_loss_price, amount)
        stop_loss_order = bitvavo.placeOrder(symbol, 'sell', 'stopLossLimit', {
             'amount': amount,
             'price': stop_loss_limit,
             'triggerType': 'price',
             'stopPrice': stop_loss_price,
             'triggerAmount': stop_loss_price,
             'triggerReference': 'bestBid'
        })

        if 'error' in stop_loss_order:
            print(f"Fout bij plaatsen stop-loss order: {stop_loss_order['error']}")
            await self._bot.send_message(chat_id=self._chat_id, text=f"Fout bij het plaatsen van stop-loss order: {stop_loss_order['error']}")
        else:
            print(f"Stop-loss order succesvol geplaatst!")
            await self._bot.send_message(chat_id=self._chat_id,
                                         text=(f"Stop-loss order succesvol geplaatst: {stop_loss_order}"))
            return stop_loss_order

    async def place_take_profit(self, symbol, amount, take_profit_price):
        take_profit_order = bitvavo.placeOrder(symbol, 'sell', 'limit', {
        'amount': amount,
        'price': take_profit_price
        })

        if 'error' in take_profit_order:
            print(f"Fout bij plaatsen take-profit order: {take_profit_order['error']}")
            await self._bot.send_message(chat_id=self._chat_id,
                                         text=f"Fout bij het plaatsen van take-profit order: {take_profit_order['error']}")
        else:
            print(f"Take-profit order succesvol geplaatst: {take_profit_order}")
            await self._bot.send_message(chat_id=self._chat_id,
                                         text=(f"Take-profit order succesvol geplaatst: {take_profit_order}"))
            return take_profit_order

    async def place_market_order(self, symbol, amount, side, stop_loss_price, stop_loss_limit, take_profit_price):
        order = bitvavo.placeOrder(symbol, side, 'market', {'amount': amount})
        if 'error' in order:
            error_message = f"Fout bij plaatsen {side} order: {order['error']}"
            await self._bot.send_message(chat_id=self._chat_id, text=error_message)

        else:
            if await self.place_stop_loss(symbol, amount, stop_loss_price, stop_loss_limit) and\
                    await self.place_take_profit(symbol, amount, take_profit_price):
                success_message = f"{side.capitalize()} Order succesvol geplaatst!"
                print(success_message)
                await self._bot.send_message(chat_id=self._chat_id, text=success_message)

    def get_market_price(self, symbol):
        ticker = bitvavo.tickerPrice({'market': symbol})
        return float(ticker['price']) if 'price' in ticker else None

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


            # RSI Overbought / Oversold
            df['RSI_Overbought'] = np.where(df['RSI'] >= 55, True, False)
            df['RSI_Oversold'] = np.where(df['RSI'] <= 35, True, False)

            # MACD Crossovers
            df['MACD_Bullish'] = np.where(
                (df['MACD'] > df['MACD_signal']) & (df['MACD'].shift(1) <= df['MACD_signal'].shift(1)),
                True, False)
            df['MACD_Bearish'] = np.where(
                (df['MACD'] < df['MACD_signal']) & (df['MACD'].shift(1) >= df['MACD_signal'].shift(1)),
                True, False)

            # Bollinger Bands Cross
            df['Bollinger_Breakout_High'] = np.where((df['close'] > df['Bollinger_High']), True, False)
            df['Bollinger_Breakout_Low'] = np.where((df['close'] < df['Bollinger_Low']), True, False)

            df['EMA_above'] = (df['EMA_8_above_EMA_13'] &
                               df['EMA_13_above_EMA_21'] &
                               df['EMA_21_above_EMA_55']).rolling(window=5).sum() == 5

            df['EMA_below'] = (~df['EMA_8_above_EMA_13'] &
                               ~df['EMA_13_above_EMA_21'] &
                               ~df['EMA_21_above_EMA_55']).rolling(window=5).sum() == 5

            return df


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

    def check_signals(self, markets):
        for market in markets:
            df = self.get_bitvavo_data(market, '1h', 100)
            df = self.add_indicators(df)
            if df is not None:
                last_row = df.iloc[-1]
                # if last_row['EMA_above'] and last_row['RSI_Overbought'] != True:
                if self.check_balance('EUR'):
                    stop_loss_percentage = 4
                    take_profit_percentage = 6
                    eur_per_trade = 6
                    current_price = self.get_market_price(market)
                    quantity = round(eur_per_trade / current_price,2)
                    amount = round(quantity * current_price,2)
                    stop_loss_price = round(current_price / (1+(stop_loss_percentage/100)),4)
                    limit_price = round(stop_loss_price * 0.99, 4)
                    take_profit_price = round(current_price * (1+(take_profit_percentage/100)),4)
                    self._buy_signals[market] = {"type": "Long", "hoeveelheid": quantity, "orderprijs": amount,
                    "take_profit": take_profit_price, "stop_loss": stop_loss_price, "stop_limit": limit_price,
                    "huidige_marktprijs": current_price}

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
            data[['open', 'high', 'low', 'close', 'volume']] = data[
                ['open', 'high', 'low', 'close', 'volume']].apply(pd.to_numeric)
            data = data.set_index('timestamp')
            data = data.sort_index()
            data['market'] = market

            return data


def main(bot):
    markets = ['BEAM-EUR', 'ARB-EUR', 'INJ-EUR', 'SOL-EUR', 'ADA-EUR', 'STX-EUR']
    for market in markets:
        df = bot.get_bitvavo_data(market, '1h', 100)
        df = bot.add_indicators(df)
        if df is not None:
            last_row = df.iloc[-1]
            # if last_row['EMA_above'] and last_row['RSI_Overbought'] != True:
            if bot.check_balance('EUR'):
                stop_loss_percentage = 4
                take_profit_percentage = 6
                eur_per_trade = 6
                current_price = bot.get_market_price(market)
                quantity = round(eur_per_trade / current_price,2)
                amount = round(quantity * current_price,2)
                stop_loss_price = round(current_price / (1+(stop_loss_percentage/100)),4)
                limit_price = round(stop_loss_price * 0.99, 4)
                take_profit_price = round(current_price * (1+(take_profit_percentage/100)),4)
                bot._buy_signals[market] = {"type": "Long", "hoeveelheid": quantity, "orderprijs": amount,
                "take_profit": take_profit_price, "stop_loss": stop_loss_price, "stop_limit": limit_price,
                "huidige_marktprijs": current_price}

    app.add_handler(CallbackQueryHandler(bot.knop_handler))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, bot.tekst_handler))
    loop = asyncio.get_event_loop()
    loop.run_until_complete(bot.stuur_order(app))
    app.run_polling()

if __name__ == '__main__':
    bot = apibot()
    main(bot)

