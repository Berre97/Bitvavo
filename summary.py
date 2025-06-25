import json
from datetime import datetime, timedelta
from telegram import Bot
import os

now = datetime.now()
past_week = now-timedelta(days=7)
buyorders = os.getenv("FILE_PATH_BUYORDERS")
api_keys = json.loads(os.getenv('API_KEYS'))
weekly_profit = []
total_profit = []
token = api_keys['token']
chat_id = api_keys["chat_id"]
bot = Bot(token=token)

with open(buyorders, "r") as f:
    data = json.load(f)
    for order in data:
        if "sold" in data:
            if order["date"] >= past_week
                weekly_profit.append(order['eur_profit'])
            total_profit.append(order['eur_profit']) 
        
if total_profit:
    bot.send_message(f"Totaal gerealiseerde winst: €{round(sum(total_profit),2)}")
    
if weekly_profit:
    total_weekly_profit = round(sum(weekly_profit),2)
    avg_weekly_profit_per_trade = round(total_weekly_profit / len(weekly_profit),2)
    bot.send_message(f"Weekoverzicht\n"
                     f"Winst: €{total_weekly_profit}\n"
                     f"Gemmidelde winst per trade: €{avg_weekly_profit_per_trade}")


