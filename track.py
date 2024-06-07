import alpaca_trade_api as tradeapi
import asyncio
from alpaca_trade_api.stream import Stream
import time
import sys
import subprocess
import random
import string
import pickle
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import datetime
import pytz

# Your paper trading API keys
APCA_API_KEY_ID = 'PK9TX08O9SCWDVBAJOI8'
APCA_API_SECRET_KEY = 'MyRdaMxjdOu2E101e2I3SRlAfaqxyCpdeeYu0Uam'
APCA_API_PAPER_URL = 'https://paper-api.alpaca.markets'
DATA_FEED = 'iex'

# Initialize the API client for paper trading
api = tradeapi.REST(APCA_API_KEY_ID, APCA_API_SECRET_KEY, APCA_API_PAPER_URL, api_version='v2')
account = api.get_account()

#===second account===
# keyID2 = 'PKG3CT5TR57CZG4UPFB5'
# secretKey2 = 'PStZwGCDGecYgBmZV2wu88oEOnY5g8nAPB8TJzkb'

# api2 = tradeapi.REST(keyID2, secretKey2, APCA_API_PAPER_URL, api_version='v2')
# account2 = api2.get_account() 



# with open('scaler.pkl', 'rb') as f:
#     scaler = pickle.load(f)

priceString = []

lastPrice = 0
originalBuyPrice = 0
originalSellPrice = 0
shortBoughtBack = False
soldBuy = False
counter = 0
shares = 1
liveTrading = False
BuyInProgress = False
SellInProgress = False

# get which stock to track
ticker = input("What stock would you like to trade? ")
model = load_model(f'test{ticker}Model.keras')
input_shape = model.input_shape
print("Model expects input shape:", input_shape)
tmp = input("Do you want to make live trades? y for yes enter for no ")
if (tmp == 'y'): 
    liveTrading = True

filePath = 'data/'
filePath += ticker
for i in range(3):
    filePath += random.choice(string.ascii_letters)

filePath += '.txt'
file = open(filePath, 'w')

lock = asyncio.Lock()

def printPower():
    print(f'Buying Power: ${account.buying_power}')

def get_yesterday_closing_price(ticker):
    # Get the current date in UTC
    now = datetime.datetime.now(pytz.utc)
    
    # Get the date for yesterday
    yesterday = now - datetime.timedelta(days=1)
    yesterday_str = yesterday.strftime('%Y-%m-%d')
    
    # Fetch aggregate data for the given ticker for yesterday
    aggs = api.get_aggs(
        ticker,
        1,  # Timespan value (e.g., 1 minute)
        yesterday_str,
        yesterday_str
    )
    
    # Get the closing price from the last aggregate data of the day
    if aggs:
        closing_price = aggs[-1].c
    else:
        closing_price = None
    
    return closing_price

def getStockPrice(ticker):
    return api.get_latest_trade(ticker)

def printActiveOrders():
    active_orders = api.list_orders(status='open')

    for order in active_orders:
        print(order)
    
def checkOrder(api, orderID):
    while True:
        order = api.get_order(orderID)
        if order.status == 'filled':
            print(f"Order {orderID} is filled.")
            return True
        elif order.status == 'rejected' or order.status == 'canceled':
            print(f"Order {orderID} was not filled. Status: {order.status}")
            return False
        time.sleep(1) 

def submitBuyOrder(ticker, qty, api):
    order = api.submit_order(
    symbol=ticker,
    qty=qty,
    side='buy',
    type='market',
    time_in_force='gtc' 
    )
    print(f"Market order submitted to buy {qty} share(s) of {ticker}.")

    order_filled = False
    while not order_filled:
        # Retrieve the updated order information
        updated_order = api.get_order(order.id)
        if updated_order.status == 'filled':
            order_filled = True
            fill_price = updated_order.filled_avg_price
        else:
            time.sleep(.15)

    return order.id, fill_price
    
def submitTrailingStopOrder(ticker,qty,trail_percent):
    trailing_stop_order = api.submit_order(
    symbol=ticker,
    qty=qty,
    side='sell',
    type='trailing_stop',
    trail_percent=trail_percent,
    time_in_force='gtc'
    )
    print(f"Trailing stop loss order submitted with a {trail_percent}% trail.")

def sellStock(ticker, qty, api):
    try:
        # price = getStockPrice(ticker)
        order = api.submit_order(
            symbol=ticker,
            qty=qty,
            side='sell',
            type='market',
            time_in_force='gtc'
        )
        print(f"Sell order submitted for {qty} share(s) of {ticker}.")
        order_filled = False
        while not order_filled:
            # Retrieve the updated order information
            updated_order = api.get_order(order.id)
            if updated_order.status == 'filled':
                order_filled = True
                fill_price = updated_order.filled_avg_price
            else:
                time.sleep(.15)

        return order.id, fill_price

    except Exception as e:
        print(f"An error occurred: {e}")

def liquidate_on_price_increase(ticker, short_entry_price, qty_to_cover):
    has_increased = False
    priceLine = short_entry_price
    while not has_increased:
        current_price = float(api2.get_latest_trade(ticker).price)

        if current_price > priceLine:
            print(f"Price of {ticker} has increased from the entry price. Liquidating the short position.")
            api2.submit_order(
                symbol=ticker,
                qty=qty_to_cover,
                side='buy',
                type='market',
                time_in_force='gtc'
            )
            has_increased = True
            print(f"Short position for {ticker} has been liquidated.")
            break
        else:
            priceLine = current_price
            time.sleep(.5)

def runItBack(trade):
    global counter, shortBoughtBack, soldBuy, originalBuyPrice, shares
    if (counter == 2): sys.exit()

    orderID, originalBuyPrice = submitBuyOrder(trade.symbol, shares, api)
    sellStock(trade.symbol, shares, api2)
    shortBoughtBack = False
    soldBuy = False
    counter += 1

def recipt():
    trades = api.list_trades(limit=2)
    for trade in trades:
        print(f"Trade ID: {trade.id}, Symbol: {trade.symbol}, Qty: {trade.qty}, Price: {trade.price}")

def firstPurchase(symbol):
    global originalBuyPrice, originalSellPrice, shares
    buyID, originalSellPrice = sellStock(symbol, shares, api2)
    sellID, originalBuyPrice = submitBuyOrder(symbol, shares, api)
    
    return originalBuyPrice

def makePrediction(priceString):
    global model, ticker, account, shares, liveTrading, api, BuyInProgress, SellInProgress
    buy = False
    distFromMean = np.array([int(priceString[-1]) - dailyMean]).reshape(1, 1)

    price_changes = np.diff(priceString)  # Calculating price changes
    if(len(price_changes) <10): price_changes = np.insert(price_changes, 0,0)
    price_changes = price_changes.reshape(-1, 1)
    print(price_changes)

    scaler = MinMaxScaler(feature_range=(0, 1))  # Creating scaler
    price_changes_scaled = scaler.fit_transform(price_changes)
    price_changes_scaled = price_changes_scaled.reshape(1, 10, 1)

    prediction = model.predict([price_changes_scaled,distFromMean])  # Making prediction
    predicted_direction = 'Up' if prediction[0][0] > 0.5 else 'Down'  # Determining predicted direction
    if (predicted_direction == 'Up'):
        buy = True
    print(f'Model predicts the price will move: {predicted_direction}')
    if (liveTrading):
        if (buy):
            submitBuyOrder(ticker,shares,api)
            BuyInProgress = True
        else:
            sellStock(ticker,shares,api)
            SellInProgress = True


async def on_price_update(trade):
    global priceString, lastPrice, counter, file, BuyInProgress, SellInProgress

    file.write(str(trade.price) + '\n')
    if (lastPrice == 0):
        lastPrice = trade.price
        iteratingPrice = lastPrice
        return 

    if (trade.price > lastPrice):
        counter += 1
        priceString.append(trade.price)
        print(f'{len(priceString)}: {lastPrice} => ({trade.price})')
        # print(f"price increased from {lastPrice} => {trade.price}")
    else:
        counter += 1
        priceString.append(trade.price)
        print(f'{len(priceString)}: {lastPrice} => ({trade.price})')
        # print(f"price decreased from {lastPrice} => {trade.price}")
        
    lastPrice = trade.price

    if (len(priceString) == 10):
        if (BuyInProgress): 
            sellStock(ticker,shares,api)
            BuyInProgress = False
        if (SellInProgress):
            submitBuyOrder(ticker,shares,api)
            SellInProgress = False
        makePrediction(priceString)
        priceString = []
        priceString.append(lastPrice)

    if (counter == 5000): sys.exit()
    # MAX DATA SIZE^^^^
    
    

def setup_stream(symbol):
    
    stream = Stream(APCA_API_KEY_ID,
                    APCA_API_SECRET_KEY,
                    base_url=APCA_API_PAPER_URL,
                    data_feed=DATA_FEED)
    stream.subscribe_trades(on_price_update, symbol)

    return stream

def start_stream(ticker):
    
    stream = setup_stream(ticker)
    loop = asyncio.get_event_loop()
    loop.run_until_complete(stream._run_forever())


def grabInput():
    global shares
    ticker = input("What stock would you like to trade? ")
    shares = int(input(f"how many shares of {ticker} do you want? "))
    return ticker, shares

dailyMean = 0
if __name__ == "__main__":
    dailyMean = float(input(f'Enter todays mean price for {ticker}: '))
    stream = start_stream(ticker)
    
    