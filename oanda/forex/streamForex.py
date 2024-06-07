import json
from oandapyV20 import API
from oandapyV20.exceptions import V20Error
from oandapyV20.endpoints.pricing import PricingStream
import random
import string
import sys

firstAsk = 0
firstBid = 0
counter= 0
def stream_pricing_data(account_id, access_token, instruments, file):
    global counter
    api = API(access_token=access_token, environment="live")
    s = PricingStream(accountID=account_id, params={"instruments": instruments})
    try:
        for data in api.request(s):
            # Extract bid and ask prices directly from the dictionary
            if 'bids' in data:
                bid_price = float(data['bids'][0]['price'])
                ask_price = float(data['asks'][0]['price'])

                margin = ask_price - bid_price
                file.write(f'{bid_price}, {ask_price}, {margin}\n')
                print(f"Bid Price: {bid_price}, Ask Price: {ask_price} Margin: {margin}")
                counter += 1

                if (counter == 3000): sys.exit()
            
    except V20Error as e:
        print("Error: {}".format(e))

if __name__ == "__main__":
    # Replace the placeholders with your actual account ID and access token
    accountID = "YOUR ACCOUNT ID HERE"
    access_token = "YOUR ACCESS TOKEN HERE"
    instruments = "EUR_USD"  # Example list of instruments
    
    filePath = 'data/'
    filePath += instruments
    for i in range(3):
        filePath += random.choice(string.ascii_letters)

    filePath += '.txt'
    file = open(filePath, 'w')
    # Start streaming pricing data
    stream_pricing_data(accountID, access_token, instruments, file)
