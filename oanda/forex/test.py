import json
from oandapyV20 import API
from oandapyV20.exceptions import V20Error
from oandapyV20.endpoints.pricing import PricingStream
import random
import string
import sys

# ACCOUNT_ID = '001-001-11990763-002'
# ACCESS_TOKEN = 'd5ab50dfd0d301a09f711297480b7f74-25c1b0e037151c7cbc7bebfac3049322'

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
    accountID = "001-001-11990763-002"
    access_token = "d5ab50dfd0d301a09f711297480b7f74-25c1b0e037151c7cbc7bebfac3049322"
    instruments = "EUR_USD"  # Example list of instruments
    
    filePath = 'data/'
    filePath += instruments
    for i in range(3):
        filePath += random.choice(string.ascii_letters)

    filePath += '.txt'
    file = open(filePath, 'w')
    # Start streaming pricing data
    stream_pricing_data(accountID, access_token, instruments, file)
