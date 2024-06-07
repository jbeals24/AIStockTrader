import json
from oandapyV20 import API
from oandapyV20.exceptions import V20Error
from oandapyV20.endpoints.pricing import PricingStream

def stream_crypto_pricing(account_id, access_token, instruments):
    api = API(access_token=access_token, environment="live") # change to non live account here if your subscription is for demo trading
    s = PricingStream(accountID=account_id, params={"instruments": instruments})
    try:
        for data in api.request(s):
            print(data)
    except V20Error as e:
        print("Error: {}".format(e))

if __name__ == "__main__":
    # Replace the placeholders with your actual account ID, access token, and list of cryptocurrency instruments
    accountID = "YOUR ACCOUNT ID HERE"
    access_token = "YOUR ACCESS TOKEN HERE"
    instruments = "BTC_USD"  # Example list of cryptocurrency instruments
    
    # Start streaming cryptocurrency pricing data
    stream_crypto_pricing(accountID, access_token, instruments)
