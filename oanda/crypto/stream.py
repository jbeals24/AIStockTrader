import json
from oandapyV20 import API
from oandapyV20.exceptions import V20Error
from oandapyV20.endpoints.pricing import PricingStream

def stream_crypto_pricing(account_id, access_token, instruments):
    api = API(access_token=access_token, environment="live")
    s = PricingStream(accountID=account_id, params={"instruments": instruments})
    try:
        for data in api.request(s):
            # Extract bid and ask prices directly from the dictionary
            print(data)
    except V20Error as e:
        print("Error: {}".format(e))

if __name__ == "__main__":
    # Replace the placeholders with your actual account ID, access token, and list of cryptocurrency instruments
    accountID = "001-001-11990763-002"
    access_token = "d5ab50dfd0d301a09f711297480b7f74-25c1b0e037151c7cbc7bebfac3049322"
    instruments = "BTC_USD"  # Example list of cryptocurrency instruments
    
    # Start streaming cryptocurrency pricing data
    stream_crypto_pricing(accountID, access_token, instruments)
