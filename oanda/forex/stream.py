import json
import websocket

# Replace with your OANDA account details
ACCOUNT_ID = '001-001-11990763-002'
ACCESS_TOKEN = 'd5ab50dfd0d301a09f711297480b7f74-25c1b0e037151c7cbc7bebfac3049322'

# Define the WebSocket URL for the OANDA API
STREAM_DOMAIN = 'stream-fxtrade.oanda.com'
INSTRUMENTS = "EUR_USD"  # Add your desired instruments

def on_message(ws, message):
    data = json.loads(message)
    print(f"Received message: {data}")

def on_error(ws, error):
    print(f"Error: {error}")
    

def on_close(ws, close_status_code, close_msg):
    print(f"Connection closed with status code: {close_status_code} and message: {close_msg}")

def on_open(ws):
    print("Connection opened")

if __name__ == "__main__":
   
    url = f"wss://{STREAM_DOMAIN}/v3/accounts/{ACCOUNT_ID}/pricing/stream?instruments={INSTRUMENTS}"
    print(f"Connecting to: {url}")
    headers = {
        'Authorization': f'Bearer {ACCESS_TOKEN}',
    }
    print(f"Using headers: {headers}")

    ws = websocket.WebSocketApp(url,
                                header=headers,
                                on_open=on_open,
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close)
    
    ws.run_forever()
