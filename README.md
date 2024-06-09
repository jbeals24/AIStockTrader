# AIStockTrader

This repository holds a number of files used to stream real-time stock, crypto, and forex data. It uses an LSTM time series machine learning algorithm to create an automated AI stock market trader. The `track.py` python file in the main directory creates a websocket stream to track every price change of a stock of your choice. To run the file, simply execute `python3 track.py` in the terminal. Assuming you have the correct packages installed, you will be prompted to enter the ticker of the stock you would like to track.

After that, a text file will be automatically generated with the name of the stock and three random ASCII characters appended to it. You can see the generated files in the `data` directory. You will then be prompted to press `y` if you would like to make live trades, or simply press enter to not. The stock price streaming data will be printed to the console and written to the created file.

# Overview of `binaryPredict.py`

## Imports and Dependencies
```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.utils import class_weight
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Concatenate
```

# Data Preparation
The `createSet` function is responsible for preparing the dataset for training and testing. It takes the following steps:

* Calculates the mean price of the stock.
* Constructs input sequences of price changes, corresponding binary labels, and scales the data for more efficient training.
* Calculates the distance of the last price in the sequence from the mean price.
* Splits the data into training and testing sets and shuffles it to prevent overfitting.

# Model Creation
The create_model function creates and compiles an LSTM model if a pre-trained model does not already exist. LSTM models are designed to remember long-term dependencies in sequential data. The model consists of two inputs: time series data and the distance from the mean price. 

# Model training and Prediction
The predict function trains the model using the training data and evaluates its performance on the testing data. It uses early stopping and learning rate reduction as callbacks to improve training efficiency and prevent overfitting.

# Main Function
The `main` function is the entry point of the script. It handles file input, data preprocessing, model training, and evaluation. The function runs a main loop testing the models accuracy with different epochs and lenghts of input to converge on the best performing model. 

