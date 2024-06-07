import numpy as np
import pandas as pd
import random
import pickle
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import load_model

fileName = input("Name of file to train on: ")
file_path = 'data/' + fileName
ticker = fileName[:4]
data = pd.read_csv(file_path, header=None, names=['price'])
scaler = MinMaxScaler(feature_range=(0, 1))

original_prices = data['price'].values.reshape(-1, 1)
data['price'] = scaler.fit_transform(data[['price']])

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

def createSet(data, original_data, lookBack, stepsAhead):
    x_scaled, y_scaled, x_unscaled = [], [], []
    for index in range(len(data) - lookBack - stepsAhead + 1):
        x_scaled.append(data.iloc[index:index + lookBack].values.flatten().tolist())
        y_scaled.append(data.iloc[index + lookBack + stepsAhead - 1]['price'])
        x_unscaled.append(original_data[index:index + lookBack].flatten().tolist())  # store unscaled data

    x_scaled, y_scaled, x_unscaled = np.array(x_scaled), np.array(y_scaled), np.array(x_unscaled)
    
    # Shuffle data
    indices = np.arange(x_scaled.shape[0])
    np.random.shuffle(indices)
    
    return x_scaled[indices], y_scaled[indices], x_unscaled[indices]

def predict(model, numEpochs, X_train_scaled, X_test_unscaled, Y_train, X_test_scaled, Y_test, flag):
    model.fit(X_train_scaled, Y_train, epochs=numEpochs, batch_size=32, validation_split=0.2, verbose=1)

    Y_pred = model.predict(X_test_scaled)

    Y_test_actual = scaler.inverse_transform(Y_test.reshape(-1, 1))
    Y_pred_actual = scaler.inverse_transform(Y_pred)

    totalErr = 0
    counter = 0
    
    for i in range(len(Y_test_actual)):
        err = abs(Y_test_actual[i][0] - Y_pred_actual[i][0])
        totalErr += err
        if (flag):
            print(f"Price from {X_test_unscaled[i][-1]} => {Y_test_actual[i][0]:.2f}, Predicted Price: {Y_pred_actual[i][0]:.2f}")
            if (Y_test_actual[i][0] >= X_test_unscaled[i][-1] and Y_pred_actual[i][0] >= X_test_unscaled[i][-1]): counter += 1
            elif (Y_test_actual[i][0] <= X_test_unscaled[i][-1] and Y_pred_actual[i][0] <= X_test_unscaled[i][-1]): counter += 1
    if (flag):
        print(f'correct direction predictions: {counter} / {len(Y_test_actual)}')

    print(f'Epochs: {numEpochs}, total error: {totalErr}')
    return numEpochs, totalErr

def train(model, train_data, test_data, original_test, original_train, lookBack, lookAhead, epochs, flag):
    # Creating training and testing datasets
    X_train_scaled, Y_train, X_train_unscaled = createSet(train_data, original_train, lookBack, lookAhead)
    X_test_scaled, Y_test, X_test_unscaled = createSet(test_data, original_test, lookBack, lookAhead)
    
    if X_test_scaled.size == 0 or X_train_scaled.size == 0:
        raise ValueError("Insufficient data for the given lookBack and lookAhead values.")

    # Reshaping input to be [samples, time steps, features]
    X_train_scaled = np.reshape(X_train_scaled, (X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
    X_test_scaled = np.reshape(X_test_scaled, (X_test_scaled.shape[0], X_test_scaled.shape[1], 1))

    numEpochs, totalErr = predict(model, epochs, X_train_scaled, X_test_unscaled, Y_train, X_test_scaled, Y_test, flag)
    return totalErr, model

def main():
    # Try to load the last model
    try:
        model = load_model(f'test{ticker}Model.keras')
        input_shape = model.input_shape
        print(model)
    except Exception as e:
        print(f"Failed to load model: {e}. Creating a new one.")
        model = Sequential([
            LSTM(100, input_shape=(10, 1), return_sequences=True),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')

    # Splitting data
    split_ratio = 0.8
    split_index = int(len(data) * split_ratio)
    train_data = data.iloc[:split_index]
    test_data = data.iloc[split_index:]
    original_train = original_prices[:split_index]
    original_test = original_prices[split_index:]

    testArray = [[10, 10, 20], [10, 5, 20], [10, 10, 25], [10, 5, 25], [10, 10, 30], [10, 5, 30], [10, 10, 35], [10, 5, 35]]

    winningIndex = 0
    minErr = float('inf')
    for i in range(len(testArray)):
        try:
            Err, _ = train(model, train_data, test_data, original_test, original_train, testArray[i][0], testArray[i][1], testArray[i][2], False)
            if (Err < minErr):
                minErr = Err
                winningIndex = i
        except ValueError as ve:
            print(f"Skipped parameters {testArray[i]} due to insufficient data: {ve}")

    if minErr == float('inf'):
        print("No suitable parameters found.")
    else:
        print(f'winning params: {testArray[winningIndex]}')
        _, model = train(model, train_data, test_data, original_test, original_train, testArray[winningIndex][0], testArray[winningIndex][1], testArray[winningIndex][2], True)
        model.save(f'test{ticker}model.keras')

if __name__ == "__main__":
    main()
