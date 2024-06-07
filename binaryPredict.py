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

def createSet(data, original_data, lookBack, stepsAhead):
    x_scaled, y_scaled, x_unscaled, distFromMean = [], [], [], []
    meanPrice = np.mean(original_data)
    
    for index in range(len(data) - lookBack - stepsAhead + 1):
        x_scaled.append(data.iloc[index:index + lookBack]['price_change'].values.flatten().tolist())
        x_unscaled.append(original_data[index:index + lookBack].flatten().tolist())
        distFromMean.append(x_unscaled[index][-1] - meanPrice)        
        future_price = original_data[index + lookBack + stepsAhead - 1]
        current_price = original_data[index + lookBack - 1]
        y_scaled.append(1 if future_price > current_price else 0)
        
    tmpXScaled, tmpYScaled, tmpXUnscaled, distFromMean = np.array(x_scaled), np.array(y_scaled), np.array(x_unscaled), np.array(distFromMean)

    split_ratio = 0.8
    split_index = int(len(data) * split_ratio)

    indices = np.arange(tmpXScaled.shape[0])
    np.random.shuffle(indices)
    shuffledXscaled = tmpXScaled[indices]
    shuffledYScaled = tmpYScaled[indices]
    shuffledXUnscaled = tmpXUnscaled[indices]
    shuffledDistFromMean= distFromMean[indices]
    
    scaledxTrain = shuffledXscaled[:split_index]
    scaledxTest = shuffledXscaled[split_index:]
    unscaledxTrain = shuffledXUnscaled[:split_index]
    unscaledxTest = shuffledXUnscaled[split_index:]
    distFromMeanTrain = shuffledDistFromMean[:split_index]
    distFromMeanTest = shuffledDistFromMean[split_index:]
    
    yTrain = shuffledYScaled[:split_index]
    yTest= shuffledYScaled[split_index:]

    # Reshape for LSTM
    X_train_scaled = np.reshape(scaledxTrain, (scaledxTrain.shape[0], scaledxTrain.shape[1], 1))
    X_test_scaled = np.reshape(scaledxTest, (scaledxTest.shape[0], scaledxTest.shape[1], 1))
    return X_train_scaled, X_test_scaled, unscaledxTest, yTrain, yTest, distFromMeanTrain, distFromMeanTest

def create_model(ticker, lookBack):
    
    try:
        modelString = ticker+'Model.keras'
        model = load_model(modelString)
    except:
        time_series_input = Input(shape=(lookBack, 1), name='time_series_input')
        x = LSTM(50, return_sequences=True)(time_series_input)
        x = Dropout(0.2)(x)
        x = LSTM(50)(x)
        x = Dropout(0.2)(x)

        # Input for distFromMean
        dist_input = Input(shape=(1,), name='dist_input')
        y = Dense(10, activation='relu')(dist_input)
        
        # Combine both inputs
        combined = Concatenate()([x, y])
        combined = Dense(10, activation='relu')(combined)
        output = Dense(1, activation='sigmoid')(combined)

        model = Model(inputs=[time_series_input, dist_input], outputs=output)
        model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    return model


def predict(model, numEpochs, X_train_scaled, distFromMeanTrain, distFromMeanTest, Y_train, X_test_scaled, Y_test, unscaledxTest, flag):
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

    class_weights = class_weight.compute_class_weight(class_weight='balanced',
                                                      classes=np.unique(Y_train),
                                                      y=Y_train)
    class_weights_dict = dict(enumerate(class_weights))
    
    model.fit([X_train_scaled, distFromMeanTrain], Y_train, epochs=numEpochs, batch_size=32, validation_split=0.2, callbacks=[early_stopping, reduce_lr], verbose=1, class_weight=class_weights_dict)
    print(distFromMeanTest)
    Y_pred = model.predict([X_test_scaled, distFromMeanTest])
    Y_pred_class = (Y_pred > 0.5).astype(int)

    Y_test_original = np.where(Y_test > 0.5, 1, 0)

    correct_direction_predictions = (Y_pred_class.flatten() == Y_test_original).sum()

    if flag:
        for i in range(len(Y_test)):
            print(f'{unscaledxTest[i]} => {Y_test[i]} Predicted: {Y_pred_class[i]}')
            
        print(f'Correct direction predictions: {correct_direction_predictions} / {len(Y_test)}')

    accuracy = correct_direction_predictions / len(Y_test)
    print(f'Epochs: {numEpochs}, Accuracy: {accuracy}')
    return numEpochs, accuracy, model

def train(model, train_data, test_data, original_test, original_train, lookBack, lookAhead, epochs, flag):
    X_train_scaled, Y_train, X_train_unscaled = createSet(train_data, original_train, lookBack, lookAhead)
    X_test_scaled, Y_test, X_test_unscaled = createSet(test_data, original_test, lookBack, lookAhead)

    X_train_scaled = np.reshape(X_train_scaled, (X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
    X_test_scaled = np.reshape(X_test_scaled, (X_test_scaled.shape[0], X_test_scaled.shape[1], 1))

    numEpochs, accuracy = predict(model, epochs, X_train_scaled, X_test_unscaled, Y_train, X_test_scaled, Y_test, flag, original_test)
    return accuracy, model

def main():
    # pd.set_option('display.max_rows', None)
    # pd.set_option('display.max_columns', None)
    fileName = input("Name of file to train on: ")
    ticker = fileName[:4]
    file_path = 'data/' + fileName
    data = pd.read_csv(file_path, header=None, names=['price'])
    scaler = MinMaxScaler(feature_range=(0, 1))

    data['price_change'] = data['price'].diff().fillna(0)
    original_prices = data['price'].values.reshape(-1, 1)
    data['price_change'] = scaler.fit_transform(data[['price_change']])
    
    testArray = [[10, 10, 10], [10, 10, 15], [10, 10, 20], [10, 10, 25], [10, 10, 30], [10, 10, 35], [10, 10, 40]]

    winningIndex = 0
    maxAccuracy = 0
    winningModel = None

    for i in range(len(testArray)):
        
        model = create_model(ticker, testArray[i][0])
        X_train_scaled, X_test_scaled, unscaledxTest, yTrain, yTest, distFromMeanTrain, distFromMeanTest = testCreateSet(data, original_prices, testArray[i][0], testArray[i][1])
        numEpochs, accuracy, model = predict(model, testArray[i][2], X_train_scaled, distFromMeanTrain, distFromMeanTest, yTrain, X_test_scaled, yTest, unscaledxTest, False)
        
        if i == 0 or accuracy > maxAccuracy:
            winningModel = model
            maxAccuracy = accuracy
            winningIndex = i
    
    print(f'Winning params: {testArray[winningIndex]}')
    X_train_scaled, X_test_scaled, unscaledxTest, yTrain, yTest, distFromMeanTrain, distFromMeanTest = testCreateSet(data, original_prices, testArray[winningIndex][0], testArray[winningIndex][1])
    predict(winningModel, testArray[winningIndex][2], X_train_scaled, distFromMeanTrain, distFromMeanTest, yTrain, X_test_scaled, yTest, unscaledxTest, True)
    
    modelString = ticker+"Model.keras"
    winningModel.save(modelString)

main()


