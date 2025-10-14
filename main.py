from tensorflow import keras
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import os
from datetime import datetime

data = pd.read_csv('MicrosoftStock.csv')
print(data.head())
print(data.info())
print(data.describe())

# initial Data Visualization
# Plot 1 - open and close price over time
plt.figure(figsize=(12,6))
plt.plot(data['date'], data['open'], label='Open Price', color='blue')
plt.plot(data['date'], data['close'], label='Close Price', color='orange')
plt.title('Microsoft Stock Prices Over Time')
plt.legend()
# plt.show()

# Plot 2 - Trading volume (check for outliers)
plt.figure(figsize=(12,6))
plt.plot(data['date'], data['volume'], label='Trading Volume', color='green')
plt.title('Microsoft Trading Volume Over Time')
# plt.show()


# Dropping non-numeric columns
numeric_data = data.select_dtypes(include=["int64", "float64"])

# Checking for correlations between features
plt.figure(figsize=(8,6))
sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
# plt.show()


# Convert date to datetime format and create date filter
data['date'] = pd.to_datetime(data['date'])


prediction = data.loc[
    (data['date'] > datetime(2013, 1, 1)) & (data['date'] < datetime(2018, 1, 1))
    ]

plt.figure(figsize=(12,6))
plt.plot(data['date'], data['close'], color='blue')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('Prices Over Time')


# LSTM Model(Sequential)

stock_close = data.filter(["close"])
dataset = stock_close.values # converting to numpy array
training_data_len = int(np.ceil( len(dataset) * .95 )) # 95% of data for training

# prepocessing data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(dataset)

training_data = scaled_data[0:training_data_len] #95% of all the data

x_train, y_train = [], []

# create a sliding window for the stock (60 days)
for i in range(60, len(training_data)):
    x_train.append(training_data[i-60:i, 0])
    y_train.append(training_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))


# Build the LSTM model
model = keras.models.Sequential()


# first LSTM layer
model.add(keras.layers.LSTM(64, return_sequences=True, input_shape=(x_train.shape[1], 1)))

# second LSTM layer
model.add(keras.layers.LSTM(64, return_sequences=False))

# Dense layer
model.add(keras.layers.Dense(128, activation='relu'))

# dropout layer
model.add(keras.layers.Dropout(0.3))

# output layer
model.add(keras.layers.Dense(1))

model.summary()
model.compile(optimizer='adam', loss='mae', 
              metrics=[keras.metrics.RootMeanSquaredError()])

# Train the model
training = model.fit(x_train, y_train, batch_size=32, epochs=20)

# Create the testing dataset
test_data = scaled_data[training_data_len - 60:]
x_test, y_test = [], dataset[training_data_len:]

for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Get the model's predicted price values
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# plot the data
train = data[:training_data_len]
test = data[training_data_len:]
test = test.copy()

test['Predictions'] = predictions

plt.figure(figsize=(12,8))
plt.plot(train['date'], train['close'], label='Train Close Price(actual data)', color='blue')
plt.plot(test['date'], test['close'], label='Test Close Price(actual data)', color='orange')
plt.plot(test['date'], test['Predictions'], label='Predicted Close Price', color = 'red')
plt.title('Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.show()        