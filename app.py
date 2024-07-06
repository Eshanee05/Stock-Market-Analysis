import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
from sklearn.preprocessing import MinMaxScaler

start = '2010-01-01'
end = '2019-12-31'

st.title('Stock Trend Prediction')

user_input = st.text_input('Enter stock ticker', 'AAPL')

# Download the stock data
df = yf.download(user_input, start=start, end=end)

# Describing data
st.subheader('Data from 2010 to 2019')
st.write(df.describe())

# Visualization
st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize=(12,6))
plt.plot(df['Close'])
plt.title('Closing Price vs Time')
st.pyplot(fig)

# Moving Average
st.subheader('Closing Price vs Time Chart with 100MA')
ma100 = df['Close'].rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(df['Close'], label='Closing Price')
plt.plot(ma100, label='100-Day MA')
plt.title('Closing Price vs Time with 100MA')
plt.legend()
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100MA and 200MA')
ma200 = df['Close'].rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(df['Close'], label='Closing Price')
plt.plot(ma100, label='100-Day MA')
plt.plot(ma200, label='200-Day MA')
plt.title('Closing Price vs Time with 100MA and 200MA')
plt.legend()
st.pyplot(fig)

# Splitting the data into training and testing
data_training = pd.DataFrame(df['Close'][0:int(len(df) * 0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.70):])

# Scaling the data
scaler = MinMaxScaler(feature_range=(0, 1))
data_training_array = scaler.fit_transform(data_training)

# Loading the model
model = load_model('keras_model.h5')

# Preparing test data
past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

# Making predictions
y_predicted = model.predict(x_test)

scaler = scaler.scale_
scale_factor = 1 / scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

# Plotting the predictions
st.subheader('Prediction vs Original')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label='Original Price')
plt.plot(y_predicted, 'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('Prediction vs Original')
plt.legend()
st.pyplot(fig2)
