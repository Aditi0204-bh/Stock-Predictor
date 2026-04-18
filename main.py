import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
data = yf.download("AAPL", start="2015-01-01", end="2024-01-01")
print(data.head())

plt.plot(data['Close'])
plt.title("Stock Price Over Time")
plt.xlabel("Date")
plt.ylabel("Price")
plt.show()

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))

X = []
y = []

for i in range(60, len(scaled_data)):
    X.append(scaled_data[i-60:i])
    y.append(scaled_data[i])

X = np.array(X)
y = np.array(y)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X.reshape(X.shape[0], -1), y)

predictions = model.predict(X.reshape(X.shape[0], -1))

plt.plot(y, label="Actual")
plt.plot(predictions, label="Predicted")
plt.legend()
plt.show()