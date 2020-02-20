import pandas as pd
import pandas_datareader as web
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, Activation
# from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau
import datetime

# Load Dataset
# data = pd.read_csv('C:/Users/505-06/Documents/005930.KS.csv')
data = web.DataReader('005930.KS', data_source='yahoo', start='2012-01-01', end='2020-2-13')
print("data.head(): \n",data.head())

# Compute Mid Price
# middle price를 예측할거다.
high_prices = data['High'].values
low_prices = data['Low'].values
mid_prices = (high_prices + low_prices) / 2

# Create Windows
# 최근 50일간의 데이터를 보고 내일을 예측한다.
#총 51개씩 저장.
seq_len = 50 # 50이란게 window size. 임의로 바꿀수 있다.
sequence_length = seq_len + 1

#result 라는 list에 51개씩 저장
result = []
for index in range(len(mid_prices) - sequence_length):
    result.append(mid_prices[index: index + sequence_length])
    
    
# Normalize Data. 이걸 해야지 모델이 더 잘 예측을 할 수 있다.
normalized_data = []
for window in result:
    normalized_window = [((float(p) / float(window[0])) - 1) for p in window]
    normalized_data.append(normalized_window)

result = np.array(normalized_data)

# split train and test data
row = int(round(result.shape[0] * 0.9))
train = result[:row, :]
np.random.shuffle(train)

x_train = train[:, :-1]
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
y_train = train[:, -1]

x_test = result[row:, :-1]
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
y_test = result[row:, -1]

print("x_train.shape, x_test.shape:\n",x_train.shape, x_test.shape)


# Build a Model
model = Sequential()

model.add(LSTM(50, return_sequences=True, input_shape=(50, 1))) # 임의로 조정 가

model.add(LSTM(64, return_sequences=False)) # 임의로 조정 가

model.add(Dense(1, activation='linear')) # 다음날 하루를 예측한다.

model.compile(loss='mse', optimizer='rmsprop')

model.summary()

# Training
model.fit(x_train, y_train,
    validation_data=(x_test, y_test),
    batch_size=10,
    epochs=20)

## #Train the model 이렇게 써도 
## model.fit(x_train, y_train, batch_size=10, epochs=20)

# Prediction
pred = model.predict(x_test)

fig = plt.figure(facecolor='white', figsize=(20, 10))
ax = fig.add_subplot(111)
ax.plot(y_test, label='True')
ax.plot(pred, label='Prediction')
ax.legend()
plt.show()