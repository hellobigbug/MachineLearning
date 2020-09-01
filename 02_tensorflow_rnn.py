#!/usr/bin/env python 
# encoding: utf-8 
"""
 @Author : hanxiaopeng
 @Time : 2020/8/31 
"""

import pandas as pd
from tensorflow.keras.layers import SimpleRNN
from tensorflow.keras.optimizers import RMSprop
from matplotlib import pyplot
import numpy as np
from sklearn.metrics import mean_squared_error
from tensorflow.python.keras.layers import Activation, Dense, Dropout, Embedding
from tensorflow.keras.models import Sequential


df = pd.read_csv('ATMP数据.csv')
df = df.set_index('数据日期')

np_data = np.array(df)

lis = []
x_list = []
y_list = []
for i in range(len(np_data)):
    if i + 6 == len(np_data):
        break
    x = np_data[i: i + 5].tolist()
    y = np_data[i + 6].tolist()
    x_list.append(x)
    y_list.append(y)
    lis.append([x,y])

# 要保存的矩阵样式
# print(lis)

x_array = np.array(x_list)
y_array = np.array(y_list)

X_train = x_array[:int(len(x_array)*0.75)]
Y_train = y_array[:int(len(y_array)*0.75)]
X_test = x_array[int(len(x_array)*0.75):]
Y_test = y_array[int(len(y_array)*0.75):]

# # normalize the dataset
# scaler = MinMaxScaler(feature_range=(0, 1))
# dataset = scaler.fit_transform(np.reshape(df['number'].values,(df.shape[0],1)))

# according to step, create dataset
# def create_dataset(dataset, step):
#     dataX, dataY = [], []
#     for i in range(len(dataset) - step - 1):
# 	    a = dataset[i:(i + step), 0]
# 	    dataX.append(a)
# 	    dataY.append(dataset[i + step, 0])
#     return np.array(dataX), np.array(dataY)
#
# step = 36
# dataX, dataY = create_dataset(dataset, step)

# split train and test dataset
# 为什么不用train_test_split。时间序列数据中应该用前期的数据训练，后期数据预测。
# X_train, X_test, Y_train, Y_test= train_test_split(dataX, dataY, test_size=0.2, random_state=0)

# test_size = int(dataX.shape[0]*0.3)
# train_size = dataX.shape[0]-test_size
# X_train = dataX[0:train_size,]
# X_test = dataX[train_size:dataX.shape[0],]
# Y_train = dataY[0:train_size,]
# Y_test = dataY[train_size:dataX.shape[0],]

# X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
# X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
# Y_train=np.reshape(Y_train, (Y_train.shape[0], 1))
# Y_test= np.reshape(Y_test, (Y_test.shape[0], 1))


# 3、model train
# model = Sequential()
# model.add(SimpleRNN(30))
# model.add(Dropout(0.5))  # dropout层防止过拟合
# model.add(Dense(6))      # 全连接层
# model.add(Activation('sigmoid'))  #激活层
# model.compile(optimizer=RMSprop(), loss='mse')
# model.fit(X_train, Y_train, batch_size=512, verbose=10)
# model.save('passager.h5')
# model = load_model('passager.h5')
model = Sequential()
model.add(Embedding(10000,32,input_length=30))
model.add(SimpleRNN(30))
model.add(Dense(6,activation="sigmoid"))
model.summary()
model.compile(optimizer=RMSprop(), loss='mse')
# model.fit(X_train, Y_train, batch_size=512, verbose=10)

history = model.fit(X_train, Y_train, batch_size=512, verbose=10, validation_data=(X_test, Y_test))
# 4、model predict
Y_predict = model.predict(X_test)

# Y_predict = scaler.inverse_transform(Y_predict)
np.reshape(Y_test, (Y_test.shape[0], 1))
# Y_test = scaler.inverse_transform(Y_test)

Y_predict = np.reshape(Y_predict,(Y_predict.shape[0],))
Y_test = np.reshape(Y_test,(Y_test.shape[0],))

# 5、model evaluation
print("model mean squared error is " + str(mean_squared_error(Y_test, Y_predict)))


# plot data
pyplot.plot(Y_predict)
pyplot.plot(Y_test)
pyplot.show()