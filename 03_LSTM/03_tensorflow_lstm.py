#!/usr/bin/env python
# encoding: utf-8
"""
 @Author : hanxiaopeng
 @Time : 2020/9/3
"""
import time

import matplotlib
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense, LSTM

starttime = time.time()
df = pd.read_csv("../data.csv", header=0)
df = df.set_index('date')
df = df['2017/6/1':'2018/6/1']
df = df.replace(0, np.nan)
df = df.fillna(df.mean(axis=0))
print(len(df))

np_data = np.array(df)

mm = MinMaxScaler()
np_data = mm.fit_transform(np_data)
ss = StandardScaler()
np_data = ss.fit_transform(np_data)

x_list = []
y_list = []
for i in range(len(np_data)):
    if i + 6 == len(np_data):
        break
    x = np_data[i: i + 5].tolist()
    y = np_data[i + 6].tolist()
    x_list.append(x)
    y_list.append(y)

x_array = np.array(x_list)
y_array = np.array(y_list)

# 预测时间长度
length = 30

len_train = int(len(x_array) * 0.6)

x_test = x_array[len_train:len_train + length]
y_test = y_array[len_train:len_train + length]

# 划分训练集，验证集，测试集
x_train = x_array[:len_train]
y_train = y_array[:len_train]

x_valid = x_array[len_train + length:]
y_valid = y_array[len_train + length:]

model = Sequential()
model.add(LSTM(30, input_shape=(x_train.shape[1], x_train.shape[2])))  # x_train.shape[1], x_train.shape[2]也就是特征维度的5，6
# model.add(Dropout(0.2))
# model.add(LSTM(128, return_sequences=True))
# model.add(Dropout(0.2))
# model.add(LSTM(128, return_sequences=False))
model.add(Dense(6))  # dense是全连接层，只需要指定输出层维度。这里是最后一层，所以直接输出标签维度6
lr = 1e-2
optimizer = tf.keras.optimizers.SGD(lr)
# 'RMSprop'
# 拟合模型，epochs为训练次数， batch_size为单次训练训练数据大小， verbose=2代表控制台输出全部训练记录
model.compile(optimizer=optimizer,
              loss='mse',
              metrics=['mae'])

history = model.fit(x_train, y_train, epochs=500, validation_data=(x_valid, y_valid), batch_size=1, verbose=2,
                    use_multiprocessing=True)
data = history.history
losses = data['loss']
mae = data['val_mae']

# 绘图参数设定
matplotlib.rcParams['font.size'] = 20
matplotlib.rcParams['figure.titlesize'] = 20
matplotlib.rcParams['figure.figsize'] = [9, 7]
matplotlib.rcParams['font.family'] = ['STKaiTi']
matplotlib.rcParams['axes.unicode_minus'] = False

plt.figure()
x = [i * 80 for i in range(len(losses))]
plt.plot(x, losses, color='C0', marker='s', label='训练')
plt.ylabel('LSTM loss')
plt.xlabel('Step')
plt.legend()

plt.figure()
x0 = mae.index(min(mae)) * 80
y0 = min(mae)
# 标记折线最低点，xy是标记点位置，txtest是文本标注位置，文本偏移x-500，y+0.1以避免遮盖折现
plt.annotate('最低点: %s' % round(y0, 3), xy=(x0, y0), xytext=(x0 - 500, y0 + 0.1),
             arrowprops=dict(facecolor='black', shrink=0.00001))
plt.plot(x, mae, color='C1', marker='s', label='测试')
plt.ylabel('LSTM valid mae')
plt.xlabel('Step')
plt.legend()

# 滚动预测
x_history = x_train.reshape(-1, 6)
y_hat_lis = []
for i in range(length):
    x = x_history[-5:, ].reshape(1, 5, 6)
    y_hat = model.predict(x)
    x_history = np.concatenate((x_history, y_hat), axis=0)
    y_hat_lis.append(y_hat)

y_hat = np.array(y_hat_lis).reshape(-1, 6)

# 反标准化和反归一化
y_hat = ss.inverse_transform(y_hat)
y_hat = mm.inverse_transform(y_hat)

y_test = ss.inverse_transform(y_test)
y_test = mm.inverse_transform(y_test)

figure = plt.figure(figsize=(15, 6))
x = [i for i in range(len(y_hat[:length]))]

ax1 = figure.add_subplot(2, 3, 1)
ax2 = figure.add_subplot(2, 3, 2)
ax3 = figure.add_subplot(2, 3, 3)
ax4 = figure.add_subplot(2, 3, 4)
ax5 = figure.add_subplot(2, 3, 5)
ax0 = figure.add_subplot(2, 3, 6)

y_test = y_test.reshape(-1, 6)

ax0.plot(x, y_hat[:length, 0])
ax0.plot(x, y_test[:length, 0])

ax1.plot(x, y_hat[:length, 1])
ax1.plot(x, y_test[:length, 1])

ax2.plot(x, y_hat[:length, 2])
ax2.plot(x, y_test[:length, 2])

ax3.plot(x, y_hat[:length, 3])
ax3.plot(x, y_test[:length, 3])

ax4.plot(x, y_hat[:length, 4])
ax4.plot(x, y_test[:length, 4])

ax5.plot(x, y_hat[:length, 5])
ax5.plot(x, y_test[:length, 5])

plt.show()

r2_test = r2_score(y_test[:length, ], y_hat)  # 这里调用内置函数计算

# 保留三位小数
print('r2_score : ', round(r2_test, 3))
print('use time: ', round(time.time() - starttime, 3), 's')

"""
r2_score :  -4.544
use time:  593.584 s
"""
