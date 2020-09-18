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

starttime = time.time()
df = pd.read_csv("../data.csv", header=0)
df = df.set_index('date')
df = df['2017/5/1':'2018/5/1']
print(len(df))

np_data = np.array(df)

# 标准化和归一化
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


class GRUModel(tf.keras.Model):
    def __init__(self, batch_size, seq_length, cell_size):
        super().__init__()
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.cell_size = cell_size

        self.layer1 = tf.keras.layers.Reshape((self.seq_length, 6), batch_size=self.batch_size)
        self.layer_GRU = tf.keras.layers.GRU(self.cell_size, return_sequences=True)
        self.layer_last_GRU = tf.keras.layers.GRU(self.cell_size)
        self.layer_dense = tf.keras.layers.Dense(6)

    def call(self, inputs):
        x = self.layer1(inputs)
        x = self.layer_GRU(x)
        x = self.layer_last_GRU(x)
        output = self.layer_dense(x)
        return output


lr = 1e-2
optimizer = tf.keras.optimizers.SGD(lr)
model = GRUModel(128, 5, 6)
#  tf.optimizers.RMSprop(0.001)
model.compile(optimizer=optimizer,
              loss='mse',
              metrics=['mae'])

history = model.fit(x_train, y_train, epochs=1000, validation_data=(x_valid, y_valid), batch_size=1, verbose=2,
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
plt.ylabel('RNN loss')
plt.xlabel('Step')
plt.legend()

plt.figure()
x0 = mae.index(min(mae)) * 80
y0 = min(mae)
# 标记折线最低点，xy是标记点位置，txtest是文本标注位置，文本偏移x-500，y+0.1以避免遮盖折现
plt.annotate('最低点: %s' % round(y0, 3), xy=(x0, y0), xytext=(x0 - 500, y0 + 0.1),
             arrowprops=dict(facecolor='black', shrink=0.00001))
plt.plot(x, mae, color='C1', marker='s', label='测试')
plt.ylabel('RNN valid mae')
plt.xlabel('Step')
plt.legend()
plt.show()
# plt.savefig('valid.svg')x

# yhat = model.predict(x_test)
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

r2_test = r2_score(y_test, y_hat)  # 这里调用内置函数计算

# 保留三位小数
print('r2_score : ', round(r2_test, 3))
print('use time: ', round(time.time() - starttime, 3), 's')

"""
r2_score :  0.717
use time:  311.277 s
"""

figure = plt.figure(figsize=(15, 6))
x = [i for i in range(len(y_hat[:100]))]

ax1 = figure.add_subplot(2, 3, 1)
ax2 = figure.add_subplot(2, 3, 2)
ax3 = figure.add_subplot(2, 3, 3)
ax4 = figure.add_subplot(2, 3, 4)
ax5 = figure.add_subplot(2, 3, 5)
ax0 = figure.add_subplot(2, 3, 6)

ax0.plot(x, y_hat[:100, 0])
ax0.plot(x, y_test[:100, 0])

ax1.plot(x, y_hat[:100, 1])
ax1.plot(x, y_test[:100, 1])

ax2.plot(x, y_hat[:100, 2])
ax2.plot(x, y_test[:100, 2])

ax3.plot(x, y_hat[:100, 3])
ax3.plot(x, y_test[:100, 3])

ax4.plot(x, y_hat[:100, 4])
ax4.plot(x, y_test[:100, 4])

ax5.plot(x, y_hat[:100, 5])
ax5.plot(x, y_test[:100, 5])

plt.show()
