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
from sklearn import preprocessing
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

starttime = time.time()
df = pd.read_csv("../ATMP数据.csv", header=0)
df = df.set_index('数据日期')

print(len(df))

np_data = np.array(df)
min_max_scaler = preprocessing.MinMaxScaler()
np_data = min_max_scaler.fit_transform(np_data)
np_data = StandardScaler().fit_transform(np_data)

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
# x_array = normalize(x_array, axis=0, norm='max')
y_array = np.array(y_list)
# y_array = normalize(y_array, axis=0, norm='max')

# 划分训练集，验证集，测试集，比例为8：2：2
x_train = x_array[:int(len(x_array) * 0.6)]
x_valid = x_array[int(len(x_array) * 0.6):int(len(x_array) * 0.8)]
x_test = x_array[int(len(x_array) * 0.8):]
y_train = y_array[:int(len(y_array) * 0.6)]
y_valid = y_array[int(len(y_array) * 0.6):int(len(y_array) * 0.8)]
y_test = y_array[int(len(y_array) * 0.8):]


# 数据预处理
def preprocess(x, y):  # 自定义的预处理函数
    x = tf.cast(x, dtype=tf.float32)
    x = tf.reshape(x, [-1, 5 * 6])  # 打平
    y = tf.cast(y, dtype=tf.float32)
    y = tf.reshape(y, [-1, 6])
    return x, y


batchsz = 128
train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train))
# train_db = train_db.shuffle(1000)  # 打乱顺序，缓冲池1000
train_db = train_db.batch(batchsz, drop_remainder=True)  # 批训练，批规模
train_db = train_db.map(preprocess)
train_db = train_db.repeat(20)

#
valid_db = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
# valid_db = valid_db.shuffle(1000)
valid_db = valid_db.batch(batchsz, drop_remainder=True).map(preprocess)
x, y = next(iter(train_db))
print('train sample:', x.shape, y.shape)


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

history = model.fit(train_db, epochs=100, validation_data=valid_db, batch_size=128, verbose=2)
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

yhat_test = model.predict(x_test)
r2_test = r2_score(y_test, yhat_test)  # 这里调用内置函数计算

# 保留三位小数
print('r2_score : ', round(r2_test, 3))
print('use time: ', round(time.time() - starttime, 3), 's')

"""
r2_score :  0.717
use time:  311.277 s
"""

figure = plt.figure(figsize=(15, 6))
x = [i for i in range(len(yhat_test[:100]))]

ax1 = figure.add_subplot(2, 3, 1)
ax2 = figure.add_subplot(2, 3, 2)
ax3 = figure.add_subplot(2, 3, 3)
ax4 = figure.add_subplot(2, 3, 4)
ax5 = figure.add_subplot(2, 3, 5)
ax0 = figure.add_subplot(2, 3, 6)

ax0.plot(x, yhat_test[:100, 0])
ax0.plot(x, y_test[:100, 0])

ax1.plot(x, yhat_test[:100, 1])
ax1.plot(x, y_test[:100, 1])

ax2.plot(x, yhat_test[:100, 2])
ax2.plot(x, y_test[:100, 2])

ax3.plot(x, yhat_test[:100, 3])
ax3.plot(x, y_test[:100, 3])

ax4.plot(x, yhat_test[:100, 4])
ax4.plot(x, y_test[:100, 4])

ax5.plot(x, yhat_test[:100, 5])
ax5.plot(x, y_test[:100, 5])

plt.show()