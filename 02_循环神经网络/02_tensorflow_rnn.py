#!/usr/bin/env python
# encoding: utf-8
"""
 @Author : hanxiaopeng
 @Time : 2020/9/3
"""

import matplotlib
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import normalize, StandardScaler
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense, Flatten
from tensorflow.python.keras.optimizer_v2.rmsprop import RMSprop

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

x_train = x_array[:int(len(x_array) * 0.75)]
y_train = y_array[:int(len(y_array) * 0.75)]
x_valid = x_array[int(len(x_array) * 0.75):]
y_valid = y_array[int(len(y_array) * 0.75):]


# tenroflow提供的数据集
# (x, y), (x_valid, y_valid) = datasets.mnist.load_data()

# print('x:', x.shape, 'y:', y.shape, 'x valid:', x_valid.shape, 'y valid:', y_valid)


# 数据预处理
def preprocess(x, y):  # 自定义的预处理函数
    x = tf.cast(x, dtype=tf.float32)
    x = tf.reshape(x, [-1, 5 * 6])  # 打平
    y = tf.cast(y, dtype=tf.float32)
    y = tf.reshape(y, [-1, 6])
    return x, y


batchsz = 512
train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_db = train_db.shuffle(1000)  # 打乱顺序，缓冲池1000
train_db = train_db.batch(batchsz, drop_remainder=True)  # 批训练，批规模
train_db = train_db.map(preprocess)
train_db = train_db.repeat(20)

#
valid_db = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
valid_db = valid_db.shuffle(1000).batch(batchsz, drop_remainder=True).map(preprocess)
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

lr = 1e-1
optimizer = tf.keras.optimizers.SGD(lr)
model = GRUModel(512, 5, 6)
#  tf.optimizers.RMSprop(0.001)
model.compile(optimizer=optimizer,
              loss='mse',
              metrics=['MAE'])

history = model.fit(train_db, epochs=30, validation_data=valid_db)
data = history.history

losses = data['loss']
MAPE = data['MAE']


# 绘图参数设定
matplotlib.rcParams['font.size'] = 20
matplotlib.rcParams['figure.titlesize'] = 20
matplotlib.rcParams['figure.figsize'] = [9, 7]
matplotlib.rcParams['font.family'] = ['STKaiTi']
matplotlib.rcParams['axes.unicode_minus'] = False

plt.figure()
x = [i * 80 for i in range(len(losses))]
plt.plot(x, losses, color='C0', marker='s', label='训练')
plt.ylabel('loss')
plt.xlabel('Step')
plt.legend()
# plt.savefig('train.svg')

plt.figure()
plt.plot(x, MAPE, color='C1', marker='s', label='测试')
plt.ylabel('误差率')
plt.xlabel('Step')
plt.legend()
plt.show()
# plt.savefig('valid.svg')
