#!/usr/bin/env python
# encoding: utf-8
"""
 @Author : hanxiaopeng
 @Time : 2020/9/22
"""

import os
import time

import matplotlib
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler

matplotlib.rcParams['font.size'] = 20
matplotlib.rcParams['figure.titlesize'] = 20
matplotlib.rcParams['figure.figsize'] = [9, 7]
matplotlib.rcParams['font.family'] = ['STKaiTi']
matplotlib.rcParams['axes.unicode_minus'] = False

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
print(tf.__version__)

# length是预测长度，look_back是滚动周期
length = 30
look_back = 30

starttime = time.time()

# 读取数据
df = pd.read_csv("../data.csv", header=0)
df = df.set_index('date')

print(len(df))
np_data = np.array(df)

# 标准化和归一化
mm = MinMaxScaler()
np_data = mm.fit_transform(np_data)
ss = StandardScaler()
np_data = ss.fit_transform(np_data)

lis = []
x_list = []
y_list = []
for i in range(len(np_data)):
    if i + look_back + 1 == len(np_data):
        break
    x = np_data[i: i + look_back].tolist()
    y = np_data[i + look_back + 1].tolist()
    x_list.append(x)
    y_list.append(y)
    lis.append([x, y])
x_array = np.array(x_list)
y_array = np.array(y_list)

# 划分训练集，验证集，测试集
len_train = int(len(x_array) * 0.8)
x_train = x_array[:len_train]
y_train = y_array[:len_train]
x_test = x_array[len_train:len_train + length]
y_test = y_array[len_train:len_train + length]
x_valid = x_array[len_train + length:]
y_valid = y_array[len_train + length:]

# 数据预处理
def preprocess(x, y):  # 自定义的预处理函数
    x = tf.cast(x, dtype=tf.float32)
    x = tf.reshape(x, [-1, look_back * 6])
    y = tf.cast(y, dtype=tf.float32)
    y = tf.reshape(y, [-1, 6])
    return x, y


batchsz = 233
train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_db = train_db.batch(batchsz)  # 批训练，批规模
train_db = train_db.map(preprocess)

train_db = train_db.repeat(120)  # 训练次数
x, y = next(iter(train_db))
print('train sample:', x.shape, y.shape)

valid_db = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
valid_db = valid_db.batch(batchsz).map(preprocess)

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
model = GRUModel(128, 30, 6)
#  tf.optimizers.RMSprop(0.001)
model.compile(optimizer=optimizer,
              loss='mse',
              metrics=['mae'])

history = model.fit(train_db, epochs=100, validation_data=(valid_db), batch_size=10, verbose=2,
                    use_multiprocessing=True)
data = history.history

losses = data['loss']
accs = data['val_mae']

plt.figure()
x_ = [i * 80 for i in range(len(losses))]
min_indx = np.argmin(losses)
plt.plot(x_, losses, color='C0', marker='s', label='训练')
plt.ylabel('线性回归 Loss')
plt.xlabel('Step')
plt.plot(min_indx, losses[min_indx], 'gs')
plt.legend()

plt.figure()
x0 = accs.index(min(accs)) * 80
y0 = min(accs)
# 标记折线最低点。xy是标记点位置，xytest是文本标注位置，文本偏移x-500，y+0.1以避免遮盖折现
plt.annotate('最低点: %s' % round(y0, 3), xy=(x0, y0), xytext=(x0 - 500, y0 + 0.1),
             arrowprops=dict(facecolor='black', shrink=0.00001))
plt.plot(x_, accs, color='C1', marker='s', label='测试')
plt.ylabel('线性回归 valid mae')
plt.xlabel('Step')
plt.legend()
plt.show()

# 直接预测
# x_test = x_test.reshape(-1, 30 * 6)
# h1 = x_test @ w1 + b1
# h1 = tf.nn.relu(h1)
# h2 = h1 @ w2 + b2
# h2 = tf.nn.relu(h2)
# out = h2 @ w3 + b3
# y_hat_lis = out

# 滚动预测
x_history = x_train.reshape(-1, 6)
y_hat_lis = []
for i in range(length):
    x = x_history[-length:, ].reshape(-1, 30 * 6)
    y_hat = model.predict(x)
    x_history = np.concatenate((x_history, y_hat), axis=0)
    y_hat_lis.append(y_hat)

y_hat = np.array(y_hat_lis).reshape(-1, 6)

# 反标准化和反归一化
y_hat = ss.inverse_transform(y_hat)
y_hat = mm.inverse_transform(y_hat)

y_test = ss.inverse_transform(y_test)
y_test = mm.inverse_transform(y_test)

# 绘制结果
figure = plt.figure(figsize=(18, 8))  # 定制图框长宽
columns_list = df.columns  # 标记特征名称
for i in range(y_hat.shape[1]):
    plt.subplot(2, 3, i + 1)
    plt.plot(y_hat[:, i][:30], label='pred')
    plt.plot(y_test[:, i][:30], label='true')
    plt.ylabel(columns_list[i])
    plt.legend()
plt.show()


def MAPE(true, pred):
    diff = np.abs(np.array(true) - np.array(pred))
    return np.mean(diff / true)


# 评价标准
r2_test = r2_score(y_test[:length, ], y_hat)
mape = MAPE(y_test[:length, ], y_hat)
mse = mean_squared_error(y_test[:length, ], y_hat)

# 保留三位小数
print('mape', round(mape, 3))
print('mse', round(mse, 3))
print('r2_score : ', round(r2_test, 3))
print('use time: ', round(time.time() - starttime, 3), 's')

"""
控制台结果：
mape 0.314
mse 910.378
r2_score :  -1.746
use time:  559.106 s
"""
