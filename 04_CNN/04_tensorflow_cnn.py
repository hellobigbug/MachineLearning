#!/usr/bin/env python 
# encoding: utf-8 
"""
 @Author : hanxiaopeng
 @Time : 2020/9/11 
"""
import time

import matplotlib
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.python.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout
from tensorflow.python.keras.models import Sequential

starttime = time.time()

# 本地CSV读取数据
df = pd.read_csv("../ATMP数据.csv", header=0)
df = df.set_index('数据日期')
np_data = np.array(df)
print('data size', len(df))

# 标准化和归一化
scaler = MinMaxScaler()
np_data = scaler.fit_transform(np_data)
StandardScaler().fit(np_data)
np_data = StandardScaler().fit_transform(np_data)

# 获取特征x和标签y
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

# 划分训练集，验证集，测试集，比例为8：2：2
x_train = x_array[:int(len(x_array) * 0.6)]
x_valid = x_array[int(len(x_array) * 0.6):int(len(x_array) * 0.8)]
x_test = x_array[int(len(x_array) * 0.8):]
y_train = y_array[:int(len(y_array) * 0.6)]
y_valid = y_array[int(len(y_array) * 0.6):int(len(y_array) * 0.8)]
y_test = y_array[int(len(y_array) * 0.8):]

# 优化器选择随机梯度下降（SGD）, lr是初始值，可调整
lr = 1e-2
optimizer = tf.keras.optimizers.SGD(lr)

# 搭建lstm模型
model = Sequential()
model.add(Conv1D(filters=128, kernel_size=2, activation='relu', input_shape=(5, 6)))  # 卷积层1
model.add(Conv1D(filters=64, kernel_size=2, activation='relu'))  # 卷积层2
model.add(MaxPooling1D(pool_size=2))  # 池化层
model.add(Dropout(0.2))
model.add(Flatten())  # 降维
model.add(Dense(50, activation='relu'))  # 全连接层
model.add(Dropout(0.2))
model.add(Dense(6))  # 全连接输出
model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])  # 使用mse调整loss，使用mae验证准确率

# 拟合模型，epochs为训练次数， batch_size为单次训练训练数据大小， verbose=2代表控制台输出全部训练记录
history = model.fit(x_train, y_train, validation_data=(x_valid, y_valid), epochs=100, batch_size=512, verbose=2)

# 标记fit时间
usetime = time.time() - starttime

# 通过训练结果获取训练集的loss和准确度变化过程
data = history.history
losses = data['loss']
mae = data['val_mae']

# 设定参数并绘图，不指定字体无法显示中文
matplotlib.rcParams['font.size'] = 20
matplotlib.rcParams['figure.titlesize'] = 20
matplotlib.rcParams['figure.figsize'] = [9, 7]
matplotlib.rcParams['font.family'] = ['STKaiTi']
matplotlib.rcParams['axes.unicode_minus'] = False

plt.figure()
x = [i * 80 for i in range(len(losses))]
plt.plot(x, losses, color='C0', marker='s', label='训练')
plt.ylabel('CNN loss')
plt.xlabel('Step')
plt.legend()

plt.figure()
x0 = mae.index(min(mae)) * 80
y0 = min(mae)
# 标记折线最低点，xy是标记点位置，txtest是文本标注位置，文本偏移x-400，y+0.1以避免遮盖折现
plt.annotate('最低点: %s' % round(y0, 3), xy=(x0, y0), xytext=(x0 - 500, y0 + 0.1),
             arrowprops=dict(facecolor='black', shrink=0.00001))
plt.plot(x, mae, color='C1', marker='s', label='测试')
plt.ylabel('CNN valid mae')
plt.xlabel('Step')
plt.legend()
# plt.show()

yhat_test = model.predict(x_test)
r2_test = r2_score(y_test, yhat_test)  # 这里调用内置函数计算

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

# 保留三位小数
print('r2_score : ', round(r2_test, 3))
print('use time: ', round(time.time() - starttime, 3), 's')

"""
训练200次
epochs=100 ：
r2_score :  0.668
use time:  109.192 s
训练50次
epochs=30 ：
    r2_score :  0.693
    use time:  10.88 s
"""
