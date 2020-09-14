#!/usr/bin/env python 
# encoding: utf-8
import time

import matplotlib
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.python.keras.layers import LSTM, Dense
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
model.add(LSTM(128, input_shape=(x_train.shape[1], x_train.shape[2])))  # x_train.shape[1], x_train.shape[2]也就是特征维度的5，6
model.add(Dense(6))  # dense是全连接层，只需要指定输出层维度。这里是最后一层，所以直接输出标签维度6
model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])  # 使用mse调整loss，使用mae验证准确率

# 拟合模型，epochs为训练次数， batch_size为单次训练训练数据大小， verbose=2代表控制台输出全部训练记录
history = model.fit(x_train, y_train, validation_data=(x_valid, y_valid), epochs=30, batch_size=128, verbose=2)

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
plt.ylabel('LSTM loss')
plt.xlabel('Step')
plt.legend()

plt.figure()
x0 = mae.index(min(mae)) * 80
y0 = min(mae)
# 标记折线最低点，xy是标记点位置，txtest是文本标注位置，文本偏移x-300，y+0.1以避免遮盖折现
plt.annotate('最低点: %s' % round(y0, 3), xy=(x0, y0), xytext=(x0 - 300, y0 + 0.1),
             arrowprops=dict(facecolor='black', shrink=0.00001))
plt.plot(x, mae, color='C1', marker='s', label='测试')
plt.ylabel('LSTM valid mae')
plt.xlabel('Step')
plt.legend()
plt.show()

"""
使用r2_score作为回归模型评价标准
R2就是决定系数（拟合优度），反映了因变量y的波动，有多少百分比能被自变量x的波动所描述
相当于回归平方和SSR除以总离差平方和SST
模型越好：r2→1
模型越差：r2→0
比如：
     y_true = [1, 2, 4]
     y_pred = [1.3, 2.5, 3.7]
     r2_score(y_true, y_pred)
Out: 0.9078571428571429

手写计算r2_score
def r_square(y_true, y_pred):
    SSR = K.mean(K.square(y_pred - K.mean(y_true)), axis=-1)
    SST = K.mean(K.square(y_true - K.mean(y_true)), axis=-1)
    return SSR / SST
"""
yhat_test = model.predict(x_test)
r2_test = r2_score(y_test, yhat_test)  # 这里调用内置函数计算

# 保留三位小数
print('r2_score : ', round(r2_test, 3))
print('use time: ', round(time.time() - starttime, 3), 's')

"""
epochs=100 ：
    r2_score :  0.731
    use time:  331.979 s
    
epochs=30 ：
    r2_score :  0.728
    use time:  88.959 s
"""
