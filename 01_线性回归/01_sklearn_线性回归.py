#!/usr/bin/env python 
# encoding: utf-8
"""
 多项式回归
 @Author : hanxiaopeng
 @Time : 2020/9/21
"""
import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler

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
    if i + 30 == len(np_data):
        break
    x = np_data[i: i + 30].tolist()
    y = np_data[i + 30].tolist()
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

model = LinearRegression()  # 线性回归

x_train = x_train.reshape(-1, 30 * 6)
history = model.fit(x_train, y_train)


# 滚动预测
x_history = x_train.reshape(-1, 6)
y_hat_list = []
for i in range(length):
    x = x_history[-30:, ].reshape(-1, 30 * 6)
    y_hat = model.predict(x).reshape(-1, 6)
    x_history = np.concatenate((x_history, y_hat), axis=0)
    y_hat_list.append(y_hat)

y_hat = np.array(y_hat_list).reshape(-1, 6)

# 反标准化和反归一化
y_hat = ss.inverse_transform(y_hat)
y_hat = mm.inverse_transform(y_hat)

y_test = ss.inverse_transform(y_test)
y_test = mm.inverse_transform(y_test)


# 绘制结果
figure = plt.figure(figsize=(18, 8))   #定制图框长宽
columns_list = df.columns              #标记特征名称
for i in range(y_hat.shape[1]):
    plt.subplot(2, 3, i + 1)
    plt.plot(y_hat[:, i], label='pred')
    plt.plot(y_test[:, i], label='true')
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