#!/usr/bin/env python 
# encoding: utf-8 
"""
 @Author : hanxiaopeng
 @Time : 2020/9/16 
"""
import matplotlib
import pandas as pd
from matplotlib import pyplot as plt

# vol = .030
# lag = 300
# df = pd.DataFrame(np.random.randn(300) * np.sqrt(vol) * np.sqrt(1 / 252.)).cumsum()
# plt.plot(df[0].tolist())
# plt.show()

# df = pd.read_csv("../ATMP数据.csv", header=0)
# df = pd.read_csv("../VE_ETL_TRANSTIMES_BAK.csv", header=0)
df = pd.read_csv("../DAILY_LOAD_CAPA_BAK.csv", header=0)
df = df.set_index('数据日期')
df_org = df

df.columns = ['DB_TIME', 'DB_TPS', 'DB_QPS']
dflast = df[~df.index.duplicated('last')]
dflast.to_csv('last.csv')

dffirst = df[~df.index.duplicated('first')]
dffirst.to_csv('first.csv')

dfmin = df[df.index.duplicated('first')]
dfmin = dfmin[~dfmin.index.duplicated('first')]
dfmin.to_csv('2.csv')

print(df)

matplotlib.rcParams['font.family'] = ['STKaiTi']
matplotlib.rcParams['axes.unicode_minus'] = False
df.plot()
plt.show()
#
# np_data = np.array(df)
# min_max_scaler = preprocessing.MinMaxScaler()
# np_data = min_max_scaler.fit_transform(np_data)
# np_data = StandardScaler().fit_transform(np_data)
#
# x_list = []
# y_list = []
# for i in range(len(np_data)):
#     if i + 6 == len(np_data):
#         break
#     x = np_data[i: i + 5].tolist()
#     y = np_data[i + 6].tolist()
#     x_list.append(x)
#     y_list.append(y)
#
# x_array = np.array(x_list)
# # x_array = normalize(x_array, axis=0, norm='max')
# y_array = np.array(y_list)
# # y_array = normalize(y_array, axis=0, norm='max')
#
# # 划分训练集，验证集，测试集，比例为8：2：2
# x_train = x_array[:int(len(x_array) * 0.6)]
# x_valid = x_array[int(len(x_array) * 0.6):int(len(x_array) * 0.8)]
# x_test = x_array[int(len(x_array) * 0.8):]
# y_train = y_array[:int(len(y_array) * 0.6)]
# y_valid = y_array[int(len(y_array) * 0.6):int(len(y_array) * 0.8)]
# y_test = y_array[int(len(y_array) * 0.8):]
#
#
#
# figure = plt.figure(figsize=(15, 6))
# x = [i for i in range(y_test.shape[0])]
#
# ax1 = figure.add_subplot(2, 3, 1)
# ax2 = figure.add_subplot(2, 3, 2)
# ax3 = figure.add_subplot(2, 3, 3)
# ax4 = figure.add_subplot(2, 3, 4)
# ax5 = figure.add_subplot(2, 3, 5)
# ax0 = figure.add_subplot(2, 3, 6)
# y_test = y_test.reshape(-1, 6)
#
# ax0.plot(x, y_test[:, 0])
#
# ax1.plot(x, y_test[:, 1])
#
# ax2.plot(x, y_test[:, 2])
#
# ax3.plot(x, y_test[:, 3])
#
# ax4.plot(x, y_test[:, 4])
#
# ax5.plot(x, y_test[:, 5])
#
# plt.show()
#
