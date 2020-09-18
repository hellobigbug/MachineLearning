#!/usr/bin/env python 
# encoding: utf-8 
"""
 @Author : hanxiaopeng
 @Time : 2020/8/27 
"""
import os
import time

import matplotlib
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from tensorflow.python.keras.metrics import MAE

matplotlib.rcParams['font.size'] = 20
matplotlib.rcParams['figure.titlesize'] = 20
matplotlib.rcParams['figure.figsize'] = [9, 7]
matplotlib.rcParams['font.family'] = ['STKaiTi']
matplotlib.rcParams['axes.unicode_minus'] = False

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
print(tf.__version__)

# 数据导入
# (x, y), (x_valid, y_valid) = datasets.mnist.load_data()
import numpy as np
import pandas as pd

starttime = time.time()

# 读取数据
df = pd.read_csv("../data.csv", header=0)
df = df.set_index('date')

print(len(df))

# 查看数据状态曲线
# df[:].plot()
# plt.show()

np_data = np.array(df)
min_max_scaler = preprocessing.MinMaxScaler()
np_data = min_max_scaler.fit_transform(np_data)
np_data = StandardScaler().fit_transform(np_data)

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
    lis.append([x, y])

# 要保存的矩阵样式
# print(lis)

x_array = np.array(x_list)
y_array = np.array(y_list)

# 预测时间长度
length = 30

x_test = x_array[-length:]
y_test = y_array[-length:]
len_train = int(len(x_array[:-length]))

# 划分训练集，验证集，测试集，比例为8：2：2
x_train = x_array[:int(len_train * 0.6)]
x_valid = x_array[int(len_train * 0.6):]

y_train = y_array[:int(len_train * 0.6)]
y_valid = y_array[int(len_train * 0.6):]


# # 保存
# np.savez('data.npz', x_array=x_array, y_array=y_array)
# # 读取
# loaddata = np.load('data.npz')
#
# x_array = loaddata['x_array']
# y_array = loaddata['y_array']


# 数据预处理
def preprocess(x, y):  # 自定义的预处理函数
    # 调用此函数时会自动传入x,y 对象，shape 为[b, 28, 28], [b]
    # 标准化到0~1
    x = tf.cast(x, dtype=tf.float32)
    x = tf.reshape(x, [-1, 5 * 6])  # 打平
    y = tf.cast(y, dtype=tf.float32)
    y = tf.reshape(y, [-1, 6])
    return x, y


batchsz = 32
train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_db = train_db.shuffle(1000)  # 打乱顺序，缓冲池1000.
train_db = train_db.batch(batchsz)  # 批训练，批规模
train_db = train_db.map(preprocess)
train_db = train_db.repeat(20)

valid_db = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
valid_db = valid_db.shuffle(1000)
valid_db = valid_db.batch(batchsz).map(preprocess)
x, y = next(iter(train_db))
print('train sample:', x.shape, y.shape)

lr = 1e-3
accs, losses = [], []

w1, b1 = tf.Variable(tf.random.normal([30, 20], stddev=0.1, seed=1)), tf.Variable(
    tf.zeros([20]))  # stddev: 正态分布的标准差，默认为1.0
w2, b2 = tf.Variable(tf.random.normal([20, 10], stddev=0.1, seed=1)), tf.Variable(tf.zeros([10]))
w3, b3 = tf.Variable(tf.random.normal([10, 6], stddev=0.1, seed=1)), tf.Variable(tf.zeros([6]))

for step, (x, y) in enumerate(train_db):
    # if len(losses) >= 500:
    #     break

    with tf.GradientTape() as tape:

        # layer1.
        h1 = x @ w1 + b1
        h1 = tf.nn.relu(h1)
        # layer2
        h2 = h1 @ w2 + b2
        h2 = tf.nn.relu(h2)
        # output
        out = h2 @ w3 + b3
        # out = tf.nn.relu(out)

        # 求误差
        loss = tf.square(y - out)
        # 求误差的请平均值
        loss = tf.reduce_mean(loss)

    # 借助于 tensorflow 自动求导
    grads = tape.gradient(loss, [w1, b1, w2, b2, w3, b3])

    # 根据梯度更新参数
    for p, g in zip([w1, b1, w2, b2, w3, b3], grads):
        p.assign_sub(lr * g)

    # 每迭代80次输出一次loss
    if step % 80 == 0:
        print(step, 'loss:', float(loss))
        losses.append(float(loss))

        # if step % 80 == 0:
        #     # evaluate/valid
        total, total_correct = 0., 0

        for step, (x, y) in enumerate(valid_db):
            # layer1.
            h1 = x @ w1 + b1
            h1 = tf.nn.relu(h1)
            # layer2
            h2 = h1 @ w2 + b2
            h2 = tf.nn.relu(h2)
            # output
            out = h2 @ w3 + b3

            correct = MAE(y, out)
            correct = tf.reduce_sum(correct)
            total_correct += int(correct)
            total += x.shape[0]

        print(step, 'Evaluate Acc:', total_correct / total)

        accs.append(total_correct / total)

plt.figure()
x = [i * 80 for i in range(len(losses))]

min_indx = np.argmin(losses)
plt.plot(x, losses, color='C0', marker='s', label='训练')
plt.ylabel('线性回归 Loss')
plt.xlabel('Step')
plt.plot(min_indx, losses[min_indx], 'gs')
plt.legend()
# plt.savefig('train.svg')

plt.figure()
x0 = accs.index(min(accs)) * 80
y0 = min(accs)
# 标记折线最低点，xy是标记点位置，txtest是文本标注位置，文本偏移x-300，y+0.1以避免遮盖折现
plt.annotate('最低点: %s' % round(y0, 3), xy=(x0, y0), xytext=(x0 - 300, y0 + 0.1),
             arrowprops=dict(facecolor='black', shrink=0.00001))
plt.plot(x, accs, color='C1', marker='s', label='测试')
plt.ylabel('线性回归 valid mae')
plt.xlabel('Step')
plt.legend()
plt.show()
# plt.savefig('valid.svg')
x_test = x_test.reshape(-1, 30)
# 使用r方验证准确性
h1 = x_test @ w1 + b1
h1 = tf.nn.relu(h1)
# layer2
h2 = h1 @ w2 + b2
h2 = tf.nn.relu(h2)
# output
out = h2 @ w3 + b3

yhat_test = out
r2_test = r2_score(y_test, yhat_test)  # 这里调用内置函数计算

# 保留三位小数
print('r2_score : ', round(r2_test, 3))
print('use time: ', round(time.time() - starttime, 3), 's')

"""
结果为：
r2_score :  0.722
use time:  19.945 s
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
