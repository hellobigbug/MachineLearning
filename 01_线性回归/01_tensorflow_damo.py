#!/usr/bin/env python 
# encoding: utf-8 
"""
 @Author : hanxiaopeng
 @Time : 2020/8/27 
"""
import os

import matplotlib
import tensorflow as tf
from matplotlib import pyplot as plt
# 绘图参数设定
from sklearn import preprocessing

matplotlib.rcParams['font.size'] = 20
matplotlib.rcParams['figure.titlesize'] = 20
matplotlib.rcParams['figure.figsize'] = [9, 7]
matplotlib.rcParams['font.family'] = ['STKaiTi']
matplotlib.rcParams['axes.unicode_minus'] = False

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
print(tf.__version__)

# 数据导入
# (x, y), (x_test, y_test) = datasets.mnist.load_data()
import numpy as np
import pandas as pd

# 读取数据
df = pd.read_csv("../ATMP数据.csv", header=0)
df = df.set_index('数据日期')

# 数据量不足，用复制来增加
# df = pd.concat([df,df,df,df,df,df,df,df,df,df,df,df])
# df = pd.concat([df,df,df,df,df,df,df,df,df,df,df,df])[:60000]

print(len(df))

# 查看数据状态曲线
# df[:].plot()
# plt.show()

min_max_scaler = preprocessing.MinMaxScaler()
np_data = np.array(df)
np_data = min_max_scaler.fit_transform(np_data)

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
# x_array = min_max_scaler.fit_transform(x_array)
# x_array = normalize(x_array, axis=0, norm='max')
y_array = np.array(y_list)
# y_array = min_max_scaler.fit_transform(y_array)
# y_array = normalize(y_array, axis=0, norm='max')

# # 保存
# np.savez('data.npz', x_array=x_array, y_array=y_array)
# # 读取
# loaddata = np.load('data.npz')
#
# x_array = loaddata['x_array']
# y_array = loaddata['y_array']

# 可以直接训练的x和y
# print(x_array)
# print(y_array)


x = x_array[:int(len(x_array) * 0.75)]
y = y_array[:int(len(y_array) * 0.75)]
x_test = x_array[int(len(x_array) * 0.75):]
y_test = y_array[int(len(y_array) * 0.75):]

# tenroflow提供的数据集
# (x, y), (x_test, y_test) = datasets.mnist.load_data()

print('x:', x.shape, 'y:', y.shape, 'x test:', x_test.shape, 'y test:', y_test)


# 数据预处理
def preprocess(x, y):  # 自定义的预处理函数
    # 调用此函数时会自动传入x,y 对象，shape 为[b, 28, 28], [b]
    # 标准化到0~1
    x = tf.cast(x, dtype=tf.float32)
    x = tf.reshape(x, [-1, 5 * 6])  # 打平
    y = tf.cast(y, dtype=tf.float32)
    y = tf.reshape(y, [-1, 6])
    # y = tf.one_hot(y, depth=6)
    return x, y


batchsz = 64
train_db = tf.data.Dataset.from_tensor_slices((x, y))
train_db = train_db.shuffle(1000)  # 打乱顺序，缓冲池1000
train_db = train_db.batch(batchsz)  # 批训练，批规模
train_db = train_db.map(preprocess)
train_db = train_db.repeat(20)

#
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_db = test_db.shuffle(1000).batch(batchsz).map(preprocess)
x, y = next(iter(train_db))
print('train sample:', x.shape, y.shape)


def main():
    # learning rate
    lr = 1e-1
    accs, losses = [], []

    # 784 => 512
    w1, b1 = tf.Variable(tf.random.normal([30, 28], stddev=0.1, seed=1)), tf.Variable(
        tf.zeros([28]))  # stddev: 正态分布的标准差，默认为1.0
    # 512 => 256
    w2, b2 = tf.Variable(tf.random.normal([28, 12], stddev=0.1, seed=1)), tf.Variable(tf.zeros([12]))
    # 256 => 10
    w3, b3 = tf.Variable(tf.random.normal([12, 6], stddev=0.1, seed=1)), tf.Variable(tf.zeros([6]))

    for step, (x, y) in enumerate(train_db):

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

        if step % 80 == 0:
            # evaluate/test
            total, total_correct = 0., 0

            for step, (x, y) in enumerate(test_db):
                # layer1.
                h1 = x @ w1 + b1
                h1 = tf.nn.relu(h1)
                # layer2
                h2 = h1 @ w2 + b2
                h2 = tf.nn.relu(h2)
                # output
                out = h2 @ w3 + b3

                # 实际按行取最大的数的列索引信息
                y = tf.argmax(y, axis=1)
                y = tf.cast(y, dtype=tf.float32)
                prob = tf.nn.softmax(out, axis=1)
                # 按行取概率最大的数的列索引信息
                preb = tf.argmax(prob, axis=1)
                preb = tf.cast(preb, dtype=tf.float32)
                # 预测值与真实值比较
                # print(y.dtype, preb.dtype)
                correct = tf.cast(tf.equal(y, preb), dtype=tf.float32)
                correct = tf.reduce_sum(correct)
                total_correct += int(correct)
                total += x.shape[0]

            print(step, 'Evaluate Acc:', total_correct / total)

            accs.append(total_correct / total)

    plt.figure()
    x = [i * 80 for i in range(len(losses))]
    plt.plot(x, losses, color='C0', marker='s', label='训练')
    plt.ylabel('Loss')
    plt.xlabel('Step')
    plt.legend()
    # plt.savefig('train.svg')

    plt.figure()
    plt.plot(x, accs, color='C1', marker='s', label='测试')
    plt.ylabel('准确率')
    plt.xlabel('Step')
    plt.legend()
    plt.show()
    # plt.savefig('test.svg')


if __name__ == '__main__':
    main()
