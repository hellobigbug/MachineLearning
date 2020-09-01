# -*- coding: utf-8 -*-

import matplotlib
from keras.layers import SimpleRNN, Activation, Dense
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.optimizers import Adam

# 定义参数

TIME_STEPS = 5  #  same as the height of the image
INPUT_SIZE = 6  #  same as the width of the image
BATCH_SIZE = 512  # 每个批次训练样本
OUTPUT_SIZE = 6  # 每张图片输出分类矩阵
CELL_SIZE = 30  # RNN中隐藏单元
LR = 0.001  # 学习率
BATCH_INDEX = 0 #分批截取数据

# 载入数据及预处理

df = pd.read_csv("ATMP数据.csv", header=0)
df = df.set_index('数据日期')

print(len(df))

# 查看数据状态曲线
# df[:].plot()
# plt.show()

np_data = np.array(df)

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

X_train = x_array[:int(len(x_array) * 0.75)]
y_train = y_array[:int(len(y_array) * 0.75)]
X_test = x_array[int(len(x_array) * 0.75):]
y_test = y_array[int(len(y_array) * 0.75):]

model = Sequential()

# RNN cell
model.add(SimpleRNN(
    batch_input_shape=(None, TIME_STEPS, INPUT_SIZE),
    output_dim=CELL_SIZE,
    unroll=True,
))

# output layer
model.add(Dense(OUTPUT_SIZE))  # 全连接层
model.add(Activation('sigmoid'))  # 激励函数

# 优化器optimizer
adam = Adam(LR)

# 激活神经网络
model.compile(optimizer=adam,
              loss='MSE',
              metrics=['accuracy'])

# 训练和预测
cost_list = []
acc_list = []
step_list = []
for step in range(4001):
    # 分批截取数据 BATCH_INDEX初始值为0 BATCH_SIZE为512 取5个步长和6个INPUT_SIZE
    # data shape = (batch_num, steps, inputs/outputs)
    X_batch = X_train[BATCH_INDEX: BATCH_INDEX + BATCH_SIZE, :, :]
    Y_batch = y_train[BATCH_INDEX: BATCH_INDEX + BATCH_SIZE, :]

    # 计算误差
    cost = model.train_on_batch(X_batch, Y_batch)

    # # 累加参数
    # BATCH_INDEX += BATCH_SIZE
    # # 如果BATCH_INDEX累加大于总体的个数 则重新赋值0开始分批计算
    # BATCH_INDEX = 0 if BATCH_INDEX >= X_train.shape[0] else BATCH_INDEX

    # 每隔200步输出
    if step % 200 == 0:
        # 评价算法
        cost, accuracy = model.evaluate(
            X_test, y_test,
            batch_size=y_test.shape[0],
            verbose=False)
        # 写入列表
        cost_list.append(cost)
        acc_list.append(accuracy)
        step_list.append(step)
        print('test cost: ', cost, 'test accuracy: ', accuracy)

# --------------------------------绘制相关曲线------------------------------
import matplotlib.pyplot as plt

# 绘制曲线图

# 绘图参数设定
matplotlib.rcParams['font.size'] = 20
matplotlib.rcParams['figure.titlesize'] = 20
matplotlib.rcParams['figure.figsize'] = [9, 7]
matplotlib.rcParams['font.family'] = ['STKaiTi']
matplotlib.rcParams['axes.unicode_minus'] = False

plt.figure()
x = [i * 80 for i in range(len(cost_list))]
plt.plot(x, cost_list, color='C0', marker='s', label='测试')
plt.ylabel('MSE')
plt.xlabel('Step')
plt.legend()
# plt.savefig('train.svg')

plt.figure()
plt.plot(x, acc_list, color='C1', marker='s', label='测试')
plt.ylabel('准确率')
plt.xlabel('Step')
plt.legend()
plt.show()
