# %%
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, losses, optimizers, Sequential
from matplotlib import pyplot as plt
import matplotlib
from tensorflow.python.framework.ops import reset_default_graph

tf.random.set_seed(22)
np.random.seed(22)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
assert tf.__version__.startswith('2.')

batchsz = 512  # 批量大小
total_words = 10000  # 词汇表大小N_vocab
max_review_len = 80  # 句子最大长度s，大于的句子部分将截断，小于的将填充
embedding_len = 100  # 词向量特征长度f
# 加载IMDB数据集，此处的数据采用数字编码，一个数字代表一个单词

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


# 数据预处理
def preprocess(x, y):  # 自定义的预处理函数
    x = tf.cast(x, dtype=tf.float32) / 255.
    # x = tf.reshape(x, [-1, 5 * 6])  # 打平
    y = tf.cast(y, dtype=tf.float32)
    y = tf.reshape(y, [-1, 6])
    return x, y


batchsz = 512
db_train = tf.data.Dataset.from_tensor_slices((X_train, y_train))
db_train = db_train.shuffle(1000)  # 打乱顺序，缓冲池1000
db_train = db_train.batch(batchsz, drop_remainder=True)  # 批训练，批规模
db_train = db_train.map(preprocess)
db_train = db_train.repeat(20)


db_test = tf.data.Dataset.from_tensor_slices((X_test, y_test))
db_test = db_test.shuffle(1000)
db_test = db_test.batch(batchsz, drop_remainder=True)
db_test = db_test.map(preprocess)
x, y = next(iter(db_train))
print('train sample:', x.shape, y.shape)



# %%

class MyRNN(keras.Model):
    # Cell方式构建多层网络
    def __init__(self, units):
        super(MyRNN, self).__init__()
        # [b, 64]，构建Cell初始化状态向量，重复使用
        self.state0 = [tf.zeros([batchsz, units])]
        self.state1 = [tf.zeros([batchsz, units])]
        # 词向量编码 [b, 80] => [b, 80, 100]
        # self.embedding = layers.Embedding(total_words, embedding_len,
        #                                   input_length=max_review_len)
        # 构建2个Cell
        self.rnn_cell0 = layers.SimpleRNNCell(units, dropout=0.5)
        self.rnn_cell1 = layers.SimpleRNNCell(units, dropout=0.5)
        # 构建分类网络，用于将CELL的输出特征进行分类，2分类
        # [b, 80, 100] => [b, 64] => [b, 1]
        self.outlayer = Sequential([
            layers.Dense(units),
            layers.Dropout(rate=0.5),
            layers.ReLU(),
            layers.Dense(6)])

    def call(self, inputs, training=None):
        x = inputs  # [b, 80]
        # embedding: [b, 80] => [b, 80, 100]
        # x = self.embedding(x)
        # rnn cell compute,[b, 80, 100] => [b, 64]
        state0 = self.state0
        state1 = self.state1
        for word in tf.unstack(x, axis=1):  # word: [b, 100]
            out0, state0 = self.rnn_cell0(word, state0, training)
            out1, state1 = self.rnn_cell1(out0, state1, training)
        # 末层最后一个输出作为分类网络的输入: [b, 64] => [b, 1]
        x = self.outlayer(out1, training)
        # p(y is pos|x)
        prob = tf.sigmoid(x)

        return prob




accs = []
losses = []
def main():
    units = 6  # RNN状态向量长度f
    epochs = 10  # 训练epochs

    model = MyRNN(units)
    # 装配
    model.compile(optimizer=optimizers.RMSprop(0.001),
                  loss='MSE',
                  metrics=['accuracy'])
    # 训练和验证
    # history = model.fit(db_train, epochs=epochs, validation_data=db_test)
    # loss_train = history.history.get('loss')
    # acce_val = history.history.get('val_accuracy')
    # # 测试
    # loss_test, acce_test = model.evaluate(db_test)

    x = X_train
    # x = tf.random.normal([4, 80, 100])
    xt = x[:, 0, :]  # 取第一个时间戳的输入 x0
    # 构建 2 个 Cell,先 cell0,后 cell1，内存状态向量长度都为 64
    cell0 = layers.SimpleRNNCell(6)
    cell1 = layers.SimpleRNNCell(6)
    h0 = [tf.zeros([512, 6])]  # cell0 的初始状态向量
    h1 = [tf.zeros([512, 6])]  # cell1 的初始状态向量

    optimizer = optimizers.SGD(learning_rate=0.001)

    for step, (x, y) in enumerate(db_train):


        # # [b, 28, 28] => [b, 784]
        # x = tf.reshape(x, (-1, 30))

        with tf.GradientTape() as tape:
        # with tf.Session() as sess:
        #     sess.run(tf.global_variables_initializer())


            for xt in tf.unstack(x, axis=1):
                # xt 作为输入，输出为 out0
                out0, h0 = cell0(xt, h0)
                # 上一个 cell 的输出 out0 作为本 cell 的输入
                out1, h1 = cell1(out0, h1)
            # layer1.
            # h1 = x @ w1 + b1
            # h1 = tf.nn.relu(h1)
            # # layer2
            # h2 = h1 @ w2 + b2
            # h2 = tf.nn.relu(h2)
            # # output
            # out = h2 @ w3 + b3
            # out = tf.nn.relu(out)

            # 求误差
            loss = tf.square(y - out1)
            # 求误差的请平均值
            loss = tf.reduce_mean(loss)

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))


        # 借助于 tensorflow 自动求导
        # grads = tape.gradient(loss, model.variables)
        # tf.keras.optimizers.Optimizer.apply_gradients(zip(grads, model.variables))
        # optimizer.apply_gradients(zip(grads, model.variables))
        # # 根据梯度更新参数
        # for p, g in zip(model.variables, grads):
        #     p.assign_sub(lr * g)

        # 每迭代80次输出一次loss
        if step % 80 == 0:
            print(step, 'loss:', float(loss))
            losses.append(float(loss))

        if step % 80 == 0:
            # evaluate/test
            total, total_correct = 0., 0

            for x, y in db_test:
                for xt in tf.unstack(x, axis=1):
                    # xt 作为输入，输出为 out0
                    out0, h0 = cell0(xt, h0)
                    # 上一个 cell 的输出 out0 作为本 cell 的输入
                    out1, h1 = cell1(out0, h1)
                pred = tf.argmax(out1, axis=1)
                # convert one_hot y to number y
                y = tf.argmax(y, axis=1)
                # bool type
                correct = tf.equal(pred, y)
                # bool tensor => int tensor => numpy
                total_correct += tf.reduce_sum(tf.cast(correct, dtype=tf.int32)).numpy()
                total += x.shape[0]

            print(step, 'Evaluate Acc:', total_correct / total)

            accs.append(total_correct / total)


    # 绘图参数设定
    matplotlib.rcParams['font.size'] = 20
    matplotlib.rcParams['figure.titlesize'] = 20
    matplotlib.rcParams['figure.figsize'] = [9, 7]
    matplotlib.rcParams['font.family'] = ['STKaiTi']
    matplotlib.rcParams['axes.unicode_minus'] = False

    plt.figure()
    x = [i * 80 for i in range(len(losses))]
    plt.plot(x, losses, color='C0', marker='s', label='训练')
    plt.ylabel('MSE')
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
