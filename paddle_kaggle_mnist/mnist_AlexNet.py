#引用需要的库和函数
import paddle
import paddle.fluid as fluid
import paddle.fluid.dygraph as dygraph
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, BatchNorm, Linear
from paddle.fluid.dygraph.base import to_variable

import numpy as np
import pandas as pd
import os
import csv

import matplotlib.pyplot as plot


# 获取数据
def load_dataset(batch_size=100):
    # 获取训练集
    df = pd.read_csv('train.csv')  # 得到的是一个字典集
    f1 = [f"pixel{i}" for i in range(0, 28 * 28)]  # 产生字符串列表，从pixel0到pixel783
    f2 = 'label'
    train_x = np.array(df[f1].values)  # 通过键获取字典数据，并且转化为矩阵
    train_y = np.array(df[f2].values)

    '''
    train_y=pd.Series(train_y)
    train_y=np.array(pd.get_dummies(train_y))#独热码实现softmax
    '''
    # print(train_y[0:12])#查看一部分标签内容
    # print(train_x.shape[0], train_x.shape[1])  # 输出训练集特征x维度,为（42000，784）
    # print(train_y.shape[0])  # 输出标签y维度（42000，）

    # 将数据处理成（样本数，通道数，长，宽）形式,像素化为0-1之间的数
    train_x = np.reshape(train_x, [train_x.shape[0], 1, 28, 28]).astype(np.float32) / 255
    train_y = np.reshape(train_y, [train_y.shape[0], 1]).astype(np.int64)

    # 定义数据读取器
    def data_generator():
        data_lists = []  # 存储minibatch的训练数据像素
        label_list = []  # 存储minibatch的训练数据标签

        for id, data in enumerate(train_x):
            data_lists.append(data)
            label_list.append(train_y[id])

            if len(data_lists) == batch_size:  # 每minibatch存储一次，batch_size设置成100大小
                yield np.array(data_lists), np.array(label_list)
                data_lists = []  # 清空存储器
                label_list = []
        if len(data_lists) > 0:  # 其余的作为一组数据
            yield np.array(data_lists), np.array(label_list)

    return data_generator()


def load_test(batch_size=100):
    # 获取测试集
    dp = pd.read_csv('test.csv')  # 得到的是一个字典集
    f = [f"pixel{i}" for i in range(0, 28 * 28)]  # 产生字符串列表，从pixel0到pixel783
    test_x = np.array(dp[f].values)  # 通过键获取字典数据，并且转化为矩阵
    # print(test_x.shape[0], test_x.shape[1])  # 输出维度，为（28000，784）

    # 将数据处理成（样本数，通道数，长，宽）形式,像素化为0-1之间的数
    test_x = np.reshape(test_x, [test_x.shape[0], 1, 28, 28]).astype(np.float32) / 255

    # 定义数据读取器
    def data_generator():
        data_lists = []  # 存储minibatch的训练数据像素

        for id, data in enumerate(test_x):
            data_lists.append(data)

            if len(data_lists) == batch_size:  # 每minibatch存储一次，batch_size设置成100大小
                yield np.array(data_lists)
                data_lists = []  # 清空存储器

        if len(data_lists) > 0:  # 其余的作为一组数据
            yield np.array(data_lists)

    return data_generator()


"""
a = load_dataset()
nums = 0
for i,set in enumerate(a):
    nums = nums+1
x,y = set
print(nums)#看看总minibatch个数是不是 42000/batch_size 大小
print(x.shape[0],x.shape[1],x.shape[2],x.shape[3])#输出最后一个minibatch维度看看
print(y.shape[0],y.shape[1])#输出最后一个minibatch维度看看
"""


# 定义 AlexNet 网络结构
class AlexNet(fluid.dygraph.Layer):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()

        # AlexNet与LeNet一样也会同时使用卷积和池化层提取图像特征
        # 与LeNet不同的是激活函数换成了‘relu’
        # 这里将conv1中的输入通道数设置成1，输入为（样本数m，1，28，28）
        self.conv1 = Conv2D(num_channels=1, num_filters=10, filter_size=11, stride=1, padding=3, act='relu')
        # 输出为（m,10，24，24）
        self.pool1 = Pool2D(pool_size=2, pool_stride=2, pool_type='max')
        # 输出为（m,10,12,12）
        self.conv2 = Conv2D(num_channels=10, num_filters=100, filter_size=5, stride=1, padding=2, act='relu')
        # （m,100,12,12）
        self.pool2 = Pool2D(pool_size=2, pool_stride=2, pool_type='max')
        # (m,100,6,6)
        self.conv3 = Conv2D(num_channels=100, num_filters=200, filter_size=3, stride=1, padding=0, act='relu')
        # (m,200,4,4)
        self.conv4 = Conv2D(num_channels=200, num_filters=200, filter_size=3, stride=1, padding=1, act='relu')
        # (m,200,4,4)
        self.conv5 = Conv2D(num_channels=200, num_filters=100, filter_size=3, stride=1, padding=1, act='relu')
        # (m,100,4,4)
        self.pool5 = Pool2D(pool_size=2, pool_stride=2, pool_type='max')  # 相当于无效
        # (m,100,2,2)
        self.fc1 = Linear(input_dim=400, output_dim=64, act='relu')
        self.drop_ratio1 = 0.5
        self.fc2 = Linear(input_dim=64, output_dim=64, act='relu')
        self.drop_ratio2 = 0.5
        self.fc3 = Linear(input_dim=64, output_dim=num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool5(x)
        x = fluid.layers.reshape(x, [x.shape[0], -1])
        x = self.fc1(x)
        # 在全连接之后使用dropout抑制过拟合
        x = fluid.layers.dropout(x, self.drop_ratio1)
        x = self.fc2(x)
        # 在全连接之后使用dropout抑制过拟合
        x = fluid.layers.dropout(x, self.drop_ratio2)
        x = self.fc3(x)
        return x

    # 定义训练过程
    def train(model):
        print('start training ... ')
        model.train()
        epoch_num = 1
        batch_size = 10
        # 定义学习率，并加载优化器参数到模型中
        total_steps = (int(42000 // batch_size) + 1) * epoch_num
        lr = fluid.dygraph.PolynomialDecay(0.001, total_steps, 0.0001)

        opt = fluid.optimizer.Momentum(learning_rate=lr, momentum=0.9, parameter_list=model.parameters())

        for epoch in range(epoch_num):
            for batch_id, data in enumerate(load_dataset(batch_size)):
                # 读入数据
                x_data, y_data = data
                # 将numpy.ndarray转化成Tensor
                img = fluid.dygraph.to_variable(x_data)
                label = fluid.dygraph.to_variable(y_data)
                # 计算模型输出
                logits = model(img)
                # 计算损失函数
                loss = fluid.layers.softmax_with_cross_entropy(logits, label)
                avg_loss = fluid.layers.mean(loss)
                if batch_id % 1000 == 0:
                    print("epoch: {}, batch_id: {}, loss is: {}".format(epoch, batch_id, avg_loss.numpy()))
                # 更新梯度
                avg_loss.backward()
                opt.minimize(avg_loss)
                # 清除梯度
                model.clear_gradients()

                # 在训练集上的评估结果
            model.eval()
            accuracies = []
            losses = []
            for batch_id, data in enumerate(load_dataset(batch_size)):
                # 调整输入数据形状和类型
                x_data, y_data = data

                # 将numpy.ndarray转化成Tensor
                img = fluid.dygraph.to_variable(x_data)
                label = fluid.dygraph.to_variable(y_data)
                # 计算模型输出
                logits = model(img)
                pred = fluid.layers.softmax(logits)
                # 计算损失函数
                loss = fluid.layers.softmax_with_cross_entropy(logits, label)
                acc = fluid.layers.accuracy(pred, label)
                accuracies.append(acc.numpy())
                losses.append(loss.numpy())
            print("[train] accuracy/loss: {}/{}".format(np.mean(accuracies), np.mean(losses)))
            model.train()

        # 预测结果:
        model.eval()
        # 获取测试集
        # 将结果写入LeNet.csv文件
        lenet = open("AlexNet.csv", "w")
        lenet.write("ImageId,Label\n")
        for id, test_data in enumerate(load_test(batch_size=1000)):
            # 将numpy.ndarray转化成Tensor
            img = fluid.dygraph.to_variable(test_data)
            # 计算模型输出
            logits = model(img)
            # 输出softmax层结果，得到图片分类
            predict = fluid.layers.softmax(logits).numpy()
            # 将预测结果最大值下标作为分类结果对应0-9

            label = []
            for i in range(predict.shape[0]):
                label.append([i + 1, np.argmax(predict[i])])

            for pred in label:
                id, y = pred[0], pred[1]
                lenet.write(str(id) + "," + str(y) + "\n")
        # 保存模型参数
        fluid.save_dygraph(model.state_dict(), 'mnist_LeNet')

    if __name__ == '__main__':
        # 创建模型
        # 是否使用gpu
        use_gpu = True  # 确认使用gpu
        place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace
        with fluid.dygraph.guard(place):
            """
            model = LeNet(num_classes=10)
            #启动训练过程
            train(model)

            """
            model = AlexNet(num_classes=10)
            train(model)
