import tensorflow as tf
import numpy as np
import scipy.special
import matplotlib.pyplot


# neural network class definition
class neuralNetwork :

    # init the network
    def __init__(self , inputnodes, hiddennodes, outputnodes, learningrate):
        # set number of nodes of each layer
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        # learning rate
        self.lr = learningrate

        # 创建权重矩阵wih和who
        # wih的意思是从输入层到隐藏层的矩阵，大小为hnodes*inodes
        # who的意思是从隐藏层到输出层的矩阵，大小为onodes*hnodes
        # np.random.rand(row，col)的功能是创建一个大小为row*col的随机数构成的矩阵
        # 在这里让矩阵的初始值，即权重的初始值为-0.5-0.5之间的随机数
        self.wih = (np.random.rand(self.hnodes, self.inodes)-0.5)
        self.who = (np.random.rand(self.onodes, self.hnodes)-0.5)

        # 激活函数，即sigmoid（x）
        self.activation_function = lambda x: scipy.special.expit(x)

        pass

    # train the network
    # 第一部分：针对给定的训练样本计算输出，这与query函数的几乎没有区别
    # 第二部分：将计算所得到的输出与目标值（已知）进行对比，使用差值来指导网络进行更新
    def train(self, inputs_list, targets_list):
        # 将输入值转换成矩阵（列向量）
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        # 计算hidden层的输入值
        hidden_inputs = np.dot(self.wih, inputs)

        # 计算hidden层的输出值，即sigmoid（hidden_inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        # 计算output层的输入值，即hidden层的输出值乘以权重矩阵
        final_inputs = np.dot(self.who, hidden_outputs)

        # 计算output的输出值，即最后的输出值
        final_outputs = self.activation_function(final_inputs)

        # --------------------------------------------
        # 一直到上面为止，这段代码与query（）中的几乎一模一样
        # --------------------------------------------

        # 计算最后输出值与期望值之间的差值，即误差e
        output_errors = targets - final_outputs

        # 计算hidden层输出的数据的误差，是以output层的误差按照权重分配的
        hidden_errors = np.dot(self.who.T, output_errors)

        # 现在，在进行一次正向传播之后，就已经获得了hidden和outputs层输出数据的误差
        # 接下来，根据误差，结合梯度下降发来对W进行优化

        # hidden层和outputs层之间的W的更新
        self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), np.transpose(hidden_outputs))

        # inputs层和hidden层之间的W的更新
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))

        pass

    # query the network，即接受神经网络的输入，返回神经网络的输出
    # query : 查询
    def query(self, inputs_list):
        # 将输入数据转换成矩阵
        inputs = np.array(inputs_list, ndmin=2).T

        # 计算hidden层的输入值
        hidden_inputs = np.dot(self.wih, inputs)

        # 计算hidden层的输出值，即sigmoid（hidden_inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        # 计算output层的输入值，即hidden层的输出值乘以权重矩阵
        final_inputs = np.dot(self.who, hidden_outputs)

        # 计算output的输出值，即最后的输出值
        final_outputs = self.activation_function(final_inputs)

        return final_outputs


if __name__ == '__main__':

    ### 对数据进行处理和打印（这一块仅供测试之用）
    ### -----开始-----
    ### 将文本中的逗号去掉
    # all_values = data_list[0].split(',')
    ### 将除了第一个标示符以外的字符串数据转换成数字，并重新聚合成28*28的矩阵
    # image_array = np.asfarray(all_values[1:]).reshape((28, 28))
    ### 画出图像，并且在点按鼠标后关闭图像
    # matplotlib.pyplot.imshow(image_array, cmap='Greys', interpolation='None')
    # matplotlib.pyplot.waitforbuttonpress()
    ### -----结束-----

    # -------------------------下面开始对数据进行处理，准备放到神经网络里面进行训练-----------------------

    # 开始创建神经网络类，首先定义各层结点的数量
    input_nodes = 784
    hidden_nodes = 100
    output_nodes = 10       # 由于要判断0-9十个数字，所以当输入一个测试集之后，最后哪个输出结点表现出了比较大的数字，就说明是哪个
    learning_rate = 0.3

    # 创建神经网络类实例
    n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

    # 导入训练集数据
    # data_[i]表示第i条数据，即第i个手写文字
    training_data_file = open("mnist_train.csv", 'r')
    training_data_list = training_data_file.readlines()
    training_data_file.close()

    count = 1.0
    # ---------------------------------开始训练--------------------------------
    # 遍历整个训练集，导入数据
    # record表示每个训练实例，在每次循环过程中增加一个，指向下一个训练实例
    for record in training_data_list:
        # 将逗号去掉
        all_values = record.split(',')

        # 将每个训练实例的第一个字符取出来，其余的输入进inputs
        inputs = (np.asfarray(all_values[1:])/255 * 0.99) + 0.01

        # 创建target集，在这里target为一个10*1（onodes*1）的列向量
        # 这个测试集的数字实际是几，第几行的值就是0.99，其余是0.01
        targets = np.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99

        # 开始训练
        n.train(inputs, targets)
        print("Has trained ", count, "times.")
        count += 1
        pass

    # ---------------------------------训练结束--------------------------------

    # ---------------------------------开始测试--------------------------------
    # 导入测试集
    test_data_file = open("mnist_test.csv", 'r')
    test_data_list = test_data_file.readlines()
    test_data_file.close()

    # all_test_values = test_data_list[0].split(',')
    # print(all_test_values[0])
    # print(n.query((np.asfarray(all_test_values[1:])/255*0.99)+0.01))

    # 定义一个记录板
    scorecard = []

    # 遍历所有测试集
    for record in test_data_list:
        # 去掉每一个测试集里面的逗号
        all_values = record.split(',')

        # 把正确的答案拿出来
        correct_label = int(all_values[0])
        print(correct_label, "is correct label")

        # 把每个训练集的第一个字符取出来，放到input里
        inputs = (np.asfarray(all_values[1:])/255 * 0.09) + 0.01

        # 测试神经网络，输出一个10*1的矩阵
        outputs = n.query(inputs)

        # 找到输出的矩阵中最大的那个数字，即为输出。
        # np.argmax()返回的是最大的那个数字的索引
        label = np.argmax(outputs)
        print(label, "network's answer")

        # 制作计分卡
        if (label == correct_label):
            scorecard.append(1)
        else:
            scorecard.append(0)
            pass
        pass

    # 计算准确率
    right = 0.0
    wrong = 0.0
    for i in scorecard:
        if (i == 1):
            right += 1
        else:
            wrong += 1
            pass
        pass
    print("正确率：", right/(right+wrong))


    # 打印计分卡
    print(scorecard)

    # ---------------------------------测试结束--------------------------------








