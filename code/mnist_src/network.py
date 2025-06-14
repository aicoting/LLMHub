import random
import time
import numpy as np

class Network(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            time1 = time.time()
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            time2 = time.time()
            if test_data:
                print("Epoch {0}: {1} / {2}, took {3:.2f} seconds".format(
                    j, self.evaluate(test_data), n_test, time2-time1))
            else:
                print("Epoch {0} complete in {1:.2f} seconds".format(j, time2-time1))

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """返回一个元组 (nabla_b, nabla_w)，表示损失函数 C_x 对每层偏置和权重的梯度"""
    
        # 初始化 nabla_b 和 nabla_w，结构与 biases 和 weights 相同，元素为 0，用于累计梯度
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
    
        # 前向传播开始，设置初始输入 activation 为 x
        activation = x
    
        # 存储所有层的激活值（包括输入层）
        activations = [x]
    
        # 存储所有层的加权输入 z（即 z = w·a + b）
        zs = []
    
        # 遍历每层的权重 w 和偏置 b，执行前向传播
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b       # 计算加权输入 z
            zs.append(z)                         # 保存 z 值供后向传播使用
            activation = sigmoid(z)              # 计算当前层的激活值
            activations.append(activation)       # 保存激活值供后向传播使用
    
        # 反向传播开始 —— 先计算输出层的误差 delta
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        
        # 输出层偏置的梯度就是 delta
        nabla_b[-1] = delta
    
        # 输出层权重的梯度是 delta 与上一层激活值的转置的点积
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
    
        # 从倒数第二层开始，逐层向前传播误差
        for l in range(2, self.num_layers):
            z = zs[-l]                            # 当前层的 z
            sp = sigmoid_prime(z)                # 当前层 z 的 sigmoid 导数
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp  # 传播误差
            nabla_b[-l] = delta                   # 当前层偏置的梯度
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())  # 当前层权重的梯度
    
        # 返回每层偏置和权重的梯度
        return (nabla_b, nabla_w)
    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)
    def cost_derivative(self, output_activations, y):
        """返回损失函数对输出层激活值的偏导数，即 ∂C / ∂a"""
        return (output_activations - y)  # 平方损失函数的导数：a - y

def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """sigmoid 函数的导数，用于反向传播时计算梯度"""
    return sigmoid(z) * (1 - sigmoid(z))  # 利用公式：σ'(z) = σ(z) * (1 - σ(z))