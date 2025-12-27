import mnist_loader
import network

# 加载 MNIST 数据集（784维输入，10维输出one-hot）
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

# 创建一个 3 层神经网络：输入层784，隐藏层30，输出层10
net = network.Network([784, 30, 10])

# 使用小批量随机梯度下降训练：30个epoch，每批10张图，学习率为3.0
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)

# ----------------------------
# 测试部分：计算在 test_data 上的准确率
# ----------------------------

def test_accuracy(net, test_data):
    """评估网络在测试集上的准确率"""
    test_results = [(int(net.feedforward(x).argmax()), y) for (x, y) in test_data]
    correct = sum(int(pred == label) for (pred, label) in test_results)
    total = len(test_data)
    print(f"Test Accuracy: {correct} / {total} ({100.0 * correct / total:.2f}%)")

# 调用测试函数
test_accuracy(net, test_data)
