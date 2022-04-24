# softmax回归的简洁实现
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from d2l import torch as d2l
from torch.utils import data
from torchvision import transforms
from IPython import display
from d2l import torch as d2l

# 从本地读取数据到内存
batch_size = 256
# data
path = r"G:\Jupyter\course1.1\data"
trans = transforms.ToTensor()

def get_dataloader_workers():  #@save
    """使用4个进程来读取数据"""
    return 4

# train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
datasets_train = torchvision.datasets.FashionMNIST(
    root=path, train=True, transform=trans, download=False)
datasets_test = torchvision.datasets.FashionMNIST(
    root=path, train=False, transform=trans, download=False)

train_iter = data.DataLoader(dataset=datasets_train, batch_size=batch_size, shuffle=True,
                             num_workers=get_dataloader_workers())
test_iter = data.DataLoader(dataset=datasets_test, batch_size=batch_size, shuffle=True,
                             num_workers=get_dataloader_workers())      

# PyTorch不会隐式地调整输入的形状。因此，
# 我们在线性层前定义了展平层（flatten），来调整网络输入的形状训练，或者说是调整参数，
# 都是为了最后使得损失函数最小的一个过程，梯度，反向传播就是为了找那个使得梯度下降最大，更新原来的参数，
# 然后再训练、再找，一遍遍的就完成了训练，使得损失函数也就最小了
# nn.Sequential():神经网络模块将按照在传入构造器的顺序依次被添加到计算图中执行，同时以神经网络模块为元素的有序字典也可以作为传入参数
# nn.Flatten():是将连续的几个维度展平成一个tensor（将一些维度合并），就是之前做的那个reshape的操作
# PyTorch的nn.Linear()是用于设置网络中的全连接层的，需要注意在二维图像处理的任务中，全连接层的输入与输出一般都设置为二维张量，
# 形状通常为[batch_size, size]，不同于卷积层要求输入输出是四维张量
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))

#初始化权重参数
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights)

# 重新审视Softmax的实现；在交叉熵损失函数中传递未规范化的预测，并同时计算softmax及其对数
# 交叉熵损失函数
loss = nn.CrossEntropyLoss(reduction='none')

# 优化算法；使用学习率为0.1的小批量随机梯度下降作为优化算法
# 优化这里就是要进行参数更新，反向传播之后的，根据其梯度
trainer = torch.optim.SGD(net.parameters(), lr=0.1)

# # 分类精度
# def accuracy(y_hat, y):  #@save
#     """计算预测正确的数量"""
#     if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
#         y_hat = y_hat.argmax(axis=1)
#     cmp = y_hat.type(y.dtype) == y
#     return float(cmp.type(y.dtype).sum())

# # 评估在任意模型net的精度]
# def evaluate_accuracy(net, data_iter):  #@save
#     """计算在指定数据集上模型的精度"""
#     if isinstance(net, torch.nn.Module):
#         net.eval()  # 将模型设置为评估模式
#     metric = Accumulator(2)  # 正确预测数、预测总数
#     with torch.no_grad():
#         for X, y in data_iter:
#             metric.add(accuracy(net(X), y), y.numel())
#     return metric[0] / metric[1]

# # 对多个变量进行累加
# class Accumulator:  #@save
#     """在n个变量上累加"""
#     def __init__(self, n):
#         self.data = [0.0] * n

#     def add(self, *args):
#         self.data = [a + float(b) for a, b in zip(self.data, args)]

#     def reset(self):
#         self.data = [0.0] * len(self.data)

#     def __getitem__(self, idx):
#         return self.data[idx]


# # 训练
# def train_epoch_ch3(net, train_iter, loss, updater):  #@save
#     """训练模型一个迭代周期（定义见第3章）"""
#     # 将模型设置为训练模式
#     if isinstance(net, torch.nn.Module):
#         net.train()
#     # 训练损失总和、训练准确度总和、样本数
#     metric = Accumulator(3)
#     for X, y in train_iter:
#         # 计算梯度并更新参数
#         y_hat = net(X)
#         l = loss(y_hat, y)
#         if isinstance(updater, torch.optim.Optimizer):
#             # 使用PyTorch内置的优化器和损失函数
#             updater.zero_grad()
#             l.mean().backward()
#             updater.step()
#         else:
#             # 使用定制的优化器和损失函数
#             l.sum().backward()
#             updater(X.shape[0])
#         metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
#     # 返回训练损失和训练精度
#     return metric[0] / metric[2], metric[1] / metric[2]


# # 动画中绘制数据的实用程序类
# class Animator:  #@save
#     """在动画中绘制数据"""
#     def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
#                  ylim=None, xscale='linear', yscale='linear',
#                  fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
#                  figsize=(3.5, 2.5)):
#         # 增量地绘制多条线
#         if legend is None:
#             legend = []
#         d2l.use_svg_display()
#         self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
#         if nrows * ncols == 1:
#             self.axes = [self.axes, ]
#         # 使用lambda函数捕获参数
#         self.config_axes = lambda: d2l.set_axes(
#             self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
#         self.X, self.Y, self.fmts = None, None, fmts

#     def add(self, x, y):
#         # 向图表中添加多个数据点
#         if not hasattr(y, "__len__"):
#             y = [y]
#         n = len(y)
#         if not hasattr(x, "__len__"):
#             x = [x] * n
#         if not self.X:
#             self.X = [[] for _ in range(n)]
#         if not self.Y:
#             self.Y = [[] for _ in range(n)]
#         for i, (a, b) in enumerate(zip(x, y)):
#             if a is not None and b is not None:
#                 self.X[i].append(a)
#                 self.Y[i].append(b)
#         self.axes[0].cla()
#         for x, y, fmt in zip(self.X, self.Y, self.fmts):
#             self.axes[0].plot(x, y, fmt)
#         self.config_axes()
#         display.display(self.fig)
#         display.clear_output(wait=True)

# # 训练函数
# def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):  #@save
#     """训练模型（定义见第3章）"""
#     animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
#                         legend=['train loss', 'train acc', 'test acc'])
#     for epoch in range(num_epochs):
#         train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
#         test_acc = evaluate_accuracy(net, test_iter)
#         animator.add(epoch + 1, train_metrics + (test_acc,))
#     train_loss, train_acc = train_metrics
#     assert train_loss < 0.5, train_loss
#     assert train_acc <= 1 and train_acc > 0.7, train_acc
#     assert test_acc <= 1 and test_acc > 0.7, test_acc

# 训练
num_epochs = 10
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
plt.show()