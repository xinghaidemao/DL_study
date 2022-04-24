import torch
import torchvision
import matplotlib.pyplot as plt
from torch import nn
from d2l import torch as d2l
from torch.utils import data
from torchvision import transforms
# 模型
# 与softmax回归的简洁实现（ :numref:sec_softmax_concise）相比， 
# 唯一的区别是我们添加了2个全连接层（之前我们只添加了1个全连接层）。 
# 第一层是[隐藏层]，它(包含256个隐藏单元，并使用了ReLU激活函数)。 第二层是输出层。

net = nn.Sequential(nn.Flatten(),
                    nn.Linear(784, 256),
                    nn.ReLU(),
                    nn.Linear(256, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights)

# [训练过程]的实现与我们实现softmax回归时完全相同， 这种模块化设计使我们能够将与和模型架构有关的内容独立出来。
batch_size, lr, num_epochs = 256, 0.1, 10
loss = nn.CrossEntropyLoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(), lr=lr)

# train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
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


d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
plt.show()