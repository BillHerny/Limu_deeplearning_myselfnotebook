
import torch
import torch
from d2l import torch as d2l
from IPython import display

class Accumulator:
    """在n个变量上累加"""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
      
# def accuracy(y_hat, y):
#     """统计预测正确的样本数量。"""
#     if y_hat.ndim > 1 and y_hat.shape[1] > 1:
#         y_hat = y_hat.argmax(dim=1)

#     comparison = y_hat.to(y.dtype) == y
#     return float(comparison.to(torch.float32).sum())


def accuracy(y_hat, y):
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.size(1) > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

# def evaluate_accuracy(net, data_iter):
#     """计算模型在数据集上的准确率。"""
#     if isinstance(net, torch.nn.Module):
#         net.eval()

#     correct = 0.0
#     total = 0

#     with torch.no_grad():
#         for X, y in data_iter:
#             y_hat = net(X)
#             correct += accuracy(y_hat, y)
#             total += y.numel()

#     return correct / total
def evaluate_accuracy(net, data_iter):  #@save
    """计算在指定数据集上模型的精度"""
    if isinstance(net, torch.nn.Module):  #判断传入的 net 是否是 PyTorch 标准模型
        net.eval()  # 将模型设置为评估模式
        #评估模式，作用：
        # 关闭 Dropout、BatchNorm 的随机训练行为；
        # 保证测试时结果稳定，不引入随机噪声。
    metric = Accumulator(2)  # 正确预测数、预测总数 
                    #Accumulator(n)：自定义累加工具，创建长度为n的数组，持续累加数值。
    with torch.no_grad():
        #评估时不需要反向传播、不需要更新权重，关闭梯度可以：
        # 大幅节省显存、内存；
        # 提升推理计算速度。
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
            #net(X)：模型前向推理，输入批次数据 X，输出预测结果y_hat（二维概率矩阵）
            #accuracy(net(X), y)：调用你上一段的函数，返回当前批次预测正确的样本个数
            #y.numel()：numel() 是张量方法，返回张量总元素数量，即当前批次总样本数
            #metric.add(a,b)：累加器分别累加两个值：
            # metric[0] += a （正确样本数累加）
            # metric[1] += b （总样本数累加）
    return metric[0] / metric[1]

class Animator:
    """在动画中绘制数据"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)
def train_epoch_ch3(net, train_iter, loss, updater):
    """训练模型一个迭代周期（定义见第3章）"""
    if isinstance(net, torch.nn.Module):
        net.train()
    metric = Accumulator(3)
    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    return metric[0] / metric[2], metric[1] / metric[2]

#@save
def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    """训练模型（定义见第3章）"""
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics
