
import torch


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
