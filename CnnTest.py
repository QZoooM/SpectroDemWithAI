# 那个 CnnEpochTest.py 是2D图像识别的训练脚本
# 这个 CnnTest.py 才是光谱分析用的CNN网络脚本
"""
CnnTest.py
用CNN解调光谱
这个脚本将会重满各种批注
Luanch.py 里面就不会写那么多了
构造过程中遇见一些问题
但已被完美解决
SingleCnn 与 CnnDem 已经融入 Launch.py
其他测试函数 *Test() 不会被融入
现在 Launch.py 中可以找的这两个类了
"""
print("Loading Torch...")
import torch
# 导入数据加载器
# from torch.utils.data import DataLoader
# 导入nn及优化器
# import torch.nn.functional as F
# import torch.optim as optim
from torch import nn

# Test Function
def filterTest():
    tens = torch.tensor([[1,2,3,5],[5,2,7,8],[45,54,62,11],[11,56,20,8]])
    mask = tens > 20

    filtered_tens = tens[mask]
    print("掩码过滤: ")
    print(filtered_tens)
    # 过滤后会输出 >>> tensor([45, 54, 62, 56])，扁了

    tensor = torch.tensor([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12], [13, 14, 15, 16, 17, 18]])
    # 使用整数索引过滤数据
    filtered_tensor = tensor[torch.tensor([0, 2])]
    print("引索过滤(选择): ")
    print(filtered_tensor)
    print("----")
    # 过滤(选择)后会输出 >>> tensor([[ 1,  2,  3,  4,  5,  6], [13, 14, 15, 16, 17, 18]])
    # 选择数字也是一样的，只不过这里过滤的是内部的张量

def simpCnnTest():
    m = nn.Conv1d(16, 33, 3, stride=2) # 将一组16个样本投影成33个？
    input = torch.randn(20, 16, 50) # 20个样本组，一组16个样本，样本维度为50
    output = m(input)
    print("输出张量尺寸：")
    print(output.shape)
    ## 预期输出 >>> torch.Size([20, 33, 24])

def simpCnnTest1():
    m = nn.Conv1d(1, 16, 5, stride=1, padding=0) # 通道数不发生变化
    input = torch.randn(2, 1, 508) # 2个样本组，一组508个采样，样本维度为1
    output = m(input)
    print("输出张量尺寸：")
    print(output.shape)

def tensorDimTest():
    # 理解张量构成
    tens = torch.randn(1, 2, 3) # 从外到内设定对象数量
    print(tens.shape)
    print(tens)

# 构建一个多层六循环CNN
## SingleCnn --> SingleCnn (单个CNN)
## CnnDemodulator --> CnnDem (CNN集和解调器，由6个SingleCnn组成)
"""
有六个卷积区块
每个卷积区块包含以下部分：
[卷积层(convolution layer)
--原文中的filter指的是conv layer的kernel卷积核--
--卷积步幅(原文指moving step)stride=1--
批归一层(batch normalization layer)
--为了避免过拟合而存在--
--同时增大训练速度--
激活函数(+ReLU)
池化层(max-pooling layer)]
每个卷积区块依次排列
"""
class SingleCnn(nn.Module):
    # 这两个channels参数会有变动
    def __init__(self, in_channels=1, out_channels=16, pool_padding=0):
        super(SingleCnn, self).__init__()
        self.conv1d = nn.Conv1d(
            in_channels, # 输入通道(输入维度)
            out_channels, # 输出通道(输出维度)
            kernel_size=5, # 卷积核大小，原文为5
            stride=1, # 步幅，走一步处理一次
            padding=2 # 填充数量，卷积核为5，在input数据两侧各添加2个"0，使得输出长度仍为508(原来的序列长度)
            )
        self.batchNormal = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.maxPool = nn.MaxPool1d(
            kernel_size=2, # 核大小
            stride=2, # 步幅
            padding=pool_padding
            # 这样设计可以实现原文中(508,16)向(254,16)的池化变换
            # 在后续的池化变换中，(B, L, C)中的C维度大小都除以二
            #### 因此，nn.MaxPool1d的参数无需改变
            # 上一句话需要修正，池化层无法池化不满足核大小的序列
            # 就会出现：127池化成63个，单独的那个被丢失了
            # 池化层的padding需要能够被修改
        )

    def forward(self, x):
        # 输入x形状: (batch_size, 508, 1)
        # 卷积层需要 (batch_size, in_channels, seq_len)，所以先转置
        # 把x的位置序列调整为(0, 2, 1)，就是把后面两个换位
        # out = x.permute(0, 2, 1)  # (batch_size, 1, 508)
        # 单个CNN网络就不做转置
        # 最开始的输入转置交给 CnnDem (CNN集和解调器)
        
        # 卷积层输出: (batch_size, 16, 508)
        out = self.conv1d(x)
        
        # 传回LSTM需要的数据形状: (batch_size, 508, 16)
        # out = out.permute(0, 2, 1)
        # 不过，这个测试文件专属于CNN，不需要转置成LSTM的格式

        # 批归一化层需要的数据形状与卷积层输出的相同
        # 无需转置
        out = self.batchNormal(out)
        out = self.relu(out)
        
        # 池化层
        # 无需转置
        out = self.maxPool(out)
        
        return out

"""
维度形状变化：
siz (  B,   L,   C)
-------------------
inp (  2, 508,   1)
--> (  2, 254,  16)
--> (  2, 127,  32)
--> (  2,  64,  32)
--> (  2,  32,  64)
--> (  2,  16,  64)
--> (  2,   8,  64)
out
B: 自己定;
L: 每次由池化层减半;
C: 交给卷积层增加通道
"""
## CNN集合(CNN解调器)
class CnnDem(nn.Module):
    def __init__(
            self, 
            in_ch = 1, 
            out_ch1 = 16, 
            out_ch2 = 32, 
            out_ch3 = 32, 
            out_ch4 = 64, 
            out_ch5 = 64, 
            out_ch6 = 64, 
            ):
        super(CnnDem, self).__init__()
        # 卷积网络组
        self.CNNs = [
            SingleCnn(in_ch, out_ch1), 
            SingleCnn(out_ch1, out_ch2), 
            SingleCnn(out_ch2, out_ch3, pool_padding=1), # 127 --> 64(not 63)
            SingleCnn(out_ch3, out_ch4), 
            SingleCnn(out_ch4, out_ch5), 
            SingleCnn(out_ch5, out_ch6)
        ]
        # 扁平层
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        # 全连接层
        self.fc = nn.Linear(out_ch6 * 8, 1)

    def forward(self, x):
        # 对输入的数据转置
        out = x.permute(0, 2, 1)

        for cnn in self.CNNs:
            # 直觉告诉我会有一些问题发生
            out = cnn(out)
            # 观察一下CNN输出的数据形状
            # print(out.shape)

        """
        第一次输出：
        >>> torch.Size([2, 16, 254])
        >>> torch.Size([2, 32, 127])
        >>> torch.Size([2, 32, 63])
        >>> torch.Size([2, 64, 31])
        >>> torch.Size([2, 64, 15])
        >>> torch.Size([2, 64, 7])
        在输出数据的序列L维度为奇数时
        下一次池化层需要设置peding=1
        --1.这个可以写成自动判定
        --2.也可以就这个项目进行有限的设置
        ----这里选择2有限的设置
        ----这个padding设置加在第2个CNN上
        预期输出：
        >>> torch.Size([2, 16, 254])
        >>> torch.Size([2, 32, 127])
        >>> torch.Size([2, 32, 64])
        >>> torch.Size([2, 64, 32])
        >>> torch.Size([2, 64, 16])
        >>> torch.Size([2, 64, 8])
        新增padding参数后达到与其输出
        """
        out = self.flatten(out)
        out = self.fc(out)

        # 注意：这个 out 不是压力值，而是特征值，真正的压力值需要配置回归模型
        return out


if __name__ == "__main__":
    pass
    ## Test Functions
    # filterTest()
    # simpCnnTest()
    # simpCnnTest1()
    # tensorDimTest()

    input_data = torch.randn(2, 508, 1)
    model = CnnDem()
    output_data = model(input_data)
    print("CnnDemodulator输出形状: ", output_data.shape)
    # 现在的预期形状为[2, 1]