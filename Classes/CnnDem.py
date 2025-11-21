import torch
import torch.nn as nn

class SingleCnn(nn.Module):
    # 这两个channels参数会有变动，池填充需根据情况配置
    def __init__(self, in_channels=1, out_channels=16, pool_padding=0):
        super(SingleCnn, self).__init__()
        self.conv1d = nn.Conv1d(
            in_channels, # 输入通道
            out_channels, # 输出通道
            kernel_size=5, # 卷积核大小
            stride=1, # 步幅
            padding=2 # 使得输出长度仍为原来的序列长度
            )
        self.batchNormal = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.maxPool = nn.MaxPool1d(
            kernel_size=2, # 核大小
            stride=2, # 步幅
            padding=pool_padding
        )

    def forward(self, x):
        out = self.conv1d(x)
        out = self.batchNormal(out)
        out = self.relu(out)
        out = self.maxPool(out)
        return out
    
### 定义CnnDem
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
        self.CNNs = [ # 卷积网络组
            SingleCnn(in_ch, out_ch1), 
            SingleCnn(out_ch1, out_ch2), 
            SingleCnn(out_ch2, out_ch3, pool_padding=1), # 127 --> 64(not 63)
            SingleCnn(out_ch3, out_ch4), 
            SingleCnn(out_ch4, out_ch5), 
            SingleCnn(out_ch5, out_ch6)
        ]
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        self.fc = nn.Linear(out_ch6 * 8, 1)

    def forward(self, x):
        out = x.permute(0, 2, 1) # 数据转置，仅此一次
        for cnn in self.CNNs:
            out = cnn(out)
        out = self.flatten(out)
        out = self.fc(out)
        return out