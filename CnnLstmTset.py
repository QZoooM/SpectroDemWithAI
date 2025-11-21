# CnnLstmTest.py
# 构建CNN+LSTM的模型
"""
CNN+LSTM模型已完成测试
它已经融入Launch.py
可以在Launch.py中被引用
"""

print("Loading torch...")
import torch
import torch.nn as nn

# "*Test()" Functions
def Test():
    print("Start *Test:\n")

# 预制两个类
class SingleCnn(nn.Module): None
class CnnDem(nn.Module): None

CnnDemFile = open("./Classes/CnnDem.py",encoding="utf-8")
exec(CnnDemFile.read())
# 从过去的文件中获取两个类

# CNN+LSTM
# CnnLstmDem(以此作为类的名字)
"""
在之前CnnDem的基础上：继承CnnDem类并重写__init__和forward方法
添加两个LSTM层
LSTM1层的输入大小为CnnDem的输出通道数：out_ch6 = 64
LSTM2层的输入大小为LSTM1的隐藏层大小：lstm_hidden_size = 16
输出大小为1
注意：这里的LSTM层是为了提取时间序列特征
因此LSTM的输入数据格式为(batch, seq_len, feature_size)
而CnnDem的输出数据格式为(batch, feature_size, seq_len) 
需要转置
"""
class CnnLstmDem(CnnDem):
    def __init__(self,
            in_ch = 1, 
            out_ch1 = 16, 
            out_ch2 = 32, 
            out_ch3 = 32, 
            out_ch4 = 64, 
            out_ch5 = 64, 
            out_ch6 = 64,
            lstm_hidden_size1=16,
            lstm_hidden_size2=1
            ):
        super(CnnLstmDem, self).__init__(
            in_ch, out_ch1, out_ch2, out_ch3, out_ch4, out_ch5, out_ch6
        )
        # LSTM层
        self.lstm1 = nn.LSTM(
            input_size=out_ch6,
            hidden_size=lstm_hidden_size1,
            batch_first=True
        )
        self.lstm2 = nn.LSTM(
            input_size=lstm_hidden_size1,
            hidden_size=lstm_hidden_size2,
            batch_first=True
        )
        # 修改全连接层的输入大小，这时的序列还有8个时间步
        self.fc = nn.Linear(8, 1) # 缩为1个输出特征值

    def forward(self, x): # 重写forward方法
        out = x.permute(0, 2, 1) # 转置为CNN需要的格式
        for cnn in self.CNNs:
            out = cnn(out)
        out = out.permute(0, 2, 1) # 转置为LSTM需要的格式
        out, _ = self.lstm1(out)
        out, _ = self.lstm2(out)
        print(out.shape)

        # 下面一步：去掉最后一维
        # 去维很重要，因为LSTM的输出是三维的(batch, seq_len, hidden_size)
        # 而hidden_size此时为1，我们只需要(batch, seq_len)
        # 不然会导致后续全连接层维度不匹配
        out = out.squeeze(-1)
        # print(out.shape) # 查看LSTM输出形状以及数据内容是否丢失
        print(out) # 查看LSTM输出
        # out = out[:, -1, :] # 取最后一个时间步的输出，但这里用不到

        out = self.fc(out)
        return out

if __name__ == "__main__":
    pass

    input_data = torch.randn(2, 508, 1)
    model = CnnLstmDem()
    output_data = model(input_data)
    print("CnnDemodulator输出形状: ", output_data.shape)
    # 现在的预期形状为
    # >>> torch.Size([2, 1])
