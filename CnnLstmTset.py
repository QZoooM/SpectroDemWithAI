# CnnLstmTest.py
# 构建CNN+LSTM的模型

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
# CnnLstmFusion(以此作为类的名字)
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
        # 修改全连接层的输入大小
        self.fc = nn.Linear(lstm_hidden_size2, 1)

    def forward(self, x): # 重写forward方法
        out = x.permute(0, 2, 1) # 转置为CNN需要的格式
        for cnn in self.CNNs:
            out = cnn(out)
        out = out.permute(0, 2, 1) # 转置为LSTM需要的格式
        out, _ = self.lstm1(out)
        print(out.shape)
        out, _ = self.lstm2(out)
        print(out.shape)
        # out = out[:, -1, :] # 取最后一个时间步的输出
        out = self.fc(out)
        return out



if __name__ == "__main__":
    pass

    input_data = torch.randn(2, 508, 1)
    model = CnnLstmDem()
    output_data = model(input_data)
    print("CnnDemodulator输出形状: ", output_data.shape)
    # 现在的预期形状为[2, 1]
