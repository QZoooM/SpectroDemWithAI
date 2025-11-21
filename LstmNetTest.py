# 现在这些代码已经融入Launch.py中，此文件保留以备参考
# 可以在Launch.py中的函数中调用此LSTM网络类
# LstmNetTest.py

print("loading torch...")
import torch
import torch.nn as nn

"""
定义一个4层LSTM神经网络
输入1维数据，由第一层LSTM处理成64维后传递给第二层LSTM
第二层LSTM处理成64维后传递给第三层LSTM
第三层LSTM处理成32维后传递给第四层LSTM
第四层LSTM处理成4维后传递给扁平层
扁平层处理成(4 * batch_size * seq_length)维后传递给全连接层
全连接层输出1个特征值
这个不是压力值，而是经过LSTM网络提取的特征值
真正的压力值需要与特征值关联的回归模型来预测
"""

# 定义LSTM网络
class LSTMNet(nn.Module):
    def __init__(self,
                 input_size=1,
                 hidden_size1=64,
                 hidden_size2=64,
                 hidden_size3=32,
                 hidden_size4=4,
                 seq_size=508,  # 假设序列长度为508，这其实是原文的输入长度
                 output_size=1):
        super(LSTMNet, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size1, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size1, hidden_size2, batch_first=True)
        self.lstm3 = nn.LSTM(hidden_size2, hidden_size3, batch_first=True)
        self.lstm4 = nn.LSTM(hidden_size3, hidden_size4, batch_first=True)
        self.flatten = nn.Flatten() # 等价为nn.Flatten(start_dim=1，end_dim=-1)
        self.fc = nn.Linear(seq_size * hidden_size4, output_size)

    # nn.Module要求必须实现forward方法
    def forward(self, x):
        out, _ = self.lstm1(x)
        out, _ = self.lstm2(out)
        out, _ = self.lstm3(out)
        out, _ = self.lstm4(out)

        # 使用nn.Flatten替代view：展平为(batch_size, 508*4) = (batch_size, 2032)
        # out = out.view(out.size(0), -1)  # 扁平化
        out = self.flatten(out)

        out = self.fc(out)
        return out
    
# 测试随机输入
if __name__ == "__main__":
    model = LSTMNet()
    """
    我们将从一个随机生成的张量测试网络的前向传播
    输入形状 (2, 508, 1) 
    可以拆解为：
    第 1 维 2: 2 个样本 (2 条不同的光谱）；
    第 2 维 508: 每个样本包含 508 个数据点 (序列长度）；
    第 3 维 1: 每个数据点的维度 (1 维）。
    """
    test_input = torch.randn(2, 508, 1)  # (batch_size=2, seq_length=508, input_size=1)
    test_output = model(test_input)
    print(test_output.shape)