# Launch.py
# 光谱分析
# 输入侦测光谱与参考光谱，或游标包络化光谱
# 进行数据处理(LSTM + CNN)
# 输出压力值预测结果
# 以及相关中间数据可视化
Hint = """
小暗示：
这个python脚本保持精简，只对必要的参数做批注
想了解更某个参数的详情，请移步至对应的 *Test.py 文件
那里有超级详细的解释
"""

from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
# 导入Torch主库
print("Loading Torch...")
import torch
# 导入数据加载器
from torch.utils.data import DataLoader
# 导入nn及优化器
import torch.nn.functional as F
import torch.optim as optim
from torch import nn

# Functions
def Func000():
    #Tensor的加法与原地加法
    x=torch.tensor([1,2])
    y=torch.tensor([3,4])
    z=x.add(y)
    print(z)
    print(x)
    x.add_(y)
    print(x)

def Func001():
    #根据list数据生成Tensor
    torch.Tensor([1,2,3,4,5,6])
    #根据指定形状生成Tensor
    torch.Tensor(2,3)
    #根据给定的Tensor的形状
    t=torch.Tensor([[1,2,3],[4,5,6]])
    #查看Tensor的形状
    t.size()
    #shape与size()等价方式
    t.shape
    #根据已有形状创建Tensor
    torch.Tensor(t.size())

# Classes
## 定义FSR点与FSR数组类
class FSR_Point():
    point = [0, 0] # 二维点数据
    isWrong = False # 是否为错误点
    otherArg = None # 其他属性
    def Reset(self):
        self.pointArr = [0, 0]
        self.isWrong = False
    def Set(self, x, y):
        self.pointArr = [x, y]

class FSR_Arr():
    arr = [] # 存储
    length = 0 # 长度
    otherArg = None # 其他属性
    def __init__(self):
        for _ in range(512):
            self.arr.append(FSR_Point())
        length = 512
    def Reset(self):
        for i in self.arr:
            i.Reset()
        length = 512
    def SetArr(self, arr):
        self.arr = arr
        self.length = len(arr)
    def SetPoint(self, index, FSRpoint):
        if (self.length <= index):
            for i in range(index - self.length + 1):
                self.arr.append(FSR_Point())
        self.length = index + 1
        self.arr[index] = FSRpoint

## 定义四层LSTM网络
"""
(batch_size, seq_length, dimension) --输入--> 
1维  --1层LSTM--> 
64维 --2层LSTM--> 
64维 --3层LSTM--> 
32维 --4层LSTM--> 
4维  --扁平层--> 
(batch_size, seq_length * 4) --全连接层-->
(batch_size, 1)
这个不是压力值，而是经过LSTM网络提取的特征值
真正的压力值需要与特征值关联的回归模型来预测
"""
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
        return out # 输出特征值
        # 必须说明一下，这个out不是压力值，而是经过LSTM网络提取的特征值

## CNN相关类
### 定义单独CNN
"""
构成：
卷积层(convolution layer)
批归一层(batch normalization layer)
激活函数(+ReLU)
池化层(max-pooling layer)
"""
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
"""
6个单独CNN(SingleCnn)
--其中第3个要配置pool_padding=1--
扁平层(nn.Flatten)
全连接层(nn.Linear)
"""
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

### 定义简单CNN
"""
这个CNN只是摆设
"""
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = torch.nn.Sequential(
            OrderedDict(
                [
                    ("conv1", torch.nn.Conv2d(3, 32, 3, 1, 1)),
                    ("relu1", torch.nn.ReLU()),
                    ("pool", torch.nn.MaxPool2d(2))
                ]
            ))

        self.dense = torch.nn.Sequential(
            OrderedDict([
                ("dense1", torch.nn.Linear(32 * 3 * 3, 128)),
                ("relu2", torch.nn.ReLU()),
                ("dense2", torch.nn.Linear(128, 10))
            ])
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 32 * 3 * 3)
        x = self.dense(x)
        return x

## nn网络集合
class NetCollection():
    lstmNet = None
    cnnDem = None


# PS: 以下的部分页面类代码摘自过去的项目
## 定义操作页面类
"""
简易的控制台页面以及页面管理器
包含初始化页面、数据加载页面、数据处理页面
那些复杂的GUI页面留待后续版本实现
模型的初始化与数据处理均在各自页面中进行
模型不会存储在页面类中
而是在NetCollection类中统一管理
"""
### 页面管理器
class PageManager():
    def __init__(self):
        self.curPageNumber = 0 # Current page
        self.total_pages = 0 # Total number of pages
        self.pages = [] # Pages list
        self.lastPageNumber = None # Last page number, None means no last page
    
    def goto_page(self, page_number):
        self.curPageNumber = page_number

    def exchage_page(self):
        if self.lastPageNumber is not None:
            temp = self.curPageNumber
            self.curPageNumber = self.lastPageNumber
            self.lastPageNumber = temp

### 单个页面
class Page():
    def __init__(self, page_number, page_manager: PageManager, net_collection: NetCollection=None):
        self.inquireString = "[?]: "
        self.titleString = "Page " + str(page_number)
        self.descriptionString = "This is page number " + str(page_number)

        self.function_dict = {} # Function dictionary
        self.pageNumber_dict = {} # Page dictionary
        self.choicesString = "" # Choices list string

        self.net_collection = net_collection # Net collection
        # Register page in manager
        self.page_number = page_number
        self.page_manager = page_manager
        self.page_manager.total_pages += 1
        self.page_manager.pages.append(self)
        
    def display(self):
        print(self.titleString)
        print(self.choicesString)
        inp = input(self.inquireString)
        if inp in self.pageNumber_dict:
            self.page_manager.goto_page(self.pageNumber_dict[inp])
        else:
            print("Invalid choice. Please try again.")

#### 主页面
class MainPage(Page):
    def __init__(self, page_number, page_manager: PageManager, net_collection: NetCollection=None):
        super().__init__(page_number, page_manager, net_collection)
        self.titleString = "Main Page"
        self.descriptionString = "This is the main page, which to select a model to run."
        self.choicesString = "[1] LSTM Page\n[2] CNN Page\n[0] Exit"
        self.pageNumber_dict = {
            "2": 2,
            "1": 1,
            "0": -1
        }

    def display(self):
        print(self.titleString)
        print(self.choicesString)
        inp = input(self.inquireString)
        if inp in self.pageNumber_dict:
            self.page_manager.goto_page(self.pageNumber_dict[inp])
        else:
            print("Invalid choice. Please try again.")

#### LSTM页面
class LSTMPage(Page):
    def __init__(self, page_number, page_manager: PageManager, net_collection: NetCollection=None):
        super().__init__(page_number, page_manager, net_collection)
        self.titleString = "LSTM Processing Page"
        self.descriptionString = "This page processes data using the LSTM model."
        self.choicesString = "[1] Init LSTM Net\n[2] Run random data into LSTM Net\n[0] Back to Main Page"

        self.function_dict = {
            "1": "Init LSTM Net",
            "2": "Run random data into LSTM Net"
        }

        self.pageNumber_dict = {
            "0": 0
        }

    def display(self):
        print(self.titleString)
        print(self.choicesString)
        inp = input(self.inquireString)
        if inp in self.pageNumber_dict:
            self.page_manager.goto_page(self.pageNumber_dict[inp])
        elif inp in self.function_dict:
            if inp == "1":
                self.init_lstm_net()
            elif inp == "2":
                if self.net_collection.lstmNet is None:
                    print("LSTM Net is not initialized. Please initialize it first.")
                else:
                    self.run_random_data()
        else:
            print("Invalid choice. Please try again.")

    def init_lstm_net(self):
        self.net_collection.lstmNet = LSTMNet()
        print("LSTM Net initialized.")

    def run_random_data(self):
        test_input = torch.randn(2, 508, 1)  # batch_size=2, seq_length=508, input_size=1
        test_output = self.net_collection.lstmNet(test_input)
        print("Test input shape:", test_input.shape)
        # print(test_input)
        print("Test output shape:", test_output.shape)
        # print(test_output)
        print("LSTM Net ran successfully.")
        input("Press Enter to continue...")

#### CNN页面
class CNNPage(Page):
    def __init__(self, page_number, page_manager: PageManager, net_collection: NetCollection=None):
        super().__init__(page_number, page_manager, net_collection)
        self.titleString = "CNN Processing Page"
        self.descriptionString = "This page processes data using the CNN model."
        self.choicesString = "[1] Init CNN Net\n[2] Run random data into CNN Net\n[0] Back to Main Page"

        self.function_dict = {
            "1": "Init CNN Net",
            "2": "Run random data into CNN Net"
        }

        self.pageNumber_dict = {
            "0": 0
        }

    def display(self):
        print(self.titleString)
        print(self.choicesString)
        inp = input(self.inquireString)
        if inp in self.pageNumber_dict:
            self.page_manager.goto_page(self.pageNumber_dict[inp])
        elif inp in self.function_dict:
            if inp == "1":
                self.init_cnn_net()
            elif inp == "2":
                if self.net_collection.cnnDem is None:
                    print("CNN Net is not initialized. Please initialize it first.")
                else:
                    self.run_random_data()
        else:
            print("Invalid choice. Please try again.")

    def init_cnn_net(self):
        self.net_collection.cnnDem = CnnDem() # CNN集合解调器
        print("CNN Net initialized.")

    def run_random_data(self):
        input_data = torch.randn(2, 508, 1)
        output_data = self.net_collection.cnnDem(input_data)
        print("输入形状: ", input_data.shape)
        print("CnnDemodulator输出形状: ", output_data.shape)
        # 预期形状为[2, 1]
        print("CNN Net ran successfully.")
        input("Press Enter to continue...")

# 主程序入口
if __name__ == "__main__":
    # 启动时的语句
    print(Hint)
    # 创建网络集合
    net_collection = NetCollection()
    # 初始化页面管理器
    page_manager = PageManager()
    # 创建页面实例
    main_page = MainPage(0, page_manager, net_collection)
    lstm_page = LSTMPage(1, page_manager, net_collection)
    cnn_page = CNNPage(2, page_manager, net_collection)
    # 主循环
    while True:
        current_page = page_manager.pages[page_manager.curPageNumber]
        current_page.display()
        if page_manager.curPageNumber == -1:
            print("Exiting program.")
            break


# 就像……写UNITY脚本一样(╰(*°▽°*)╯)
# 以后可以考虑把页面类改成用Tkinter写的GUI界面
# 现在先用控制台界面实现基本功能再说