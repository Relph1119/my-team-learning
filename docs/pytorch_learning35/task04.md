# Task04 PyTorch模型定义


```python
import os
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
```

## 1 模型定义的方式

- Sequential：`nn.Sequential`，可接收一个子模块的有序字典(OrderedDict)或者一系列子模块作为参数来逐一添加Module的实例


```python
# 采用直接排列方式
net = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10), 
)
print(net)
```

    Sequential(
      (0): Linear(in_features=784, out_features=256, bias=True)
      (1): ReLU()
      (2): Linear(in_features=256, out_features=10, bias=True)
    )
    


```python
# 采用OrderedDict方式
net2 = nn.Sequential(collections.OrderedDict([
    ('fc1', nn.Linear(784, 256)),
    ('relu1', nn.ReLU()),
    ('fc2', nn.Linear(256, 10))
]))
print(net2)
```

    Sequential(
      (fc1): Linear(in_features=784, out_features=256, bias=True)
      (relu1): ReLU()
      (fc2): Linear(in_features=256, out_features=10, bias=True)
    )
    

- ModuleList：`nn.MoudleList`，接收一个子模块（或层，需属于nn.Module类）的列表


```python
net = nn.ModuleList([
    nn.Linear(784, 256), 
    nn.ReLU()])

# 类似List的append操作
net.append(nn.Linear(256, 10))
# 类似List的索引访问
print("最后一个层:\n", net[-1])  
print("整个网络层:\n", net)
```

    最后一个层:
     Linear(in_features=256, out_features=10, bias=True)
    整个网络层:
     ModuleList(
      (0): Linear(in_features=784, out_features=256, bias=True)
      (1): ReLU()
      (2): Linear(in_features=256, out_features=10, bias=True)
    )
    

- ModuleDict：`nn.ModuleDict`，和ModuleList类似


```python
net = nn.ModuleDict({
    'linear': nn.Linear(784, 256),
    'act': nn.ReLU(),
})
# 添加模型的层
net['output'] = nn.Linear(256, 10)
# 访问linear层
print("linear层:\n", net['linear'])
print("ouput层:\n", net.output)
print("整个模型网络层:\n", net)
```

    linear层:
     Linear(in_features=784, out_features=256, bias=True)
    ouput层:
     Linear(in_features=256, out_features=10, bias=True)
    整个模型网络层:
     ModuleDict(
      (linear): Linear(in_features=784, out_features=256, bias=True)
      (act): ReLU()
      (output): Linear(in_features=256, out_features=10, bias=True)
    )
    

- 比较与适用场景
  1. Sequential适合快速验证结果, 不需要同时写\_\_init\_\_和forward
  2. ModuleList和ModuleDict适用于复用

## 2 搭建模型

模型搭建基本方法：
1. 模型块分析
2. 模型块实现
3. 利用模型块组装模型

以U-Net模型为例，该模型为分割模型，通过残差连接结构解决了模型学习中的退化问题，使得神经网络的深度能够不断扩展。

![U-Net](images/ch04/01.png)

### 2.1 模型块分析

1. 每个子块内部的两次卷积`DoubleConv`
2. 左侧模型块之间的下采样连接`Down`，通过Max pooling来实现
3. 右侧模型块之间的上采样连接`Up`
4. 输出层的处理`OutConv`
5. 模型块之间的横向连接，输入和U-Net底部的连接等计算，这些单独的操作可以通过forward函数来实现

### 2.2 U-Net模型块实现


```python
# 两次卷积 conv 3x3, ReLU
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
```


```python
# 下采样 max pool 2x2
class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)
```


```python
# 上采样 up-conv 2x2
class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
```


```python
# 输出 conv 1x1
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
```

### 2.3 利用模型快组装U-Net


```python
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
    
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
```

## 3 修改模型


```python
import torchvision.models as models
net = models.resnet50()
```

- 修改模型层：观察模型层，根据需求定义模型层，然后在对应的模型层上赋值


```python
from collections import OrderedDict

# 以10分类任务为例，根据需求定义模型层
classifier = nn.Sequential(
    OrderedDict([
        ('fc1', nn.Linear(2048, 128)),
        ('relu1', nn.ReLU()), 
        ('dropout1',nn.Dropout(0.5)),
        ('fc2', nn.Linear(128, 10)),
        ('output', nn.Softmax(dim=1))
]))

# 修改模型层
net.fc = classifier
```


```python
net.fc
```




    Sequential(
      (fc1): Linear(in_features=2048, out_features=128, bias=True)
      (relu1): ReLU()
      (dropout1): Dropout(p=0.5, inplace=False)
      (fc2): Linear(in_features=128, out_features=10, bias=True)
      (output): Softmax(dim=1)
    )



- 添加外部输入：将原模型添加外部输入位置前的部分作为一个整体，同时在`forward`中定义好原模型不变的部分、添加的输入和后续层之间的连接关系


```python
class Model(nn.Module):
    def __init__(self, net):
        super(Model, self).__init__()
        self.net = net
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc_add = nn.Linear(1001, 10, bias=True)
        self.output = nn.Softmax(dim=1)
        
    def forward(self, x, add_variable):
        x = self.net(x)
        # 增加一个额外的输入变量add_variable，辅助预测
        x = torch.cat((self.dropout(self.relu(x)), add_variable.unsqueeze(1)),1)
        x = self.fc_add(x)
        x = self.output(x)
        return x
```

- 添加额外输出：修改模型定义中的`forward`函数的`return`返回


```python
class Model(nn.Module):
    def __init__(self, net):
        super(Model, self).__init__()
        self.net = net
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(1000, 10, bias=True)
        self.output = nn.Softmax(dim=1)
        
    def forward(self, x, add_variable):
        x1000 = self.net(x)
        x10 = self.dropout(self.relu(x1000))
        x10 = self.fc1(x10)
        x10 = self.output(x10)
        # 添加额外输出
        return x10, x1000
```

## 4 模型保存与读取

- 模型存储格式：pkl、pt、pth
- 模型存储内容：存储整个模型（模型结构和权重）`model`、只存储模型权重`model.state_dict`
- 多卡模型存储：`torch.nn.DataParallel(model).cuda()`

以resnet50模型的单卡保存和单卡加载为例


```python
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
model = models.resnet50(pretrained=True)
```


```python
save_dir = './models/resnet50.pkl'

# 保存整个模型
torch.save(model, save_dir)
# 读取整个模型
loaded_model = torch.load(save_dir)
```


```python
save_dir = './models/resnet50_state_dict.pkl'

# 保存模型结构
torch.save(model.state_dict(), save_dir)
# 读取模型结构
loaded_dict = torch.load(save_dir)
loaded_model = models.resnet50()
# 定义模型结构
# loaded_model.load_state_dict(loaded_dict)
loaded_model.state_dict = loaded_dict
```

注：多卡模式下建议适用权重的方式存储和读取模型

## 5 总结

&emsp;&emsp;本次任务，主要介绍了PyTorch模型定义方式、利用模型块快速搭建复杂模型、修改模型、保存和读取模型。
1. PyTorch模型主要有三种定义方式，分别是`Sequential`、`ModuleList`和`ModuleDict`。
2. 对于大型复杂的网络，通过构建模型块，利用`forward`连接模型，从而可以快速搭建。
3. 根据自身需求对已有模型的修改，可有三种方式：修改模型层、添加额外输入、添加额外输出。
4. 利用模型保存和读取函数，可以在单卡和多卡的环境上，存储整个模型，或只存储模型权重。
