# Task05 PyTorch进阶训练技巧


```python
import torch
import torch.nn as nn
import torch.nn.functional as F
```

## 1 自定义损失函数

- 以函数方式定义：通过输出值和目标值进行计算，返回损失值

- 以类方式定义：通过继承`nn.Module`，将其当做神经网络的一层来看待

以DiceLoss损失函数为例，定义如下：
$$
DSC = \frac{2|X∩Y|}{|X|+|Y|}
$$


```python
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss,self).__init__()
        
    def forward(self, inputs, targets, smooth=1):
        inputs = F.sigmoid(inputs)       
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()                   
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        return 1 - dice
```

## 2 动态调整学习率

- Scheduler：学习率衰减策略，解决学习率选择的问题，用于提高精度

- PyTorch Scheduler策略：
    - [lr_scheduler.LambdaLR](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.LambdaLR.html#torch.optim.lr_scheduler.LambdaLR)
    - [lr_scheduler.MultiplicativeLR](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.MultiplicativeLR.html#torch.optim.lr_scheduler.MultiplicativeLR)
    - [lr_scheduler.StepLR](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html#torch.optim.lr_scheduler.StepLR)
    - [lr_scheduler.MultiStepLR](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.MultiStepLR.html#torch.optim.lr_scheduler.MultiStepLR)
    - [lr_scheduler.ExponentialLR](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ExponentialLR.html#torch.optim.lr_scheduler.ExponentialLR)
    - [lr_scheduler.CosineAnnealingLR](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html#torch.optim.lr_scheduler.CosineAnnealingLR)
    - [lr_scheduler.ReduceLROnPlateau](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html#torch.optim.lr_scheduler.ReduceLROnPlateau)
    - [lr_scheduler.CyclicLR](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CyclicLR.html#torch.optim.lr_scheduler.CyclicLR)
    - [lr_scheduler.OneCycleLR](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html#torch.optim.lr_scheduler.OneCycleLR)
    - [lr_scheduler.CosineAnnealingWarmRestarts](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingWarmRestarts.html#torch.optim.lr_scheduler.CosineAnnealingWarmRestarts)

- 使用说明：需要将`scheduler.step()`放在`optimizer.step()`后面

- 自定义Scheduler：通过自定义函数对学习率进行修改

## 3 模型微调

- 概念：找到一个同类已训练好的模型，调整模型参数，使用数据进行训练。

- 模型微调的流程
  1. 在源数据集上预训练一个神经网络模型，即源模型
  2. 创建一个新的神经网络模型，即目标模型，该模型复制了源模型上除输出层外的所有模型设计和参数
  3. 给目标模型添加一个输出大小为目标数据集类别个数的输出层，并随机初始化改成的模型参数
  4. 使用目标数据集训练目标模型

![模型微调流程](images/ch05/01.png)

- 使用已有模型结构：通过传入`pretrained`参数，决定是否使用预训练好的权重

- 训练特定层：使用`requires_grad=False`冻结部分网络层，只计算新初始化的层的梯度


```python
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
```


```python
import torchvision.models as models
# 冻结参数的梯度
feature_extract = True
model = models.resnet50(pretrained=True)
set_parameter_requires_grad(model, feature_extract)
# 修改模型
num_ftrs = model.fc.in_features
model.fc = nn.Linear(in_features=512, out_features=4, bias=True)
```


```python
model.fc
```




    Linear(in_features=512, out_features=4, bias=True)



注：在训练过程中，model仍会回传梯度，但是参数更新只会发生在`fc`层。

## 4 半精度训练

- 半精度优势：减少显存占用，提高GPU同时加载的数据量

- 设置半精度训练：
  1. 导入`torch.cuda.amp`的`autocast`包
  2. 在模型定义中的`forward`函数上，设置`autocast`装饰器
  3. 在训练过程中，在数据输入模型之后，添加`with autocast()`

- 适用范围：适用于数据的size较大的数据集（比如3D图像、视频等）

## 5 总结

&emsp;&emsp;本次任务，主要介绍了PyTorch的进阶训练技巧，包括自定义损失函数、动态调整学习率、模型微调和半精度训练等技巧。
1. 自定义损失函数可以通过二种方式：函数方式和类方式，建议全程使用PyTorch提供的张量计算方法。
2. 通过使用PyTorch中的scheduler动态调整学习率，也支持自定义scheduler
3. 模型微调主要使用已有的预训练模型，调整其中的参数构建目标模型，在目标数据集上训练模型。
4. 半精度训练主要适用于数据的size较大的数据集（比如3D图像、视频等）。
