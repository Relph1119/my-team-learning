#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: fashion_mnist_clf.py
@time: 2022/3/9 9:24
@project: my-team-learning
@desc: 
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from torchvision import datasets

# 使用GPU环境
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# 配置其他超参数
batch_size = 256
# 在Windows环境下，需要将num_workers改为0，否则会存在多线程问题
num_workers = 0
lr = 1e-4
epochs = 20

# 设置数据变换
image_size = 28
data_transform = transforms.Compose([
    transforms.ToPILImage(),  # 这一步取决于后续的数据读取方式，如果使用内置数据集则不需要
    transforms.Resize(image_size),
    transforms.ToTensor()
])


class FMDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        self.images = df.iloc[:, 1:].values.astype(np.uint8)
        self.labels = df.iloc[:, 0].values

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].reshape(28, 28, 1)
        label = int(self.labels[idx])
        if self.transform is not None:
            image = self.transform(image)
        else:
            image = torch.tensor(image / 255., dtype=torch.float)
        label = torch.tensor(label, dtype=torch.long)
        return image, label

train_df = pd.read_csv("../my_homework/FashionMNIST/fashion-mnist_train.csv")
test_df = pd.read_csv("../my_homework/FashionMNIST/fashion-mnist_test.csv")
train_data = FMDataset(train_df, data_transform)
test_data = FMDataset(test_df, data_transform)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

image, label = next(iter(train_loader))
print(image.shape, label.shape)
plt.imshow(image[0][0], cmap="gray")
