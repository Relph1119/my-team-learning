# Task07 PyTorch生态简介

## 1 torchvision（图像）

- torchvision.datasets：计算机视觉领域常见的数据集，包括CIFAR、EMNIST、Fashion-MNIST等

- torchvision.transforms：数据预处理方法，可以进行图片数据的放大、缩小、水平或垂直翻转等

- torchvision.models：预训练模型，包括图像分类、语义分割、物体检测、实例分割、人体关键点检测、视频分类等模型

- torchvision.io：视频、图片和文件的IO操作，包括读取、写入、编解码处理等

- torchvision.ops：计算机视觉的特定操作，包括但不仅限于NMS，RoIAlign（MASK R-CNN中应用的一种方法），RoIPool（Fast R-CNN中用到的一种方法）

- torchvision.utils：图片拼接、可视化检测和分割等操作

## 2 PyTorchVideo（视频）

- 简介：PyTorchVideo是一个专注于视频理解工作的深度学习库，提供加速视频理解研究所需的可重用、模块化和高效的组件，使用PyTorch开发，支持不同的深度学习视频组件，如视频模型、视频数据集和视频特定转换。

- 特点：基于PyTorch，提供Model Zoo，支持数据预处理和常见数据，采用模块化设计，支持多模态，优化移动端部署

- 使用方式：[TochHub](https://pytorchvideo.org/docs/tutorial_torchhub_inference)、[PySlowFast](https://github.com/facebookresearch/SlowFast/)、[PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning)

## 3 torchtext（文本）

- 简介：torchtext是PyTorch的自然语言处理（NLP）的工具包，可对文本进行预处理，例如截断补长、构建词表等操作

- 构建数据集：使用`Field`类定义不同类型的数据

- 评测指标：使用`torchtext.data.metrics`下的方法，对NLP任务进行评测

## 4 总结

&emsp;&emsp;本次任务，主要介绍了PyTorch生态在图像、视频、文本等领域中的发展，并介绍了相关工具包的使用。
1. 图像：torchvision主要提供在计算机视觉中常常用到的数据集、模型和图像处理操作。
2. 视频：PyTorchVideo主要基于PyTorch，提供Model Zoo，支持数据预处理和常见数据，采用模块化设计，支持多模态，优化移动端部署。
3. 文本：torchtext是PyTorch的自然语言处理（NLP）的工具包，可对文本进行预处理，例如截断补长、构建词表等操作。
