# 第51期《扩散模型从原理到实战》学习笔记

本次任务需要提前配置代理
## 设置Jupyter Notebook代理
```shell
set HTTPS_PROXY=http://127.0.0.1:19180
set HTTP_PROXY=http://127.0.0.1:19180
```
设置代理之后，启动Jupyter Notebook
```shell
jupyter notebook
```

## Task01 扩散模型库入门与零基础实战

本次任务，主要了解了扩散模型的基本原理（前向过程、反向过程、优化目标），并学习了Hugging Face核心功能的使用，根据教程从零搭建了扩散模型，并将BasicUNet和UNet2Model两个模型预测结果进行对比。

个人笔记如下： 
- [第1章 扩散模型简介](diffusion_models_learning51/ch01.md) 
- [第2章 Hugging Face简介](diffusion_models_learning51/ch02.md)
- [第3章 从零开始搭建扩散模型](diffusion_models_learning51/ch03/ch03.md)

## Task02 微调与引导

本次任务，主要基于Diffusers库，了解Diffusers核心API（管线、模型和调度器），通过蝴蝶图像生成的代码实战，学习扩散模型从定义、训练到图像生成一系列的实战内容；并了解微调和引导这两个技术，通过CLIP引导可以用文字描述控制一个没有条件约束的扩散模型的生成过程。

个人笔记如下：
- [第4章 Diffusers实战](diffusion_models_learning51/ch04/ch04.md) 
- [第5章 微调和引导](diffusion_models_learning51/ch05/ch05.md) 

## Task03 Stable Diffusion原理与实战

本次任务，主要介绍了Stable Diffusion文本条件隐式扩散模型，包括隐式扩散、文本条件生成的概念，结合Stable Diffusion默认使用的管线，使用提示语句生成图片，还介绍了其他管线，比如Img2Img、Depth2Img，这两个管线的生成效果也非常优秀。

个人笔记如下：
- [第6章 Stable Diffusion](diffusion_models_learning51/ch06/ch06.md)

## Task04 DDIM反转与音频扩散模型

个人笔记如下：
- [第7章 DDIM反转](diffusion_models_learning51/ch07.md) 
- [第8章 音频扩散模型](diffusion_models_learning51/ch08.md) 