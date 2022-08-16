# Task01 OpenVINO核心组件资源和开发流程

## 1 课程概览
在这次课程中，将深入学习OpenVINO，并了解基于OpenVINO生态系统的软件工具：
- 学习拓展资料：OpenModelZoo中经过深度优化的模型，提供很多C++和Python的Demo
- 异构系统设计
- 推理性能的优化方式
- 真正产品应用的重要性
- 低精度的模型校准
- 构建复杂的DL-Streamer流水线处理
- 音频和视频的处理及项目应用

## 2 第1课：OpenVINO核心组件资源和开发流程

### 2.1 OpenVINO套件架构
- 支持Intel架构的平台，包括Intel集成显卡、智能硬件等
- 传统计算机视觉工具：OpenCV、OpenVX，主要用于Intel CPU和集成显卡
- 工具包：Intel Media SDK（快速视频解码）、OpenCL驱动和运行环境，以及用于Linux的Optimize Intel FPGA运行环境等
- DLDT（深度学习部署套件）：模型优化器（支持各种模型转换，用于预处理，输出IR文件）和推理引擎（使用IR文件执行推理请求），IR文件包括xml文件（描述网络结构）和bin文件（参数权重）
- Open Model Zoo：提供预训练模型、示例、模型下载器
- 模型工具：Calibration Tool、Benchmark app、Model Analyzer、Accuracy Checker、Aux Capabilities
- DL Streamer：构建Gstreamer的视频分析流程
- OVMS：OpenVINO模型服务器，用于构建推理服务器
- Training extensions：提供模型的重训练
- DL WorkBench：基于可视化的Benchmarking应用工具

### 2.2 实验演示

- 3D Human Pose Demo：实时检测视频中多人的人体3D姿势，用于电影制作、零售、游戏、增强现实等场景，难点在于人体姿势中的小细节，比如：耳朵、眼睛、鼻子、肩膀、膝盖、衣服和灯光的变化，以及其他物体造成的遮挡。

- Colorization Demo：拍摄灰度图像，还原颜色并重新着色，颜色可增强图片效果，提升图像和视频的逼真度；难点是如何在颜色消失后，重新添加颜色。

- Audio Detection Demo：处理音频信号，基于DL-Streamer的声音检测和分类演示

- Formula Recognition：检测自由格式的手写公式或Latex公式的示例，通过Encoder（卷积神经网络）提取图像中的特征，主要是识别字母或图像的边界框，再使用Decoder查找符号。

- Mono-Depth Demo：从简单的2D图像中创建一个3D图像

- Object Detection Demo：对象检测演示

- BERT Question Answering Demo：使用BERT处理文本文件，智能问答演示