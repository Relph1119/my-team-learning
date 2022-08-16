# Task03 AI推理中的推理性能

## 1 第4课：OpenVINO推理性能

- 性能：
  1. 吞吐量：FPS（frames/sec）或(Inferences/sec)
  2. 延迟：处理一次推理所需的时间(Milliseconds)
  3. 效率：功耗、价格等
- 影响应用推理性能的参数：
  1. 网络参数：网络拓扑参数，各层输入的连接、图像大小等
  2. 设备参数：数据格式、内存大小、计算量
  3. 推理执行参数：通过benchmark_app.py指定执行参数
- 推理执行参数：
  1. 同步/异步执行
  2. Batch size：处理多个数据输入，更高的批次不一定会产生更高的性能，可能在内存过载时降低性能，可能会增加延迟
  3. 视频流的数量：在CPU吞吐量模式(Throughput Mode)下，线程将进行流分组，从而提升内核和内部资源的分配效率，流内部的同步开销更低，数据局部性更佳，有助于提高性能
  4. 线程数：将线程固定到特定核心
  5. 数据格式
  
## 2 第5课：AI推理中整数精度的推理

- 数据格式的性能影响：内存大小（1 FP32 = 4 INT8），可以将多个INT8进行打包，一并执行单指令多数据操作；SSE4.2可打包16个INT8，AVX2可打包32个INT8，AVX512可打包64个INT8
- 数据格式可以影响准确性：浮点数有非常宽的动态范围，整数的范围很窄
  1. 使用低精度的权重重新训练模型，可使用OpenVION training extensions进行重训练
  2. 转换模型精度：模型优化器可以执行简单的转换，例如FP32转为FP16
- 模型校准为整数的工具：DL Workbench、POT(Post-Training Optimization Tool)
- POT介绍：具有整套处理流程（reader -> Annotation conversion -> Pre-processing -> Launcher(INFERENCE) -> Calculate merics），提供两种模式（默认量化模式、准确性感知量化模式）
