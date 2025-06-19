#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: model_pretrain.py
@time: 2025/6/19 16:24
@project: my-team-learning
@desc: 
"""

# 加载定义好的模型参数-此处以 Qwen-2.5-1.5B 为例
# 使用 transforemrs 的 Config 类进行加载
from transformers import AutoConfig
from transformers import AutoModelForCausalLM

# 下载参数的本地路径
model_path = "../../models/"
config = AutoConfig.from_pretrained(model_path)

# 使用该配置生成一个定义好的模型
model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

# 打印模型结构
print(model)

