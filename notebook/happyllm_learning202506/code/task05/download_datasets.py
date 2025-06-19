#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: download_datasets.py
@time: 2025/6/19 18:19
@project: my-team-learning
@desc: 
"""

import os

# 设置环境变量
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# dataset dir 下载到本地目录
dataset_dir = "../../datasets/"

# 下载预训练数据集
os.system(
    f"modelscope download --dataset ddzhu123/seq-monkey mobvoi_seq_monkey_general_open_corpus.jsonl.tar.bz2 --local_dir {dataset_dir}")

# 解压预训练数据集
# tar -xvf "${dataset_dir}/mobvoi_seq_monkey_general_open_corpus.jsonl.tar.bz2" -C "${dataset_dir}"
