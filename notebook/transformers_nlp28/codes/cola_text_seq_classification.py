#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: cola_text_seq_classification.py
@time: 2021/8/24 15:03
@project: my-team-learning
@desc: Task06 文本分类任务
"""
import numpy as np
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

GLUE_TASKS = ["cola", "mnli", "mnli-mm", "mrpc",
              "qnli", "qqp", "rte", "sst2", "stsb", "wnli"]

# 分类任务为CoLA任务
task = "cola"
# BERT模型
model_checkpoint = "distilbert-base-uncased"
# 根据GPU调整batch_size大小，避免显存溢出
batch_size = 16

# 加载数据和评测方法
actual_task = "mnli" if task == "mnli-mm" else task
dataset = load_dataset("glue", actual_task)
metric = load_metric('glue', actual_task)

# 构造词分类器
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
# 定义如下dict，用于对数据格式进行检查
task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mnli-mm": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}
sentence1_key, sentence2_key = task_to_keys[task]


# 构造数据预处理函数
def preprocess_function(examples):
    if sentence2_key is None:
        return tokenizer(examples[sentence1_key], truncation=True)
    return tokenizer(examples[sentence1_key], examples[sentence2_key], truncation=True)


# 对所有数据进行预处理
encoded_dataset = dataset.map(preprocess_function, batched=True)

# 加载分类模型
num_labels = 3 if task.startswith("mnli") else 1 if task == "stsb" else 2
model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint, num_labels=num_labels)

# 设定训练参数
metric_name = "pearson" if task == "stsb" else "matthews_correlation" if task == "cola" else "accuracy"
args = TrainingArguments(
    "test-glue",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name,
)


# 根据任务名称获取评测方法
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    if task != "stsb":
        predictions = np.argmax(predictions, axis=1)
    else:
        predictions = predictions[:, 0]
    return metric.compute(predictions=predictions, references=labels)


# 构造训练器Trainer
validation_key = "validation_mismatched" if task == "mnli-mm" else "validation_matched" if task == "mnli" else "validation"
trainer = Trainer(
    model,
    args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset[validation_key],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# 模型训练
trainer.train()

# 模型评价
trainer.evaluate()


# 设置初始化模型
def model_init():
    return AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint, num_labels=num_labels)


# 使用1/10数据进行搜索
best_run = trainer.hyperparameter_search(n_trials=10, direction="maximize")

# 使用效果最好的参数进行模型训练
for n, v in best_run.hyperparameters.items():
    setattr(trainer.args, n, v)

trainer.train()

# 模型评价
trainer.evaluate()
