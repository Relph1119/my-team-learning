#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: ner_seq_labeling.py
@time: 2021/8/26 0:49
@project: my-team-learning
@desc: Task07 序列标注任务
"""
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
import numpy as np
from transformers import DataCollatorForTokenClassification

# 设置分类任务
task = "ner"
# 设置BERT模型
model_checkpoint = "distilbert-base-uncased"
# 根据GPU调整batch_size大小，避免显存溢出
batch_size = 16

# 加载CONLL 2003数据集
datasets = load_dataset("conll2003")

label_list = datasets["train"].features[f"{task}_tags"].feature.names

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

label_all_tokens = True


def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples[f"{task}_tags"]):
        # 获取subtokens位置
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []

        # 遍历subtokens位置索引
        for word_idx in word_ids:
            if word_idx is None:
                # 将特殊字符的label设置为-100
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx

        # 对齐word
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


# 对数据集datasets所有样本进行预处理
tokenized_datasets = datasets.map(tokenize_and_align_labels, batched=True)

# 加载分类模型
model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=len(label_list))

# 设定训练参数
args = TrainingArguments(
    f"test-{task}",
    # 每个epcoh会做一次验证评估
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=3,
    weight_decay=0.01,
    log_level='error',
    logging_strategy="no",
    report_to="none"
)

# 数据收集器，用于将处理好的数据输入给模型
data_collator = DataCollatorForTokenClassification(tokenizer)

# 设定评估方法
metric = load_metric("seqeval")


def compute_metrics(p):
    """模型预测"""
    predictions, labels = p
    # 选择预测分类最大概率的下标
    predictions = np.argmax(predictions, axis=2)

    # 将下标转化为label，并忽略-100的位置
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


# 构造训练器Trainer
trainer = Trainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()

trainer.evaluate()

# 输出单个类别的precision/recall/f1
predictions, labels, _ = trainer.predict(tokenized_datasets["validation"])
predictions = np.argmax(predictions, axis=2)

# Remove ignored index (special tokens)
true_predictions = [
    [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
    for prediction, label in zip(predictions, labels)
]
true_labels = [
    [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
    for prediction, label in zip(predictions, labels)
]

results = metric.compute(predictions=true_predictions, references=true_labels)
print(results)
