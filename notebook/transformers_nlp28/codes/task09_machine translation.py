#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: task09_machine translation.py
@time: 2021/8/27 15:59
@project: my-team-learning
@desc: Task09 机器翻译任务
"""
from transformers import AutoTokenizer
from datasets import load_dataset, load_metric
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
import numpy as np

# 使用opus-mt-en-ro模型
model_checkpoint = "Helsinki-NLP/opus-mt-en-ro"

# 加载WMT数据集
raw_datasets = load_dataset("wmt16", "ro-en")
# 加载sacrebleu评测方法
metric = load_metric("sacrebleu")

# 构建模型对应的tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

max_input_length = 128
max_target_length = 128
source_lang = "en"
target_lang = "ro"
prefix = ""


def preprocess_function(examples):
    """预处理函数"""
    inputs = [prefix + ex[source_lang] for ex in examples["translation"]]
    targets = [ex[target_lang] for ex in examples["translation"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# 对数据集datasets所有样本进行预处理
tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)

# 加载模型
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

# 设置训练参数
batch_size = 16
args = Seq2SeqTrainingArguments(
    "test-translation",
    # 每个epcoh会做一次验证评估
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=1,
    predict_with_generate=True,
    fp16=False,
    log_level='error',
    logging_strategy="no",
    report_to="none"
)

# 数据收集器，用于将处理好的数据输入给模型
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result

trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()