# Task06 Transformers解决文本分类任务、超参搜索

## 1 文本分类任务简介
- 使用Transformers代码库中的模型来解决文本分类任务，任务来源于[GLUE Benchmark](https://gluebenchmark.com/)
- GLUE榜单的9个级别的分类任务：
  1. CoLA (Corpus of Linguistic Acceptability)：鉴别一个句子是否语法正确.
  2. MNLI (Multi-Genre Natural Language Inference)：给定一个假设，判断另一个句子与该假设的关系：entails、contradicts、unrelated。
  3. MRPC (Microsoft Research Paraphrase Corpus)：判断两个句子是否互为paraphrases
  4. QNLI (Question-answering Natural Language Inference)：判断第2句是否包含第1句问题的答案
  5. QQP (Quora Question Pairs2)：判断两个问句是否语义相同
  6. RTE (Recognizing Textual Entailment)：判断一个句子是否与假设成entail关系
  7. SST-2 (Stanford Sentiment Treebank)：判断一个句子的情感正负向
  8. STS-B (Semantic Textual Similarity Benchmark)：判断两个句子的相似性（分数为1-5分）
  9. WNLI (Winograd Natural Language Inference)：判断带有匿名代词的句子中，是否存在能够替换该代词的子句


```python
GLUE_TASKS = ["cola", "mnli", "mnli-mm", "mrpc",
              "qnli", "qqp", "rte", "sst2", "stsb", "wnli"]
```


```python
# 任务为CoLA任务
task = "cola"
# BERT模型
model_checkpoint = "distilbert-base-uncased"
# 根据GPU调整batch_size大小，避免显存溢出
batch_size = 16
```

## 2 加载数据

### 2.1 加载数据和对应的评测方式


```python
from datasets import load_dataset, load_metric
```


```python
actual_task = "mnli" if task == "mnli-mm" else task
# 加载GLUE数据集
dataset = load_dataset("glue", actual_task)
# 加载GLUE的评测方式
metric = load_metric('glue', actual_task)
```

    Reusing dataset glue (C:\Users\hurui\.cache\huggingface\datasets\glue\cola\1.0.0\dacbe3125aa31d7f70367a07a8a9e72a5a0bfeb5fc42e75c9db75b96da6053ad)
    

### 2.2 查看数据


```python
# 对于训练集、验证集和测试集，只需要使用对应的key（train，validation，test）即可得到相应的数据
dataset
```




    DatasetDict({
        train: Dataset({
            features: ['sentence', 'label', 'idx'],
            num_rows: 8551
        })
        validation: Dataset({
            features: ['sentence', 'label', 'idx'],
            num_rows: 1043
        })
        test: Dataset({
            features: ['sentence', 'label', 'idx'],
            num_rows: 1063
        })
    })




```python
# 查看训练集第一条数据
dataset["train"][0]
```




    {'sentence': "Our friends won't buy this analysis, let alone the next one we propose.",
     'label': 1,
     'idx': 0}




```python
import datasets
import random
import pandas as pd
from IPython.display import display, HTML


def show_random_elements(dataset, num_examples=10):
    """从数据集中随机选择几条数据"""
    assert num_examples <= len(
        dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset)-1)
        while pick in picks:
            pick = random.randint(0, len(dataset)-1)
        picks.append(pick)

    df = pd.DataFrame(dataset[picks])
    for column, typ in dataset.features.items():
        if isinstance(typ, datasets.ClassLabel):
            df[column] = df[column].transform(lambda i: typ.names[i])
    display(HTML(df.to_html()))
```


```python
show_random_elements(dataset["train"])
```


<table border="0" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sentence</th>
      <th>label</th>
      <th>idx</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>No one can forgive you that comment.</td>
      <td>acceptable</td>
      <td>2078</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Bill and Kathy married.</td>
      <td>acceptable</td>
      <td>2318</td>
    </tr>
    <tr>
      <th>2</th>
      <td>$5 will buy a ticket.</td>
      <td>acceptable</td>
      <td>2410</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Which books did Robin talk to Chris and read?</td>
      <td>unacceptable</td>
      <td>7039</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Jill offered the ball towards Bob.</td>
      <td>unacceptable</td>
      <td>2053</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Has not Henri studied for his exam?</td>
      <td>unacceptable</td>
      <td>7466</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Fanny stopped talking when in came Aunt Norris.</td>
      <td>unacceptable</td>
      <td>6778</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Who do you think that would be nominated for the position?</td>
      <td>unacceptable</td>
      <td>4784</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Mickey teamed with the women up.</td>
      <td>unacceptable</td>
      <td>440</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Baseballs toss easily.</td>
      <td>unacceptable</td>
      <td>2783</td>
    </tr>
  </tbody>
</table>


### 2.3 查看评测方法


```python
metric
```




    Metric(name: "glue", features: {'predictions': Value(dtype='int64', id=None), 'references': Value(dtype='int64', id=None)}, usage: """
    Compute GLUE evaluation metric associated to each GLUE dataset.
    Args:
        predictions: list of predictions to score.
            Each translation should be tokenized into a list of tokens.
        references: list of lists of references for each translation.
            Each reference should be tokenized into a list of tokens.
    Returns: depending on the GLUE subset, one or several of:
        "accuracy": Accuracy
        "f1": F1 score
        "pearson": Pearson Correlation
        "spearmanr": Spearman Correlation
        "matthews_correlation": Matthew Correlation
    Examples:
    
        >>> glue_metric = datasets.load_metric('glue', 'sst2')  # 'sst2' or any of ["mnli", "mnli_mismatched", "mnli_matched", "qnli", "rte", "wnli", "hans"]
        >>> references = [0, 1]
        >>> predictions = [0, 1]
        >>> results = glue_metric.compute(predictions=predictions, references=references)
        >>> print(results)
        {'accuracy': 1.0}
    
        >>> glue_metric = datasets.load_metric('glue', 'mrpc')  # 'mrpc' or 'qqp'
        >>> references = [0, 1]
        >>> predictions = [0, 1]
        >>> results = glue_metric.compute(predictions=predictions, references=references)
        >>> print(results)
        {'accuracy': 1.0, 'f1': 1.0}
    
        >>> glue_metric = datasets.load_metric('glue', 'stsb')
        >>> references = [0., 1., 2., 3., 4., 5.]
        >>> predictions = [0., 1., 2., 3., 4., 5.]
        >>> results = glue_metric.compute(predictions=predictions, references=references)
        >>> print({"pearson": round(results["pearson"], 2), "spearmanr": round(results["spearmanr"], 2)})
        {'pearson': 1.0, 'spearmanr': 1.0}
    
        >>> glue_metric = datasets.load_metric('glue', 'cola')
        >>> references = [0, 1]
        >>> predictions = [0, 1]
        >>> results = glue_metric.compute(predictions=predictions, references=references)
        >>> print(results)
        {'matthews_correlation': 1.0}
    """, stored examples: 0)




```python
# 调用metric的compute方法，计算评测值
import numpy as np

fake_preds = np.random.randint(0, 2, size=(64,))
fake_labels = np.random.randint(0, 2, size=(64,))
metric.compute(predictions=fake_preds, references=fake_labels)
```




    {'matthews_correlation': -0.00392156862745098}



### 2.4 文本分类任务与评测方法

| 任务 | 评测方法 |
| :---: | :---: |
| CoLA | [Matthews Correlation Coefficient](https://en.wikipedia.org/wiki/Matthews_correlation_coefficient) | 
| MNLI | Accuracy |
| MRPC | Accuracy and [F1 score](https://en.wikipedia.org/wiki/F1_score) |
| QNLI | Accuracy |
| QQP | Accuracy and F1 score |
| RTE | Accuracy |
| SST-2 | Accuracy |
| STS-B | [Pearson Correlation Coefficient](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient) and [Spearman's_Rank_Correlation_Coefficient](https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient) |
| WNLI | Accuracy |

## 3 数据预处理

### 3.1 数据预处理流程
- 使用工具：Tokenizer
- 流程：
  1. 对输入数据进行tokenize，得到tokens
  2. 将tokens转化为预训练模型中需要对应的token ID
  3. 将token ID转化为模型需要的输入格式

### 3.2 构建模型对应的tokenizer


```python
from transformers import AutoTokenizer
    
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
```


```python
tokenizer("Hello, this one sentence!", "And this sentence goes with it.")
```




    {'input_ids': [101, 7592, 1010, 2023, 2028, 6251, 999, 102, 1998, 2023, 6251, 3632, 2007, 2009, 1012, 102], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}



### 3.3 对数据集datasets所有样本进行预处理


```python
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
```


```python
# 对训练数据集的第1条数据进行数据格式检查
sentence1_key, sentence2_key = task_to_keys[task]
if sentence2_key is None:
    print(f"Sentence: {dataset['train'][0][sentence1_key]}")
else:
    print(f"Sentence 1: {dataset['train'][0][sentence1_key]}")
    print(f"Sentence 2: {dataset['train'][0][sentence2_key]}")
```

    Sentence: Our friends won't buy this analysis, let alone the next one we propose.
    


```python
# 构造数据预处理函数
def preprocess_function(examples):
    if sentence2_key is None:
        return tokenizer(examples[sentence1_key], truncation=True)
    return tokenizer(examples[sentence1_key], examples[sentence2_key], truncation=True)
```


```python
# 对所有数据进行预处理
encoded_dataset = dataset.map(preprocess_function, batched=True)
```

    Loading cached processed dataset at C:\Users\hurui\.cache\huggingface\datasets\glue\cola\1.0.0\dacbe3125aa31d7f70367a07a8a9e72a5a0bfeb5fc42e75c9db75b96da6053ad\cache-fd5eee62c2b8c26e.arrow
    Loading cached processed dataset at C:\Users\hurui\.cache\huggingface\datasets\glue\cola\1.0.0\dacbe3125aa31d7f70367a07a8a9e72a5a0bfeb5fc42e75c9db75b96da6053ad\cache-0ce499346cf9c20b.arrow
    


    HBox(children=(FloatProgress(value=0.0, max=2.0), HTML(value='')))


    
    

## 4 微调预训练模型

### 4.1 加载分类模型


```python
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

num_labels = 3 if task.startswith("mnli") else 1 if task == "stsb" else 2
model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint, num_labels=num_labels)
```

    Some weights of the model checkpoint at distilbert-base-uncased were not used when initializing DistilBertForSequenceClassification: ['vocab_layer_norm.weight', 'vocab_transform.weight', 'vocab_projector.bias', 'vocab_projector.weight', 'vocab_transform.bias', 'vocab_layer_norm.bias']
    - This IS expected if you are initializing DistilBertForSequenceClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
    - This IS NOT expected if you are initializing DistilBertForSequenceClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
    Some weights of DistilBertForSequenceClassification were not initialized from the model checkpoint at distilbert-base-uncased and are newly initialized: ['pre_classifier.bias', 'classifier.weight', 'classifier.bias', 'pre_classifier.weight']
    You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
    

### 4.2 设定训练参数


```python
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
    log_level='error',
    logging_strategy="no",
    report_to="none"
)
```


```python
# 根据任务名称获取评测方法
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    if task != "stsb":
        predictions = np.argmax(predictions, axis=1)
    else:
        predictions = predictions[:, 0]
    return metric.compute(predictions=predictions, references=labels)
```


```python
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
```

### 4.3 训练模型


```python
trainer.train()
```


    TrainOutput(global_step=2675, training_loss=0.2717150308484229, metrics={'train_runtime': 100.5668, 'train_samples_per_second': 425.14, 'train_steps_per_second': 26.599, 'total_flos': 229537542078168.0, 'train_loss': 0.2717150308484229, 'epoch': 5.0})



### 4.4 模型评估


```python
trainer.evaluate()
```



<div>

  <progress value='66' max='66' style='width:300px; height:20px; vertical-align: middle;'></progress>
  [66/66 00:00]
</div>






    {'eval_loss': 0.8624260425567627,
     'eval_matthews_correlation': 0.519563286537562,
     'eval_runtime': 0.6501,
     'eval_samples_per_second': 1604.31,
     'eval_steps_per_second': 101.519,
     'epoch': 5.0}



## 5 超参数搜索

### 5.1 设置初始化模型


```python
def model_init():
    return AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint, num_labels=num_labels)
```


```python
trainer = Trainer(
    model_init=model_init,
    args=args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset[validation_key],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)
```

### 5.2 超参数搜索


```python
# 使用1/10数据进行搜索
best_run = trainer.hyperparameter_search(n_trials=10, direction="maximize")
```

```python
# 得到效果最好的模型参数
best_run
```




    BestRun(run_id='3', objective=0.5504031254980248, hyperparameters={'learning_rate': 4.301257551502102e-05, 'num_train_epochs': 5, 'seed': 20, 'per_device_train_batch_size': 8})



### 5.3 设置效果最好的参数并训练模型


```python
for n, v in best_run.hyperparameters.items():
    setattr(trainer.args, n, v)

trainer.train()
```


    TrainOutput(global_step=5345, training_loss=0.26719996967083726, metrics={'train_runtime': 178.4912, 'train_samples_per_second': 239.536, 'train_steps_per_second': 29.945, 'total_flos': 413547436355364.0, 'train_loss': 0.26719996967083726, 'epoch': 5.0})


```python
trainer.evaluate()
```


    {'eval_loss': 0.9789257049560547,
     'eval_matthews_correlation': 0.5548273578107759,
     'eval_runtime': 0.6556,
     'eval_samples_per_second': 1590.796,
     'eval_steps_per_second': 100.664,
     'epoch': 5.0}



## 6 总结

&emsp;&emsp;本次任务，主要介绍了用BERT模型解决文本分类任务的方法及步骤，步骤主要分为加载数据、数据预处理、微调预训练模型和超参数搜索。在加载数据阶段中，必须使用与分类任务相应的评测方法；在数据预处理阶段中，对tokenizer分词器的建模，并完成数据集中所有样本的预处理；在微调预训练模型阶段，通过对模型参数进行设置，并构建Trainner训练器，进行模型训练和评估；最后在超参数搜索阶段，使用hyperparameter_search方法，搜索效果最好的超参数，并进行模型训练和评估。  
&emsp;&emsp;其中在数据集下载时，需要使用外网方式建立代理。如果使用conda安装ray\[tune\]包时，请下载对应ray-tune依赖包。
