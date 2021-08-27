# Task09 Transformers解决机器翻译任务

## 1 加载数据


```python
# 使用opus-mt-en-ro模型
model_checkpoint = "Helsinki-NLP/opus-mt-en-ro" 
```


```python
from datasets import load_dataset, load_metric

# 加载WMT数据集
raw_datasets = load_dataset("wmt16", "ro-en")
# 加载sacrebleu评测方法
metric = load_metric("sacrebleu")
```

    Reusing dataset wmt16 (C:\Users\hurui\.cache\huggingface\datasets\wmt16\ro-en\1.0.0\0d9fb3e814712c785176ad8cdb9f465fbe6479000ee6546725db30ad8a8b5f8a)
    


```python
raw_datasets
```




    DatasetDict({
        train: Dataset({
            features: ['translation'],
            num_rows: 610320
        })
        validation: Dataset({
            features: ['translation'],
            num_rows: 1999
        })
        test: Dataset({
            features: ['translation'],
            num_rows: 1999
        })
    })




```python
# 查看训练集第一条数据
raw_datasets["train"][0]
```




    {'translation': {'en': 'Membership of Parliament: see Minutes',
      'ro': 'Componenţa Parlamentului: a se vedea procesul-verbal'}}




```python
import datasets
import random
import pandas as pd
from IPython.display import display, HTML


def show_random_elements(dataset, num_examples=5):
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
show_random_elements(raw_datasets["train"])
```


<table border="0" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>translation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>{'en': 'The Commission's proposal is quite simply not enough.', 'ro': 'Propunerea Comisiei este pur şi simplu insuficientă.'}</td>
    </tr>
    <tr>
      <th>1</th>
      <td>{'en': 'I voted in favour of the request made by Italy for aid in relation to accessing the European Union Solidarity Fund for rebuilding the Abruzzo region, extensively destroyed by the earthquake in April 2009, based on humanitarian considerations for a country in trouble.', 'ro': 'în scris. - Am votat în favoarea cererii de ajutorare venite din partea Italiei privind accesul la Fondul de Solidaritate al Uniunii Europene pentru reconstrucţia regiunii Abruzzo, distrusă masiv de cutremurul din Aprilie 2009, din considerente umanitare pentru o ţară aflată în necaz.'}</td>
    </tr>
    <tr>
      <th>2</th>
      <td>{'en': '0.35% of the EU's GDP may not be sufficient to achieve our objectives.', 'ro': '0,35% din PIB-ul UE ar putea fi insuficient pentru îndeplinirea obiectivelor noastre.'}</td>
    </tr>
    <tr>
      <th>3</th>
      <td>{'en': 'It includes graduated public funding that reflects employment and environmental considerations, the fact that agriculture produces public goods and the concept of active farmers.', 'ro': 'Include finanțare publică progresivă care să reflecte ocuparea forței de muncă și considerentele de mediu, faptul că agricultura produce bunuri publice și conceptul de agricultori activi.'}</td>
    </tr>
    <tr>
      <th>4</th>
      <td>{'en': 'We are expanding our cooperation beyond economic issues to cover key strategic issues - climate change, non-proliferation, tackling drugs, the promotion of peace and security worldwide.', 'ro': 'Ne extindem cooperarea dincolo de chestiunile economice pentru a acoperi probleme strategice cheie: schimbările climatice, neproliferarea, contrabanda cu droguri, promovarea păcii şi securităţii în întreaga lume.'}</td>
    </tr>
  </tbody>
</table>



```python
# 查看metric
metric
```




    Metric(name: "sacrebleu", features: {'predictions': Value(dtype='string', id='sequence'), 'references': Sequence(feature=Value(dtype='string', id='sequence'), length=-1, id='references')}, usage: """
    Produces BLEU scores along with its sufficient statistics
    from a source against one or more references.
    
    Args:
        predictions: The system stream (a sequence of segments).
        references: A list of one or more reference streams (each a sequence of segments).
        smooth_method: The smoothing method to use. (Default: 'exp').
        smooth_value: The smoothing value. Only valid for 'floor' and 'add-k'. (Defaults: floor: 0.1, add-k: 1).
        tokenize: Tokenization method to use for BLEU. If not provided, defaults to 'zh' for Chinese, 'ja-mecab' for
            Japanese and '13a' (mteval) otherwise.
        lowercase: Lowercase the data. If True, enables case-insensitivity. (Default: False).
        force: Insist that your tokenized input is actually detokenized.
    
    Returns:
        'score': BLEU score,
        'counts': Counts,
        'totals': Totals,
        'precisions': Precisions,
        'bp': Brevity penalty,
        'sys_len': predictions length,
        'ref_len': reference length,
    
    Examples:
    
        >>> predictions = ["hello there general kenobi", "foo bar foobar"]
        >>> references = [["hello there general kenobi", "hello there !"], ["foo bar foobar", "foo bar foobar"]]
        >>> sacrebleu = datasets.load_metric("sacrebleu")
        >>> results = sacrebleu.compute(predictions=predictions, references=references)
        >>> print(list(results.keys()))
        ['score', 'counts', 'totals', 'precisions', 'bp', 'sys_len', 'ref_len']
        >>> print(round(results["score"], 1))
        100.0
    """, stored examples: 0)



## 2 数据预处理

### 2.1 数据预处理流程
- 使用工具：Tokenizer
- 流程：
  1. 对输入数据进行tokenize，得到tokens
  2. 将tokens转化为预训练模型中需要对应的token ID
  3. 将token ID转化为模型需要的输入格式

### 2.2 构建模型对应的tokenizer


```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
```

### 2.3 整合预处理函数


```python
max_input_length = 128
max_target_length = 128
source_lang = "en"
target_lang = "ro"
prefix = ""


def preprocess_function(examples):
    inputs = [prefix + ex[source_lang] for ex in examples["translation"]]
    targets = [ex[target_lang] for ex in examples["translation"]]
    model_inputs = tokenizer(
        inputs, max_length=max_input_length, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets, max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
```


```python
preprocess_function(raw_datasets['train'][:2])
```




    {'input_ids': [[393, 4462, 14, 1137, 53, 216, 28636, 0], [24385, 14, 28636, 14, 4646, 4622, 53, 216, 28636, 0]], 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], 'labels': [[42140, 494, 1750, 53, 8, 59, 903, 3543, 9, 15202, 0], [36199, 6612, 9, 15202, 122, 568, 35788, 21549, 53, 8, 59, 903, 3543, 9, 15202, 0]]}



### 2.4 对数据集datasets所有样本进行预处理


```python
tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)
```


## 3 微调预训练模型

### 3.1 加载seq2seq模型


```python
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer

model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
```

### 3.2 设定训练参数


```python
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
```


```python
# 数据收集器，用于将处理好的数据输入给模型
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
```

### 3.3 数据后处理


```python
import numpy as np

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
```

### 3.4 训练模型


```python
trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)
```


```python
trainer.train()
```



<div>

  <progress value='38145' max='38145' style='width:300px; height:20px; vertical-align: middle;'></progress>
  [38145/38145 1&#58;05&#58;45, Epoch 1/1]
</div>
<table border="0" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>Epoch</th>
      <th>Training Loss</th>
      <th>Validation Loss</th>
      <th>Bleu</th>
      <th>Gen Len</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>No log</td>
      <td>1.288526</td>
      <td>27.987600</td>
      <td>34.085500</td>
    </tr>
  </tbody>
</table><p>


    E:\LearningDisk\Learning_Projects\MyPythonProjects\my-team-learning\venv\lib\site-packages\torch\_tensor.py:575: UserWarning: floor_divide is deprecated, and will be removed in a future version of pytorch. It currently rounds toward 0 (like the 'trunc' function NOT 'floor'). This results in incorrect rounding for negative values.
    To keep the current behavior, use torch.div(a, b, rounding_mode='trunc'), or for actual floor division, use torch.div(a, b, rounding_mode='floor'). (Triggered internally at  ..\aten\src\ATen\native\BinaryOps.cpp:467.)
      return torch.floor_divide(self, other)
    




    TrainOutput(global_step=38145, training_loss=0.7717016278345786, metrics={'train_runtime': 3946.4558, 'train_samples_per_second': 154.65, 'train_steps_per_second': 9.666, 'total_flos': 2.128216892689613e+16, 'train_loss': 0.7717016278345786, 'epoch': 1.0})



## 4 总结

&emsp;&emsp;本次任务，主要介绍了用Helsinki-NLP/opus-mt-en-ro模型解决机器翻译任务的方法及步骤，步骤主要分为加载数据、数据预处理、微调预训练模型。在加载数据阶段中，使用WMT数据集；在数据预处理阶段中，对tokenizer分词器的建模，使用as_target_tokenizer控制target对应的特殊token，并完成数据集中所有样本的预处理；在微调预训练模型阶段，使用Seq2SeqTrainingArguments对模型参数进行设置，并构建Seq2SeqTrainer训练器，进行模型训练和评估。  
&emsp;&emsp;其中在数据集下载时，需要使用外网方式建立代理；sacrebleu需要安装1.5.1版本；本次任务中的模型训练，笔者使用的是3070  GPU显卡，需要训练模型长达1小时。
