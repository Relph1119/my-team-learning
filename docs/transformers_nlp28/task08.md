# Task08 Transformers解决抽取式问答任务

## 1 抽取式问答任务简介

- 抽取式问答任务：给定一个问题和一段文本，从这段文本中找出能回答该问题的文本片段（span）
- 本次示例可用于解决任何与SQUAD 1和SQUAD 2类似的抽取式问答任务


```python
# squad_v2等于True或者False分别代表使用SQUAD v2 或者 SQUAD v1。
# True表示部分问题不能给出答案，False表示所有问题必须回答。
squad_v2 = False
# 设置BERT模型
model_checkpoint = "distilbert-base-uncased"
# 根据GPU调整batch_size大小，避免显存溢出
batch_size = 16
```

## 2 加载数据集


```python
from datasets import load_dataset, load_metric
```


```python
# 加载SQUAD数据集
datasets = load_dataset("squad_v2" if squad_v2 else "squad")
```

    Reusing dataset squad (C:\Users\hurui\.cache\huggingface\datasets\squad\plain_text\1.0.0\d6ec3ceb99ca480ce37cdd35555d6cb2511d223b9150cce08a837ef62ffea453)
    


```python
datasets
```




    DatasetDict({
        train: Dataset({
            features: ['id', 'title', 'context', 'question', 'answers'],
            num_rows: 87599
        })
        validation: Dataset({
            features: ['id', 'title', 'context', 'question', 'answers'],
            num_rows: 10570
        })
    })




```python
# 查看训练集第一条数据
datasets["train"][0]
```




    {'id': '5733be284776f41900661182',
     'title': 'University_of_Notre_Dame',
     'context': 'Architecturally, the school has a Catholic character. Atop the Main Building\'s gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend "Venite Ad Me Omnes". Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary.',
     'question': 'To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?',
     'answers': {'text': ['Saint Bernadette Soubirous'], 'answer_start': [515]}}




```python
from datasets import ClassLabel, Sequence
import random
import pandas as pd
from IPython.display import display, HTML

def show_random_elements(dataset, num_examples=10):
    """从数据集中随机选择几条数据"""
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset)-1)
        while pick in picks:
            pick = random.randint(0, len(dataset)-1)
        picks.append(pick)
    
    df = pd.DataFrame(dataset[picks])
    for column, typ in dataset.features.items():
        if isinstance(typ, ClassLabel):
            df[column] = df[column].transform(lambda i: typ.names[i])
        elif isinstance(typ, Sequence) and isinstance(typ.feature, ClassLabel):
            df[column] = df[column].transform(lambda x: [typ.feature.names[i] for i in x])
    display(HTML(df.to_html()))
```


```python
show_random_elements(datasets["train"], num_examples=2)
```


<table border="0" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>title</th>
      <th>context</th>
      <th>question</th>
      <th>answers</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>571ae05d9499d21900609b85</td>
      <td>Athanasius_of_Alexandria</td>
      <td>After the death of the replacement bishop Gregory in 345, Constans used his influence to allow Athanasius to return to Alexandria in October 345, amidst the enthusiastic demonstrations of the populace. This began a "golden decade" of peace and prosperity, during which time Athanasius assembled several documents relating to his exiles and returns from exile in the Apology Against the Arians. However, upon Constans's death in 350, another civil war broke out, which left pro-Arian Constantius as sole emperor. An Alexandria local council in 350 replaced (or reaffirmed) Athanasius in his see.</td>
      <td>In what writing did he recount his time in exiles?</td>
      <td>{'text': ['Apology Against the Arians'], 'answer_start': [366]}</td>
    </tr>
    <tr>
      <th>1</th>
      <td>572fee1ea23a5019007fcb66</td>
      <td>Premier_League</td>
      <td>Stadium attendances are a significant source of regular income for Premier League clubs. For the 2009–10 season, average attendances across the league clubs were 34,215 for Premier League matches with a total aggregate attendance figure of 13,001,616. This represents an increase of 13,089 from the average attendance of 21,126 recorded in the league's first season (1992–93). However, during the 1992–93 season the capacities of most stadiums were reduced as clubs replaced terraces with seats in order to meet the Taylor Report's 1994–95 deadline for all-seater stadiums. The Premier League's record average attendance of 36,144 was set during the 2007–08 season. This record was then beaten in the 2013–14 season recording an average attendance of 36,695 with a total attendance of just under 14 million, the highest average in England's top flight since 1950.</td>
      <td>What was the Premier Leagues standard attendance in the 2007-08 season?</td>
      <td>{'text': ['The Premier League's record average attendance of 36,144 was set during the 2007–08 season.'], 'answer_start': [574]}</td>
    </tr>
  </tbody>
</table>


## 3 预处理数据


```python
import numpy as np
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer
from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer
from transformers import default_data_collator
import torch
import collections
from tqdm.auto import tqdm
```

### 3.1 数据预处理流程
- 使用工具：Tokenizer
- 流程：
  1. 对输入数据进行tokenize，得到tokens
  2. 将tokens转化为预训练模型中需要对应的token ID
  3. 将token ID转化为模型需要的输入格式

### 3.2 构建模型对应的tokenizer


```python
from transformers import AutoTokenizer
    
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
```


```python
import transformers

# 模型使用的时fast tokenizer
assert isinstance(tokenizer, transformers.PreTrainedTokenizerFast)
```

- 使用tokenizer的tokenize方法，查看tokenizer预处理之后的文本格式
- add_special_tokens参数，表示增加预训练模型所要求的特殊token


```python
print("单个文本tokenize: {}".format(tokenizer.tokenize(
    "What is your name?"), add_special_tokens=True))
print("2个文本tokenize: {}".format(tokenizer.tokenize(
    "My name is Sylvain.", add_special_tokens=True)))
```

    单个文本tokenize: ['what', 'is', 'your', 'name', '?']
    2个文本tokenize: ['[CLS]', 'my', 'name', 'is', 'sy', '##lva', '##in', '.', '[SEP]']
    


```python
# 对单个文本进行预处理
tokenizer("What is your name?")
```




    {'input_ids': [101, 2054, 2003, 2115, 2171, 1029, 102], 'attention_mask': [1, 1, 1, 1, 1, 1, 1]}




```python
# 对2个文本进行预处理
tokenizer("What is your name?", "My name is Sylvain.")
```




    {'input_ids': [101, 2054, 2003, 2115, 2171, 1029, 102, 2026, 2171, 2003, 25353, 22144, 2378, 1012, 102], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}



### 3.3 处理长文本

处理长文本流程：
1. 使用truncation和padding对超长文本进行切片，允许相邻切片之间有交集
2. 使用overflow_to_sample_mapping和offset_mapping，映射切片前的原始位置，用于找到答案的起始和结束位置
3. 对于所有切片进行遍历  
  （1）对于无答案的context，使用CLS所在的位置标注答案位置  
  （2）对于有答案的context，找到切片前的起始和结束位置，找到切片后token的起始和结束位置  
  （3）检测答案是否超出文本长度，超出则用CLS位置标注，没有超出，找到答案token的start和end位置
4. 返回tokenizer预处理之后的数据，满足预训练模型输入格式


```python
# 输入feature的最大长度，question和context拼接之后
max_length = 384 
# 2个切片之间的重合token数量
doc_stride = 128 
```


```python
# question拼接context，即context在右边
pad_on_right = tokenizer.padding_side == "right"
```


```python
def prepare_train_features(examples):
    # 既要对examples进行truncation（截断）和padding（补全）还要还要保留所有信息，所以要用的切片的方法。
    # 每一个一个超长文本example会被切片成多个输入，相邻两个输入之间会有交集。
    tokenized_examples = tokenizer(
        examples["question" if pad_on_right else "context"],
        examples["context" if pad_on_right else "question"],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # 使用overflow_to_sample_mapping参数来映射切片ID到原始ID。
    # 比如有2个expamples被切成4片，那么对应是[0, 0, 1, 1]，前两片对应原来的第一个example。
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    # offset_mapping也对应4片
    # offset_mapping参数帮助我们映射到原始输入，由于答案标注在原始输入上，用于找到答案的起始和结束位置。
    offset_mapping = tokenized_examples.pop("offset_mapping")

    # 重新标注数据
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        # 对每一片进行处理
        # 将无答案的样本标注到CLS上
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        # 区分question和context
        sequence_ids = tokenized_examples.sequence_ids(i)

        # 拿到原始的example 下标.
        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]
        # 如果没有答案，则使用CLS所在的位置为答案.
        if len(answers["answer_start"]) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            # 答案的character级别Start/end位置.
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            # 找到token级别的index start.
            token_start_index = 0
            while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                token_start_index += 1

            # 找到token级别的index end.
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                token_end_index -= 1

            # 检测答案是否超出文本长度，超出的话也适用CLS index作为标注.
            if not (offsets[token_start_index][0] <= start_char 
                    and offsets[token_end_index][1] >= end_char):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # 如果不超出则找到答案token的start和end位置。.
                # Note: we could go after the last offset if the answer is the last word (edge case).
                while token_start_index < len(offsets) \
                    and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_examples["start_positions"].append(
                    token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)

    return tokenized_examples
```


```python
# 处理5个样本
features = prepare_train_features(datasets['train'][:5])
```

### 3.4 对数据集datasets所有样本进行预处理


```python
tokenized_datasets = datasets.map(
    prepare_train_features, batched=True, remove_columns=datasets["train"].column_names)
```

    Loading cached processed dataset at C:\Users\hurui\.cache\huggingface\datasets\squad\plain_text\1.0.0\d6ec3ceb99ca480ce37cdd35555d6cb2511d223b9150cce08a837ef62ffea453\cache-c905aaf8d72f926b.arrow
    Loading cached processed dataset at C:\Users\hurui\.cache\huggingface\datasets\squad\plain_text\1.0.0\d6ec3ceb99ca480ce37cdd35555d6cb2511d223b9150cce08a837ef62ffea453\cache-c29fbdb11009bad4.arrow
    

## 4 微调预训练模型

### 4.1 加载预训练模型


```python
from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer

model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)
```

    Some weights of the model checkpoint at distilbert-base-uncased were not used when initializing DistilBertForQuestionAnswering: ['vocab_transform.bias', 'vocab_transform.weight', 'vocab_projector.weight', 'vocab_layer_norm.weight', 'vocab_projector.bias', 'vocab_layer_norm.bias']
    - This IS expected if you are initializing DistilBertForQuestionAnswering from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
    - This IS NOT expected if you are initializing DistilBertForQuestionAnswering from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
    Some weights of DistilBertForQuestionAnswering were not initialized from the model checkpoint at distilbert-base-uncased and are newly initialized: ['qa_outputs.bias', 'qa_outputs.weight']
    You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
    

### 4.2 设定训练参数


```python
args = TrainingArguments(
    f"test-squad",
    evaluation_strategy = "epoch", # 每个epcoh会做一次验证评估
    learning_rate=2e-5, #学习率
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=3, # 训练的论次
    weight_decay=0.01,
    log_level='error',
    logging_strategy="no",
    report_to="none"
)
```


```python
from transformers import default_data_collator

# 数据收集器，用于将处理好的数据输入给模型
data_collator = default_data_collator
```

### 4.3 训练模型


```python
trainer = Trainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)
```


```python
trainer.train()
```



<div>

  <progress value='16599' max='16599' style='width:300px; height:20px; vertical-align: middle;'></progress>
  [16599/16599 1&#58;03&#58;17, Epoch 3/3]
</div>
<table border="0" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>Epoch</th>
      <th>Training Loss</th>
      <th>Validation Loss</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>No log</td>
      <td>1.148925</td>
    </tr>
    <tr>
      <td>2</td>
      <td>No log</td>
      <td>1.115199</td>
    </tr>
    <tr>
      <td>3</td>
      <td>No log</td>
      <td>1.157934</td>
    </tr>
  </tbody>
</table><p>





    TrainOutput(global_step=16599, training_loss=1.0829564516311223, metrics={'train_runtime': 3798.7824, 'train_samples_per_second': 69.91, 'train_steps_per_second': 4.37, 'total_flos': 2.602335381127373e+16, 'train_loss': 1.0829564516311223, 'epoch': 3.0})



## 5 模型评估

模型评估流程：
1. 得到模型预测的输出结果（answer所在start/end位置的logits）
2. 将answer的start和end的logits相加打分，在n_best_size个(start,end)对，得到相应的答案
3. 检查答案是否有效，检查start和end位置对应的文本是否在content里，而不在question里
4. 根据score值，对valid_answers进行排序，选择得分最高的作为答案
5. 将features和example进行map映射，用于计算评测指标
6. 解决无答案的情况：  
（1）将无答案的预测得分进行收集  
（2）检测在多个features里是否都无答案  
（3）选择所有features的无答案里得分最小的作为答案
7. 在原始预测上使用后处理函数
8. 使用squad评测方法，基于预测和标注对评测指标进行计算

### 5.1 得到模型预测输出结果


```python
# 得到模型预测输出结果
for batch in trainer.get_eval_dataloader():
    break
batch = {k: v.to(trainer.args.device) for k, v in batch.items()}
with torch.no_grad():
    output = trainer.model(**batch)
output.keys()
```




    odict_keys(['loss', 'start_logits', 'end_logits'])




```python
n_best_size = 20

start_logits = output.start_logits[0].cpu().numpy()
end_logits = output.end_logits[0].cpu().numpy()
# 收集最佳的start和end logits的位置
start_indexes = np.argsort(start_logits)[-1: -n_best_size - 1: -1].tolist()
end_indexes = np.argsort(end_logits)[-1: -n_best_size - 1: -1].tolist()
valid_answers = []
for start_index in start_indexes:
    for end_index in end_indexes:
        if start_index <= end_index:  # 如果start < end，那么合理的
            valid_answers.append(
                {
                    "score": start_logits[start_index] + end_logits[end_index],
                    "text": ""  # 后续需要根据token的下标将答案找出来
                }
            )
```

### 5.2 对验证集进行处理


```python
def prepare_validation_features(examples):
    # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    tokenized_examples = tokenizer(
        examples["question" if pad_on_right else "context"],
        examples["context" if pad_on_right else "question"],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

    # We keep the example_id that gave us this feature and we will store the offset mappings.
    tokenized_examples["example_id"] = []

    for i in range(len(tokenized_examples["input_ids"])):
        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)
        context_index = 1 if pad_on_right else 0

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        tokenized_examples["example_id"].append(examples["id"][sample_index])

        # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
        # position is part of the context or not.
        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_index else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]

    return tokenized_examples
```


```python
# 对验证集进行处理
validation_features = datasets["validation"].map(
    prepare_validation_features,
    batched=True,
    remove_columns=datasets["validation"].column_names
)
```


```python
# 对验证集进行预测
raw_predictions = trainer.predict(validation_features)

validation_features.set_format(type=validation_features.format["type"],
                               columns=list(validation_features.features.keys()))
```



<div>

  <progress value='674' max='674' style='width:300px; height:20px; vertical-align: middle;'></progress>
  [674/674 00:49]
</div>



### 5.3 得到验证结果


```python
max_answer_length = 30
```


```python
start_logits = output.start_logits[0].cpu().numpy()
end_logits = output.end_logits[0].cpu().numpy()
offset_mapping = validation_features[0]["offset_mapping"]
# The first feature comes from the first example. For the more general case, we will need to be match the example_id to
# an example index
context = datasets["validation"][0]["context"]
```


```python
# Gather the indices the best start/end logits:
start_indexes = np.argsort(start_logits)[-1: -n_best_size - 1: -1].tolist()
end_indexes = np.argsort(end_logits)[-1: -n_best_size - 1: -1].tolist()
valid_answers = []
for start_index in start_indexes:
    for end_index in end_indexes:
        # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
        # to part of the input_ids that are not in the context.
        if (
                start_index >= len(offset_mapping)
                or end_index >= len(offset_mapping)
                or offset_mapping[start_index] is None
                or offset_mapping[end_index] is None
        ):
            continue
        # Don't consider answers with a length that is either < 0 or > max_answer_length.
        if end_index < start_index or end_index - start_index + 1 > max_answer_length:
            continue
        if start_index <= end_index:  # We need to refine that test to check the answer is inside the context
            start_char = offset_mapping[start_index][0]
            end_char = offset_mapping[end_index][1]
            valid_answers.append(
                {
                    "score": start_logits[start_index] + end_logits[end_index],
                    "text": context[start_char: end_char]
                }
            )
```


```python
valid_answers = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[:n_best_size]
print(valid_answers)
```

    [{'score': 17.878677, 'text': 'Denver Broncos'}, {'score': 15.834391, 'text': 'Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers'}, {'score': 14.081308, 'text': 'Broncos'}, {'score': 13.745024, 'text': 'Carolina Panthers'}, {'score': 12.03702, 'text': 'Broncos defeated the National Football Conference (NFC) champion Carolina Panthers'}, {'score': 11.468799, 'text': 'Denver'}, {'score': 10.857439, 'text': 'The American Football Conference (AFC) champion Denver Broncos'}, {'score': 9.668666, 'text': 'American Football Conference (AFC) champion Denver Broncos'}, {'score': 8.859463, 'text': 'Panthers'}, {'score': 8.81315, 'text': 'The American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers'}, {'score': 8.652461, 'text': 'Denver Broncos defeated the National Football Conference (NFC) champion Carolina'}, {'score': 8.415495, 'text': 'Denver Broncos defeated the National Football Conference (NFC)'}, {'score': 8.272964, 'text': 'Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24–10'}, {'score': 7.7231836, 'text': 'Denver Broncos defeated the National Football Conference'}, {'score': 7.6243773, 'text': 'American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers'}, {'score': 7.359405, 'text': 'Denver Broncos defeated the National Football Conference (NFC'}, {'score': 6.8753233, 'text': 'Denver Broncos defeated the National Football Conference (NFC) champion'}, {'score': 6.569023, 'text': 'AFC) champion Denver Broncos'}, {'score': 6.5630946, 'text': 'Carolina'}, {'score': 6.3930545, 'text': 'Denver Broncos defeated the National Football'}]
    

### 5.4 评测指标的计算


```python
def postprocess_qa_predictions(examples, features, raw_predictions, n_best_size=20, max_answer_length=30):
    all_start_logits, all_end_logits = raw_predictions
    # Build a map example to its corresponding features.
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(
            i)

    # The dictionaries we have to fill.
    predictions = collections.OrderedDict()

    # Logging.
    print(
        f"Post-processing {len(examples)} example predictions split into {len(features)} features.")

    # Let's loop over all the examples!
    for example_index, example in enumerate(tqdm(examples)):
        # Those are the indices of the features associated to the current example.
        feature_indices = features_per_example[example_index]

        min_null_score = None  # Only used if squad_v2 is True.
        valid_answers = []

        context = example["context"]
        # Looping through all the features associated to the current example.
        for feature_index in feature_indices:
            # We grab the predictions of the model for this feature.
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            # This is what will allow us to map some the positions in our logits to span of texts in the original
            # context.
            offset_mapping = features[feature_index]["offset_mapping"]

            # Update minimum null prediction.
            cls_index = features[feature_index]["input_ids"].index(
                tokenizer.cls_token_id)
            feature_null_score = start_logits[cls_index] + \
                end_logits[cls_index]
            if min_null_score is None or min_null_score < feature_null_score:
                min_null_score = feature_null_score

            # Go through all possibilities for the `n_best_size` greater start and end logits.
            start_indexes = np.argsort(
                start_logits)[-1: -n_best_size - 1: -1].tolist()
            end_indexes = np.argsort(
                end_logits)[-1: -n_best_size - 1: -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
                    # to part of the input_ids that are not in the context.
                    if (
                            start_index >= len(offset_mapping)
                            or end_index >= len(offset_mapping)
                            or offset_mapping[start_index] is None
                            or offset_mapping[end_index] is None
                    ):
                        continue
                    # Don't consider answers with a length that is either < 0 or > max_answer_length.
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue

                    start_char = offset_mapping[start_index][0]
                    end_char = offset_mapping[end_index][1]
                    valid_answers.append(
                        {
                            "score": start_logits[start_index] + end_logits[end_index],
                            "text": context[start_char: end_char]
                        }
                    )

        if len(valid_answers) > 0:
            best_answer = sorted(
                valid_answers, key=lambda x: x["score"], reverse=True)[0]
        else:
            # In the very rare edge case we have not a single non-null prediction, we create a fake prediction to avoid
            # failure.
            best_answer = {"text": "", "score": 0.0}

        # Let's pick our final answer: the best one or the null answer (only for squad_v2)
        if not squad_v2:
            predictions[example["id"]] = best_answer["text"]
        else:
            answer = best_answer["text"] if best_answer["score"] > min_null_score else ""
            predictions[example["id"]] = answer

    return predictions
```


```python
final_predictions = postprocess_qa_predictions(
    datasets["validation"], validation_features, raw_predictions.predictions)
```

    Post-processing 10570 example predictions split into 10784 features.


```python
# 加载评测指标
metric = load_metric("squad_v2" if squad_v2 else "squad")
```


```python
if squad_v2:
    formatted_predictions = [{"id": k, "prediction_text": v, "no_answer_probability": 0.0} for k, v in
                             final_predictions.items()]
else:
    formatted_predictions = [{"id": k, "prediction_text": v}
                             for k, v in final_predictions.items()]
references = [{"id": ex["id"], "answers": ex["answers"]}
              for ex in datasets["validation"]]
metric_score = metric.compute(
    predictions=formatted_predictions, references=references)
```


```python
metric_score
```




    {'exact_match': 76.91579943235573, 'f1': 85.23761154962662}



## 6 总结

&emsp;&emsp;本次任务，主要介绍了用BERT模型解决抽取式问答任务的方法及步骤，步骤主要分为加载数据、数据预处理、微调预训练模型和模型评估。在加载数据阶段中，使用SQUAD数据集；在数据预处理阶段中，对tokenizer分词器的建模，处理长文本，并完成数据集中所有样本的预处理；在微调预训练模型阶段，通过对模型训练参数进行设置，训练并保存模型；在模型评估阶段，通过对模型预测的输出结果进行处理，解决无答案情况，最后使用squad评测方法，基于预测和标注对评测指标进行计算。  
&emsp;&emsp;其中在数据集下载时，需要使用外网方式建立代理；本次任务中的模型训练，笔者使用的是3070  GPU显卡，需要训练模型长达1小时。
