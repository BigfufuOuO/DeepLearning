import random
from datasets import load_dataset
import pandas as pd

dataset = load_dataset(path='./simpleai/HC3-Chinese/simpleai/HC3-Chinese/master/meta/HC3-Chinese.py', name='all')


dataset = dataset["train"].shuffle(seed=42).select(range(5000))
# 取出其中的两列。
dataset = dataset.select_columns(['question', 'human_answers', 'chatgpt_answers'])
print(dataset)

def process_dataset(examples):
    inputs = []
    labels = []
    for human, gpt in zip(examples["human_answers"], examples["chatgpt_answers"]):
        if random.random() < 0.5:
            inputs.append(human)
            labels.append(1)
        else:
            inputs.append(gpt)
            labels.append(0)
            
    return {"inputs": inputs, "labels": labels}

dataset = dataset.map(process_dataset, batched=True, batch_size=1)
print(dataset)

# 按照8:1:1
train_dataset = dataset.select(range(4000))
eval_dataset = dataset.select(range(4000, 4500))
test_dataset = dataset.select(range(4500, 5000))

train_df = pd.DataFrame({
    'inputs': train_dataset["inputs"],
    'labels': train_dataset["labels"]
})

eval_df = pd.DataFrame({
    'inputs': eval_dataset["inputs"],
    'labels': eval_dataset["labels"]
})

test_df = pd.DataFrame({
    'inputs': test_dataset["inputs"],
    'labels': test_dataset["labels"]
})

train_df.to_csv('./data/train.csv', index=False)
eval_df.to_csv('./data/eval.csv', index=False)
test_df.to_csv('./data/test.csv', index=False)
