import os
import torch
import tqdm
from torch.utils.data import DataLoader

import evaluate

from datasets import load_dataset, load_metric
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, get_scheduler
)
from peft import LoraConfig, TaskType, get_peft_model

train_dataset = load_dataset('csv', data_files='./data/train.csv')
eval_dataset = load_dataset('csv', data_files='./data/eval.csv')
test_dataset = load_dataset('csv', data_files='./data/test.csv')


# 模型
PATH = './FineTunning/Model/qwen/Qwen1___5-1___8B'

tokenizer = AutoTokenizer.from_pretrained(PATH, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(PATH, local_files_only=True, num_labels=2)

def tokenize_function(examples):
    prompt = '以下是一段文本，判断其是人类写的还是ChatGPT写的，以1代表人类写的，0代表ChatGPT写的。'
    return tokenizer([prompt + ' <文本开始> ' + ex + '<文本结束>' for ex in examples['inputs']],
                     padding=True, truncation=True, max_length=512)

train_dataset = train_dataset.map(tokenize_function, batched=True).remove_columns(['inputs'])
eval_dataset = eval_dataset.map(tokenize_function, batched=True).remove_columns(['inputs'])
test_dataset = test_dataset.map(tokenize_function, batched=True).remove_columns(['inputs'])

# train_dataset.rename_column('labels', 'label')
# eval_dataset.rename_column('labels', 'label')
# test_dataset.rename_column('labels', 'label')

# print(tokenizer.decode(train_dataset['train']['input_ids'][0]))
# # 打印数据集
# for example in train_dataset.data['train']['input_ids'][:5]:
#     print(tokenizer.decode(example))

train_dataset.set_format('torch')
eval_dataset.set_format('torch')
test_dataset.set_format('torch')

collector = DataCollatorWithPadding(tokenizer)

train_loader = DataLoader(train_dataset['train'].select(range(500)), batch_size=4, shuffle=True)
eval_loader = DataLoader(eval_dataset['train'].select(range(100)), batch_size=4)
test_loader = DataLoader(test_dataset['train'].select(range(100)), batch_size=4)

# for batch in train_loader:
#     print({k: v.shape for k, v in batch.items()})
#     batch = {k: v.to(device) for k, v in batch.items()}
#     output = model(**batch)
#     print(output.loss, output.logits.shape)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=['q_proj', 'v_proj']
)

model = get_peft_model(model, lora_config)

# 优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
num_epochs = 3
num_training_steps = num_epochs * len(train_loader)

lr_scheduler = get_scheduler(
    name="linear", 
    optimizer=optimizer, 
    num_warmup_steps=0, 
    num_training_steps=num_training_steps
)


# 训练
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)
model.to(device)
model.config.pad_token_id = tokenizer.pad_token_id

# 先测试未训练的模型
# 测试
print("*************Testing untrained model************")
model.eval()
metric = evaluate.load("./metrics/accuracy.py")
test_loss = 0
for batch in test_loader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)
        test_loss += outputs.loss.item()
    predictions = torch.argmax(outputs.logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])

acc = metric.compute()
print(f"Test accuracy: {acc}")
print(f"Test loss: {test_loss / len(test_loader)}")


# 开始训练
print("*************Training************")

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    progress = tqdm.trange(len(train_loader))
    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        # print(outputs.loss, outputs.logits.shape)
        loss = outputs.loss
        loss.backward()
        
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        
        epoch_loss += loss.item()
        
        progress.set_description(f"Epoch {epoch}")
        progress.update(1)
        
    
    print(f"Epoch {epoch} loss: {epoch_loss / len(train_loader)}")
    
    model.eval()
    metric = evaluate.load("./metrics/accuracy.py")
    eval_loss = 0
    for batch in eval_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
            eval_loss += outputs.loss.item()
        predictions = torch.argmax(outputs.logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])
    
    acc = metric.compute()
    print(f"Epoch {epoch} eval accuracy: {acc}")
    print(f"Epoch {epoch} eval loss: {eval_loss / len(eval_loader)}")
    
# 测试
model.eval()
metric = evaluate.load("./metrics/accuracy.py")
test_loss = 0
for batch in test_loader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)
        test_loss += outputs.loss.item()
    predictions = torch.argmax(outputs.logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])

acc = metric.compute()
print(f"Test accuracy: {acc}")
print(f"Test loss: {test_loss / len(test_loader)}")
        
