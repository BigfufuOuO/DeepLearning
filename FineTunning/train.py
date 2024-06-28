import os
import torch
from torch.utils.data import DataLoader

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler
from peft import LoraConfig, TaskType, get_peft_model

dataset = load_dataset(path='./simpleai/HC3-Chinese/master/meta/HC3-Chinese.py', name='all')

dataset = dataset["train"].shuffle(seed=42).select(range(5000))
# 取出其中的两列。
dataset = dataset.select_columns(['human_answers', 'source'])
print(dataset)



train_dataset = dataset.select(range(4000))
eval_dataset = dataset.select(range(4000, 4500))
test_dataset = dataset.select(range(4500, 5000))

# 模型
PATH = 'FineTunning/Model/FineTunning/Model/qwen/Qwen1___5-1___8B'

tokenizer = AutoTokenizer.from_pretrained(PATH, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(PATH, local_files_only=True)

# 预处理
def preprocess_function(examples):
    prompt = "以下文本是基于问题得到的回答，其中的文本归属为几个类别，可供选择的类别有：\
                1.open_qa;\n \
                2.baike;\n \
                3.nlpcc_dbqa;\n \
                4.medical;\n \
                5.finance;\n \
                6.psychology;\n \
                7.law;\n \
                请你根据文本内容选择一个类别。\
            "
    return tokenizer([prompt + ex for ex in examples["human_answers"]], 
                     truncation=True, padding="max_length", max_length=256)

encoded_train_dataset = train_dataset.map(preprocess_function, batched=True)
encoded_eval_dataset = eval_dataset.map(preprocess_function, batched=True)
encoded_test_dataset = test_dataset.map(preprocess_function, batched=True)

train_loader = DataLoader(encoded_train_dataset, batch_size=8, shuffle=True)
eval_loader = DataLoader(encoded_eval_dataset, batch_size=8)
test_loader = DataLoader(encoded_test_dataset, batch_size=8)

lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
)

model = get_peft_model(model, lora_config)

# 优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
num_epochs = 3
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

# 训练
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        
        epoch_loss += loss.item()
    
    print(f"Epoch {epoch} loss: {epoch_loss / len(train_loader)}")
    
    model.eval()
    eval_loss = 0
    for batch in eval_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
            eval_loss += outputs.loss.item()
        
    print(f"Epoch {epoch} eval loss: {eval_loss / len(eval_loader)}")
    
# 测试
model.eval()
test_loss = 0
for batch in test_loader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)
        test_loss += outputs.loss.item()
            
        
