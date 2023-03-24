import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup


# 1. 读取数据
# 自行准备数据，格式为：(text, label)，其中，label为0（非优质店铺）或1（优质店铺）

# 2. 数据预处理
class ShopDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, label = self.data[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
max_length = 128
train_dataset = ShopDataset(train_data, tokenizer, max_length)
val_dataset = ShopDataset(val_data, tokenizer, max_length)

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# 3. 加载预训练的BERT模型
model = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=2)
model = model.to('cuda')

# 4. 微调模型
optimizer = AdamW(model.parameters(), lr=2e-5)
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)


# 5. 添加全连接层进行分类
# BERT模型中已经包含了一个分类器，我们只需要微调即可

# 6. 训练模型
def train_epoch(model, data_loader, optimizer, scheduler, device):
    model.train()
    for batch in data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        # 7. 评估模型

    def eval_epoch(model, data_loader, device):
        model.eval()
        total_acc = 0
        total_count = 0

        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                preds = torch.argmax(outputs.logits, dim=1)
                total_acc += (preds == labels).sum().item()
                total_count += labels.size(0)

        return total_acc / total_count

    epochs = 4
    device = 'cuda'

    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')
        train_epoch(model, train_dataloader, optimizer, scheduler, device)
        acc = eval_epoch(model, val_dataloader, device)
        print(f'Validation accuracy: {acc:.4f}')
