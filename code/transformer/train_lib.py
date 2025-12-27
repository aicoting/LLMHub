'''
Author: zhangting
Date: 2025-06-16 21:43:35
LastEditors: Do not edit
LastEditTime: 2025-06-17 18:48:54
FilePath: /zhangting/LLMHub/code/transformer/train_lib.py
'''
# train_agnews.py

import math, time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchtext.datasets import AG_NEWS
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

# 数据加载与预处理
tokenizer = get_tokenizer('basic_english')

def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)

train_iter = AG_NEWS(split='train')
vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=['<unk>', '<pad>'])
vocab.set_default_index(vocab['<unk>'])

# 标准分类 pipeline
text_pipeline = lambda x: vocab(tokenizer(x))
label_pipeline = lambda x: int(x) - 1  # 标签 0–3

def collate_batch(batch):
    texts, labels = [], []
    for label, text in batch:
        labels.append(label_pipeline(label))
        texts.append(torch.tensor(text_pipeline(text), dtype=torch.long))
    texts = nn.utils.rnn.pad_sequence(texts, padding_value=vocab['<pad>'], batch_first=True)
    return texts, torch.tensor(labels, dtype=torch.long)

# 模型定义：很基础的 Transformer 分析器
class SimpleClassifier(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=8, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.classifier = nn.Linear(d_model, 4)

    def forward(self, x):
        x = self.embedding(x).transpose(0, 1)  # seq-first
        out = self.transformer(x)               # shape: (seq_len, batch, d_model)
        out = out.mean(dim=0)                   # average pooling
        return self.classifier(out)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 训练流程
def train_epoch(loader, model, criterion, optimizer):
    model.train()
    total_loss = 0
    for texts, labels in loader:
        texts, labels = texts.to(device), labels.to(device)
        optimizer.zero_grad()
        preds = model(texts)
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def eval_epoch(loader, model, criterion):
    model.eval()
    total_loss, total_acc, count = 0, 0, 0
    with torch.no_grad():
        for texts, labels in loader:
            texts, labels = texts.to(device), labels.to(device)
            preds = model(texts)
            total_loss += criterion(preds, labels).item()
            total_acc += (preds.argmax(1) == labels).sum().item()
            count += labels.size(0)
    return total_loss / len(loader), total_acc / count

# 主训练逻辑
if __name__=='__main__':
    BATCH = 64
    train_dataset = list(AG_NEWS(split='train'))
    test_dataset = list(AG_NEWS(split='test'))

    train_loader = DataLoader(train_dataset, batch_size=BATCH, shuffle=True, collate_fn=collate_batch)
    test_loader = DataLoader(test_dataset, batch_size=BATCH, shuffle=False, collate_fn=collate_batch)

    model = SimpleClassifier(len(vocab)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, 6):
        t0 = time.time()
        train_loss = train_epoch(train_loader, model, criterion, optimizer)
        test_loss, test_acc = eval_epoch(test_loader, model, criterion)
        print(f"Epoch {epoch}: Train L={train_loss:.4f} | Test L={test_loss:.4f}, Acc={test_acc:.4f} | {time.time()-t0:.1f}s")
