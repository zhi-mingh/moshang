import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from collections import Counter, defaultdict
from tqdm import tqdm
import random
import json


# 设置随机种子确保结果可复现
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(42)

# 检测GPU可用性
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


# 加载SNLI数据集
def load_snli_data(data_dir, split='train'):
    data_path = os.path.join(data_dir, f'snli_1.0_{split}.jsonl')
    premises = []
    hypotheses = []
    labels = []

    label_map = {'entailment': 0, 'contradiction': 1, 'neutral': 2}

    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            example = json.loads(line)
            label = example['gold_label']

            # 忽略标签为'-'的样本
            if label == '-':
                continue

            premise = example['sentence1']
            hypothesis = example['sentence2']

            premises.append(premise)
            hypotheses.append(hypothesis)
            labels.append(label_map[label])

    return premises, hypotheses, labels


# 文本预处理
def preprocess_text(text):
    # 简单的文本预处理：转为小写并移除特殊字符
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text


# SNLI数据集类
class SNLIDataset(Dataset):
    def __init__(self, premises, hypotheses, labels=None, max_len=50, vocab=None):
        self.premises = [preprocess_text(p) for p in premises]
        self.hypotheses = [preprocess_text(h) for h in hypotheses]
        self.labels = labels
        self.max_len = max_len

        if vocab is None:
            # 构建词汇表
            all_texts = self.premises + self.hypotheses
            self.build_vocab(all_texts)
        else:
            self.vocab = vocab

        # 未知词和填充词的索引
        self.unk_idx = self.vocab.get('<UNK>', 0)
        self.pad_idx = self.vocab.get('<PAD>', 1)

        # 将文本转换为索引
        self.premises_idx = [self.text_to_indices(p) for p in self.premises]
        self.hypotheses_idx = [self.text_to_indices(h) for h in self.hypotheses]

    def build_vocab(self, texts, min_freq=5):
        # 统计词频
        word_freq = defaultdict(int)
        for text in texts:
            for word in text.split():
                word_freq[word] += 1

        # 构建词汇表
        self.vocab = {'<UNK>': 0, '<PAD>': 1}
        for word, freq in word_freq.items():
            if freq >= min_freq:
                self.vocab[word] = len(self.vocab)

    def text_to_indices(self, text):
        # 将文本转换为索引序列
        return [self.vocab.get(word, self.unk_idx) for word in text.split()][:self.max_len]

    def pad_sequence(self, indices):
        # 填充序列到固定长度
        padding = [self.pad_idx] * (self.max_len - len(indices))
        return indices + padding

    def __len__(self):
        return len(self.premises)

    def __getitem__(self, idx):
        premise_indices = self.premises_idx[idx]
        hypothesis_indices = self.hypotheses_idx[idx]

        premise_padded = self.pad_sequence(premise_indices)
        hypothesis_padded = self.pad_sequence(hypothesis_indices)

        if self.labels is not None:
            return (torch.tensor(premise_padded),
                    torch.tensor(hypothesis_padded),
                    torch.tensor(self.labels[idx]))
        else:
            return (torch.tensor(premise_padded),
                    torch.tensor(hypothesis_padded))


# 注意力机制的文本匹配模型（简化版ESIM）
class ESIM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, dropout=0.5):
        super(ESIM, self).__init__()

        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=1)

        # 输入编码层 - 双向LSTM
        self.input_encoder = nn.LSTM(
            embed_dim,
            hidden_dim,
            bidirectional=True,
            batch_first=True
        )

        # 推理组合层 - 双向LSTM
        self.composition = nn.LSTM(
            hidden_dim * 8,  # 输入特征维度：增强后的表示
            hidden_dim,
            bidirectional=True,
            batch_first=True
        )

        # 池化后的全连接层
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 8, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def soft_attention_align(self, premise, hypothesis, mask_p, mask_h):
        """
        双向注意力对齐
        """
        # 计算相似度矩阵
        attention = torch.bmm(premise, hypothesis.transpose(1, 2))  # [batch_size, seq_len_p, seq_len_h]

        # 应用掩码
        mask_p = mask_p.unsqueeze(2)  # [batch_size, seq_len_p, 1]
        mask_h = mask_h.unsqueeze(1)  # [batch_size, 1, seq_len_h]

        attention_p = attention.masked_fill(mask_h == 0, -1e9)  # 对hypothesis的掩码
        attention_h = attention.masked_fill(mask_p == 0, -1e9)  # 对premise的掩码

        # 计算注意力权重
        weight_p = F.softmax(attention_p, dim=2)  # [batch_size, seq_len_p, seq_len_h]
        weight_h = F.softmax(attention_h, dim=1)  # [batch_size, seq_len_p, seq_len_h]

        # 计算对齐后的向量
        aligned_h = torch.bmm(weight_p, hypothesis)  # [batch_size, seq_len_p, hidden_dim*2]
        aligned_p = torch.bmm(weight_h.transpose(1, 2), premise)  # [batch_size, seq_len_h, hidden_dim*2]

        return aligned_p, aligned_h

    def submul(self, x1, x2):
        """
        计算差和积
        """
        sub = x1 - x2
        mul = x1 * x2
        return torch.cat([x1, x2, sub, mul], dim=-1)

    def apply_multiple(self, x):
        """
        应用平均池化和最大池化
        """
        # 平均池化
        avg_pool = F.avg_pool1d(x.transpose(1, 2), x.size(1)).squeeze(-1)
        # 最大池化
        max_pool = F.max_pool1d(x.transpose(1, 2), x.size(1)).squeeze(-1)

        return torch.cat([avg_pool, max_pool], dim=1)

    def forward(self, premise, hypothesis):
        # 创建掩码
        mask_p = premise != 1  # [batch_size, seq_len_p]
        mask_h = hypothesis != 1  # [batch_size, seq_len_h]

        # 嵌入层
        premise_embed = self.embedding(premise)  # [batch_size, seq_len_p, embed_dim]
        hypothesis_embed = self.embedding(hypothesis)  # [batch_size, seq_len_h, embed_dim]

        # 输入编码
        premise_encoded, _ = self.input_encoder(premise_embed)  # [batch_size, seq_len_p, hidden_dim*2]
        hypothesis_encoded, _ = self.input_encoder(hypothesis_embed)  # [batch_size, seq_len_h, hidden_dim*2]

        # 双向注意力对齐
        aligned_p, aligned_h = self.soft_attention_align(
            premise_encoded, hypothesis_encoded, mask_p, mask_h
        )

        # 增强表示
        premise_enhanced = self.submul(premise_encoded, aligned_h)  # [batch_size, seq_len_p, hidden_dim*8]
        hypothesis_enhanced = self.submul(hypothesis_encoded, aligned_p)  # [batch_size, seq_len_h, hidden_dim*8]

        # 组合层
        premise_composed, _ = self.composition(premise_enhanced)  # [batch_size, seq_len_p, hidden_dim*2]
        hypothesis_composed, _ = self.composition(hypothesis_enhanced)  # [batch_size, seq_len_h, hidden_dim*2]

        # 池化
        premise_pooled = self.apply_multiple(premise_composed)  # [batch_size, hidden_dim*4]
        hypothesis_pooled = self.apply_multiple(hypothesis_composed)  # [batch_size, hidden_dim*4]

        # 合并特征
        combined = torch.cat([premise_pooled, hypothesis_pooled], dim=1)  # [batch_size, hidden_dim*8]

        # 分类器
        logits = self.fc(combined)

        return logits


# 训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, device):
    model = model.to(device)
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(epochs):
        # 训练模式
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        # 使用tqdm显示进度条
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
        for i, (premise, hypothesis, labels) in progress_bar:
            premise, hypothesis, labels = premise.to(device), hypothesis.to(device), labels.to(device)

            # 梯度清零
            optimizer.zero_grad()

            # 前向传播
            outputs = model(premise, hypothesis)
            loss = criterion(outputs, labels)

            # 反向传播
            loss.backward()
            optimizer.step()

            # 统计
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

            # 更新进度条
            progress_bar.set_description(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}')

        # 计算训练指标
        train_acc = 100.0 * train_correct / train_total
        train_loss = train_loss / len(train_loader)

        # 验证
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)

        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_esim_model.pt')
            print(f"最佳模型已保存，验证准确率: {val_acc:.2f}%")

    return history


# 评估函数
def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for premise, hypothesis, labels in data_loader:
            premise, hypothesis, labels = premise.to(device), hypothesis.to(device), labels.to(device)

            outputs = model(premise, hypothesis)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return total_loss / len(data_loader), 100.0 * correct / total


# 预测函数 - 修改版本
def predict(model, data_loader, device):
    model.eval()
    all_preds = []

    with torch.no_grad():
        # 修改这里以处理三元组输入
        for batch in data_loader:
            # 检查输入的长度
            if len(batch) == 3:
                premise, hypothesis, _ = batch  # 忽略标签
            else:
                premise, hypothesis = batch

            premise, hypothesis = premise.to(device), hypothesis.to(device)

            outputs = model(premise, hypothesis)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())

    return all_preds


# 绘制训练历史
def plot_training_history(history):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig('esim_training_history.png')
    plt.close()


# 主函数
def main():
    # 加载数据
    print("加载数据...")
    data_dir = r"data\nlp3\snli_1.0"

    # 加载训练集
    train_premises, train_hypotheses, train_labels = load_snli_data(data_dir, 'train')

    # 加载验证集
    val_premises, val_hypotheses, val_labels = load_snli_data(data_dir, 'dev')

    # 创建数据集
    max_len = 50  # 最大序列长度
    train_dataset = SNLIDataset(train_premises, train_hypotheses, train_labels, max_len)
    val_dataset = SNLIDataset(val_premises, val_hypotheses, val_labels, max_len, vocab=train_dataset.vocab)

    # 创建数据加载器
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # 模型参数
    vocab_size = len(train_dataset.vocab)
    embed_dim = 300
    hidden_dim = 300
    num_classes = 3  # entailment, contradiction, neutral
    dropout = 0.5

    # 初始化模型
    model = ESIM(vocab_size, embed_dim, hidden_dim, num_classes, dropout)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0004)

    # 训练模型
    print("开始训练模型...")
    epochs = 5
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        epochs=epochs,
        device=device
    )

    # 绘制训练历史
    plot_training_history(history)
    print("训练历史图表已保存为 esim_training_history.png")

    # 在验证集上评估
    print("\n在验证集上评估...")
    val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
    print(f'最终验证集准确率: {val_acc:.2f}%')

    # 加载测试集并预测
    print("\n加载测试集并进行预测...")
    test_premises, test_hypotheses, test_labels = load_snli_data(data_dir, 'test')
    test_dataset = SNLIDataset(test_premises, test_hypotheses, test_labels, max_len, vocab=train_dataset.vocab)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # 加载最佳模型
    model.load_state_dict(torch.load('best_esim_model.pt'))
    model = model.to(device)

    # 预测
    test_preds = predict(model, test_loader, device)

    # 计算测试集准确率
    test_correct = sum([1 for pred, true in zip(test_preds, test_labels) if pred == true])
    test_acc = 100.0 * test_correct / len(test_labels)
    print(f'测试集准确率: {test_acc:.2f}%')

    # 保存预测结果
    label_map = {0: 'entailment', 1: 'contradiction', 2: 'neutral'}
    with open('esim_predictions.txt', 'w', encoding='utf-8') as f:
        for i in range(len(test_premises)):
            f.write(f"Premise: {test_premises[i]}\n")
            f.write(f"Hypothesis: {test_hypotheses[i]}\n")
            f.write(f"True Label: {label_map[test_labels[i]]}\n")
            f.write(f"Predicted Label: {label_map[test_preds[i]]}\n")
            f.write("-" * 50 + "\n")

    print("预测结果已保存为 esim_predictions.txt")


if __name__ == "__main__":
    main()