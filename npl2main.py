import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from collections import Counter, defaultdict
from tqdm import tqdm
import random
from datetime import datetime


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


# 生成带时间戳的文件名
def generate_timestamped_filename(base_name, extension):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base_name}_{timestamp}.{extension}"


# 文本数据集类
class TextDataset(Dataset):
    def __init__(self, texts, labels=None, max_len=100, vocab=None):
        self.texts = texts
        self.labels = labels
        self.max_len = max_len

        if vocab is None:
            # 构建词汇表
            self.build_vocab(texts)
        else:
            self.vocab = vocab

        # 未知词和填充词的索引
        self.unk_idx = self.vocab.get('<UNK>', 0)
        self.pad_idx = self.vocab.get('<PAD>', 1)

        # 将文本转换为索引
        self.texts_idx = [self.text_to_indices(text) for text in texts]

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
        return len(self.texts)

    def __getitem__(self, idx):
        indices = self.texts_idx[idx]
        padded_indices = self.pad_sequence(indices)

        if self.labels is not None:
            return torch.tensor(padded_indices), torch.tensor(self.labels[idx])
        else:
            return torch.tensor(padded_indices)


# CNN文本分类模型
class CNNTextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_filters, filter_sizes, output_dim, dropout=0.5):
        super(CNNTextClassifier, self).__init__()

        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # 卷积层集合
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (fs, embed_dim))
            for fs in filter_sizes
        ])

        # 全连接层和Dropout
        self.fc = nn.Linear(len(filter_sizes) * num_filters, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [batch_size, seq_len]

        # 嵌入层
        embedded = self.embedding(x)  # [batch_size, seq_len, embed_dim]

        # 添加通道维度
        embedded = embedded.unsqueeze(1)  # [batch_size, 1, seq_len, embed_dim]

        # 卷积和池化
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in
                  self.convs]  # 每个元素: [batch_size, num_filters, seq_len - filter_size + 1]

        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]  # 每个元素: [batch_size, num_filters]

        # 连接所有特征
        cat = self.dropout(torch.cat(pooled, dim=1))  # [batch_size, num_filters * len(filter_sizes)]

        # 全连接层
        return self.fc(cat)


# RNN文本分类模型
class RNNTextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout=0.5):
        super(RNNTextClassifier, self).__init__()

        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # RNN层 (可以是LSTM或GRU)
        self.rnn = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=n_layers,
            bidirectional=bidirectional,
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True
        )

        # 全连接层
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)

        # Dropout层
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [batch_size, seq_len]

        # 嵌入层
        embedded = self.dropout(self.embedding(x))  # [batch_size, seq_len, embed_dim]

        # RNN层
        output, (hidden, cell) = self.rnn(embedded)

        # 处理双向RNN的隐藏状态
        if self.rnn.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        else:
            hidden = self.dropout(hidden[-1, :, :])

        # 全连接层
        return self.fc(hidden)


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
        for i, (inputs, labels) in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            # 梯度清零
            optimizer.zero_grad()

            # 前向传播
            outputs = model(inputs)
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
            model_path = generate_timestamped_filename('best_model', 'pt')
            torch.save(model.state_dict(), model_path)
            print(f"最佳模型已保存为: {model_path}")

    return history


# 评估函数
def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return total_loss / len(data_loader), 100.0 * correct / total


# 预测函数
def predict(model, data_loader, device):
    model.eval()
    all_preds = []

    with torch.no_grad():
        for inputs in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())

    return all_preds


# 绘制训练历史
def plot_training_history(history, model_type):
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
    filename = generate_timestamped_filename(f'training_history_{model_type}', 'png')
    plt.savefig(filename)
    print(f"训练历史图表已保存为: {filename}")
    plt.close()


# 主函数
def main():
    # 创建结果目录
    if not os.path.exists('results'):
        os.makedirs('results')

    # 加载数据
    print("加载数据...")
    data_dir = "data/nlp1"
    train_path = os.path.join(data_dir, "train.tsv")
    test_path = os.path.join(data_dir, "test.tsv")

    # 读取训练数据
    train_df = pd.read_csv(train_path, sep='\t')

    # 数据清洗：确保所有文本都是字符串类型
    train_df['Phrase'] = train_df['Phrase'].astype(str)
    train_df['Phrase'] = train_df['Phrase'].fillna('')

    texts = train_df['Phrase'].tolist()
    labels = train_df['Sentiment'].tolist()

    # 划分训练集和验证集
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42)

    # 创建数据集
    max_len = 100  # 最大序列长度
    train_dataset = TextDataset(train_texts, train_labels, max_len)
    val_dataset = TextDataset(val_texts, val_labels, max_len, vocab=train_dataset.vocab)

    # 创建数据加载器
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # 模型参数
    vocab_size = len(train_dataset.vocab)
    embed_dim = 100
    output_dim = len(set(labels))
    dropout = 0.5

    # 选择模型类型
    model_type = input("请选择模型类型 (CNN/RNN): ").upper()

    if model_type == 'CNN':
        # CNN模型参数
        num_filters = 100
        filter_sizes = [3, 4, 5]

        # 初始化CNN模型
        model = CNNTextClassifier(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_filters=num_filters,
            filter_sizes=filter_sizes,
            output_dim=output_dim,
            dropout=dropout
        )

    elif model_type == 'RNN':
        # RNN模型参数
        hidden_dim = 256
        n_layers = 2
        bidirectional = True

        # 初始化RNN模型
        model = RNNTextClassifier(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            n_layers=n_layers,
            bidirectional=bidirectional,
            dropout=dropout
        )

    else:
        print("无效的模型类型选择，默认为CNN")
        model = CNNTextClassifier(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_filters=100,
            filter_sizes=[3, 4, 5],
            output_dim=output_dim,
            dropout=dropout
        )
        model_type = 'CNN'

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    print(f"开始训练{model_type}模型...")
    epochs = 10
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
    plot_training_history(history, model_type)

    # 在验证集上评估
    print("\n在验证集上评估...")
    val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
    print(f'最终验证集准确率: {val_acc:.2f}%')

    # 预测测试集
    print("\n预测测试集...")
    test_df = pd.read_csv(test_path, sep='\t')

    # 关键修复：确保测试集文本是字符串类型
    test_df['Phrase'] = test_df['Phrase'].astype(str)
    test_df['Phrase'] = test_df['Phrase'].fillna('')

    test_texts = test_df['Phrase'].tolist()
    test_dataset = TextDataset(test_texts, None, max_len, vocab=train_dataset.vocab)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # 加载最佳模型
    best_model_path = sorted([f for f in os.listdir('.') if f.startswith('best_model_') and f.endswith('.pt')])[-1]
    model.load_state_dict(torch.load(best_model_path))
    model = model.to(device)

    # 预测
    test_preds = predict(model, test_loader, device)

    # 保存预测结果
    result_filename = generate_timestamped_filename(f'predictions_{model_type}', 'csv')
    test_df['Sentiment'] = test_preds
    test_df[['PhraseId', 'Sentiment']].to_csv(os.path.join('results', result_filename), index=False)
    print(f"预测结果已保存为: results/{result_filename}")

    # 打印模型架构
    print("\n模型架构:")
    print(model)


if __name__ == "__main__":
    main()