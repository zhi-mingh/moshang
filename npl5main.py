import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import time
import torch.nn.functional as F
from tqdm import tqdm

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


# 读取唐诗数据集
def load_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text


# 字符级语言模型数据集
class PoetryDataset(Dataset):
    def __init__(self, text, seq_length, char_to_idx, idx_to_char):
        self.text = text
        self.seq_length = seq_length
        self.char_to_idx = char_to_idx
        self.idx_to_char = idx_to_char
        self.vocab_size = len(char_to_idx)

        # 将文本转换为索引
        self.text_idx = [char_to_idx[c] for c in text]

        # 计算样本数量
        self.num_samples = len(self.text_idx) - seq_length

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 获取输入序列和目标序列
        inputs = torch.tensor(self.text_idx[idx:idx + self.seq_length], dtype=torch.long)
        targets = torch.tensor(self.text_idx[idx + 1:idx + self.seq_length + 1], dtype=torch.long)
        return inputs, targets


# 字符级语言模型
class CharLM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout=0.5, rnn_type='lstm'):
        super(CharLM, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn_type = rnn_type.lower()

        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # RNN层 (LSTM或GRU)
        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(
                embed_dim,
                hidden_dim,
                num_layers,
                dropout=dropout,
                batch_first=True
            )
        elif self.rnn_type == 'gru':
            self.rnn = nn.GRU(
                embed_dim,
                hidden_dim,
                num_layers,
                dropout=dropout,
                batch_first=True
            )
        else:
            raise ValueError("rnn_type must be 'lstm' or 'gru'")

        # 输出层
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        # x: [batch_size, seq_length]

        # 嵌入层
        embedded = self.embedding(x)  # [batch_size, seq_length, embed_dim]

        # RNN层
        if self.rnn_type == 'lstm':
            output, (h_n, c_n) = self.rnn(embedded, hidden)
        else:
            output, h_n = self.rnn(embedded, hidden)

        # 输出层
        logits = self.fc(output)  # [batch_size, seq_length, vocab_size]

        return logits, (h_n, c_n) if self.rnn_type == 'lstm' else h_n

    def init_hidden(self, batch_size):
        # 初始化隐藏状态
        weight = next(self.parameters())
        if self.rnn_type == 'lstm':
            return (weight.new_zeros(self.num_layers, batch_size, self.hidden_dim),
                    weight.new_zeros(self.num_layers, batch_size, self.hidden_dim))
        else:
            return weight.new_zeros(self.num_layers, batch_size, self.hidden_dim)


# 训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, device, clip=0.25):
    model = model.to(device)
    best_val_loss = float('inf')
    history = {'train_loss': [], 'train_ppl': [], 'val_loss': [], 'val_ppl': []}

    for epoch in range(epochs):
        start_time = time.time()

        # 训练模式
        model.train()
        train_loss = 0.0
        train_ppl = 0.0

        # 使用tqdm显示进度条
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
        for i, (inputs, targets) in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)

            # 梯度清零
            optimizer.zero_grad()

            # 初始化隐藏状态
            hidden = model.init_hidden(inputs.size(0))

            # 前向传播
            outputs, _ = model(inputs, hidden)

            # 计算损失
            loss = criterion(outputs.view(-1, model.vocab_size), targets.view(-1))

            # 反向传播
            loss.backward()

            # 梯度裁剪，防止梯度爆炸
            nn.utils.clip_grad_norm_(model.parameters(), clip)

            # 更新参数
            optimizer.step()

            # 统计
            train_loss += loss.item()
            train_ppl += torch.exp(loss).item()

            # 更新进度条
            progress_bar.set_description(
                f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}, PPL: {torch.exp(loss).item():.2f}')

        # 计算平均训练损失和困惑度
        train_loss = train_loss / len(train_loader)
        train_ppl = train_ppl / len(train_loader)

        # 验证
        val_loss, val_ppl = evaluate_model(model, val_loader, criterion, device)

        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_ppl'].append(train_ppl)
        history['val_loss'].append(val_loss)
        history['val_ppl'].append(val_ppl)

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_type = 'lstm' if model.rnn_type == 'lstm' else 'gru'
            torch.save(model.state_dict(), f'best_{model_type}_model.pt')
            print(f"最佳模型已保存，验证困惑度: {val_ppl:.2f}")

        # 打印本轮结果
        end_time = time.time()
        epoch_time = end_time - start_time
        print(f'Epoch {epoch + 1}/{epochs} | Time: {epoch_time:.2f}s | '
              f'Train Loss: {train_loss:.4f} | Train PPL: {train_ppl:.2f} | '
              f'Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f}')

    return history


# 评估函数
def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_ppl = 0.0

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # 初始化隐藏状态
            hidden = model.init_hidden(inputs.size(0))

            # 前向传播
            outputs, _ = model(inputs, hidden)

            # 计算损失
            loss = criterion(outputs.view(-1, model.vocab_size), targets.view(-1))

            # 统计
            total_loss += loss.item()
            total_ppl += torch.exp(loss).item()

    # 计算平均损失和困惑度
    avg_loss = total_loss / len(data_loader)
    avg_ppl = total_ppl / len(data_loader)

    return avg_loss, avg_ppl


# 生成文本函数
def generate_text(model, char_to_idx, idx_to_char, start_text, length=200, temperature=1.0, device=device):
    model.eval()

    # 将起始文本转换为索引
    inputs = torch.tensor([char_to_idx[c] for c in start_text], dtype=torch.long).unsqueeze(0).to(device)

    # 生成的文本
    generated_text = start_text

    # 初始化隐藏状态
    hidden = model.init_hidden(1)

    with torch.no_grad():
        for i in range(length):
            # 前向传播
            outputs, hidden = model(inputs, hidden)

            # 获取最后一个时间步的输出
            logits = outputs[:, -1, :]

            # 应用温度参数
            logits = logits / temperature

            # 计算概率分布
            probs = F.softmax(logits, dim=1)

            # 采样下一个字符
            next_idx = torch.multinomial(probs, num_samples=1).item()
            next_char = idx_to_char[next_idx]

            # 添加到生成的文本中
            generated_text += next_char

            # 准备下一个输入
            inputs = torch.tensor([[next_idx]]).to(device)

    return generated_text


# 绘制训练历史
def plot_training_history(history, model_type):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{model_type.upper()} Training and Validation Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_ppl'], label='Train Perplexity')
    plt.plot(history['val_ppl'], label='Validation Perplexity')
    plt.xlabel('Epoch')
    plt.ylabel('Perplexity')
    plt.title(f'{model_type.upper()} Training and Validation Perplexity')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'{model_type}_training_history.png')
    plt.close()


# 主函数
def main():
    # 加载数据
    print("加载数据...")
    file_path = r"data\nlp5\poetryFromTang.txt"
    text = load_dataset(file_path)

    # 构建字符到索引和索引到字符的映射
    chars = sorted(list(set(text)))
    char_to_idx = {c: i for i, c in enumerate(chars)}
    idx_to_char = {i: c for i, c in enumerate(chars)}
    vocab_size = len(chars)

    print(f"总字符数: {len(text)}")
    print(f"词汇表大小: {vocab_size}")

    # 划分训练集和验证集
    train_size = int(0.9 * len(text))
    train_text = text[:train_size]
    val_text = text[train_size:]

    # 创建数据集和数据加载器
    seq_length = 100
    batch_size = 64

    train_dataset = PoetryDataset(train_text, seq_length, char_to_idx, idx_to_char)
    val_dataset = PoetryDataset(val_text, seq_length, char_to_idx, idx_to_char)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # 模型参数
    embed_dim = 256
    hidden_dim = 512
    num_layers = 2
    dropout = 0.5
    learning_rate = 0.001
    epochs = 20

    # 选择模型类型
    model_type = input("请选择模型类型 (LSTM/GRU): ").upper()

    # 初始化模型
    if model_type == 'LSTM':
        model = CharLM(vocab_size, embed_dim, hidden_dim, num_layers, dropout, 'lstm')
    elif model_type == 'GRU':
        model = CharLM(vocab_size, embed_dim, hidden_dim, num_layers, dropout, 'gru')
    else:
        print("无效的模型类型选择，默认为LSTM")
        model = CharLM(vocab_size, embed_dim, hidden_dim, num_layers, dropout, 'lstm')
        model_type = 'LSTM'

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练模型
    print(f"开始训练{model_type}模型...")
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
    plot_training_history(history, model_type.lower())
    print(f"{model_type}训练历史图表已保存")

    # 加载最佳模型
    model.load_state_dict(torch.load(f'best_{model_type.lower()}_model.pt'))
    model = model.to(device)

    # 在验证集上计算最终困惑度
    _, val_ppl = evaluate_model(model, val_loader, criterion, device)
    print(f"{model_type}模型最终验证困惑度: {val_ppl:.2f}")

    # 生成文本
    start_text = input("请输入生成文本的起始字符: ")
    if not start_text:
        start_text = random.choice(chars)

    generated_text = generate_text(
        model=model,
        char_to_idx=char_to_idx,
        idx_to_char=idx_to_char,
        start_text=start_text,
        length=300,
        temperature=0.7
    )

    print("\n生成的文本:")
    print("=" * 50)
    print(generated_text)
    print("=" * 50)

    # 保存生成的文本
    with open(f'{model_type.lower()}_generated_text.txt', 'w', encoding='utf-8') as f:
        f.write(generated_text)

    print(f"生成的文本已保存为 {model_type.lower()}_generated_text.txt")


if __name__ == "__main__":
    main()