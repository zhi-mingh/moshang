import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import random
import time
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from tqdm import tqdm

# 设置中文字体，确保可视化正常显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
font = FontProperties(family="SimHei", size=10)


# 设置随机种子确保结果可复现
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


set_seed(42)

# 强制使用CPU
device = torch.device("cpu")
print(f"使用设备: {device}")


# 加载CONLL 2003数据集
def load_conll_data(data_dir, split='train'):
    data_path = os.path.join(data_dir, f'{split}.txt')
    sentences = []
    labels = []

    current_sentence = []
    current_labels = []

    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('-DOCSTART-') or line == '':
                if current_sentence:
                    sentences.append(current_sentence)
                    labels.append(current_labels)
                    current_sentence = []
                    current_labels = []
                continue

            parts = line.split()
            if len(parts) >= 4:
                word = parts[0]
                ner_tag = parts[3]
                current_sentence.append(word)
                current_labels.append(ner_tag)

    if current_sentence:
        sentences.append(current_sentence)
        labels.append(current_labels)

    return sentences, labels


# 构建词汇表
def build_vocab(sentences, labels, min_freq=2):
    word_freq = {}
    for sentence in sentences:
        for word in sentence:
            word_freq[word] = word_freq.get(word, 0) + 1

    word_to_idx = {'<PAD>': 0, '<UNK>': 1}
    for word, freq in word_freq.items():
        if freq >= min_freq:
            word_to_idx[word] = len(word_to_idx)

    label_to_idx = {'<PAD>': 0}
    all_labels = []
    for label_sequence in labels:
        all_labels.extend(label_sequence)

    unique_labels = sorted(list(set(all_labels)))
    print(f"数据集中的唯一标签: {unique_labels}")

    for label in unique_labels:
        if label not in label_to_idx:
            label_to_idx[label] = len(label_to_idx)

    print("标签到索引的映射:")
    for label, idx in label_to_idx.items():
        print(f"{label}: {idx}")

    return word_to_idx, label_to_idx


# 验证数据集中的标签索引
def validate_data(dataset, max_label_idx):
    invalid_samples = []
    for i, (_, labels_idx) in enumerate(dataset):
        for idx in labels_idx:
            if idx < 0 or idx > max_label_idx:
                invalid_samples.append(i)
                break

    if invalid_samples:
        print(f"发现{len(invalid_samples)}个包含非法标签索引的样本")
        print(f"前5个非法样本索引: {invalid_samples[:5]}")
    else:
        print("所有样本的标签索引都合法")

    return invalid_samples


# NER数据集类
class NERDataset(Dataset):
    def __init__(self, sentences, labels, word_to_idx, label_to_idx, max_len=None):
        self.sentences = sentences
        self.labels = labels
        self.word_to_idx = word_to_idx
        self.label_to_idx = label_to_idx
        self.max_len = max_len

        self.sentences_idx = []
        self.labels_idx = []

        for sentence, label_sequence in zip(sentences, labels):
            sentence_idx = [word_to_idx.get(word, word_to_idx['<UNK>']) for word in sentence]
            label_idx = [label_to_idx[label] for label in label_sequence]

            if max_len is not None:
                sentence_idx = sentence_idx[:max_len]
                label_idx = label_idx[:max_len]

            self.sentences_idx.append(sentence_idx)
            self.labels_idx.append(label_idx)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences_idx[idx], self.labels_idx[idx]


# 填充批次函数
def collate_fn(batch):
    max_len = max(len(sentence) for sentence, _ in batch)

    sentences_padded = []
    labels_padded = []
    masks = []

    for sentence, labels in batch:
        padded_sentence = sentence + [0] * (max_len - len(sentence))
        sentences_padded.append(padded_sentence)

        padded_labels = labels + [0] * (max_len - len(labels))
        labels_padded.append(padded_labels)

        mask = [1] * len(sentence) + [0] * (max_len - len(sentence))
        masks.append(mask)

    return (torch.tensor(sentences_padded, dtype=torch.long),
            torch.tensor(labels_padded, dtype=torch.long),
            torch.tensor(masks, dtype=torch.bool))


# CRF层实现
class CRF(nn.Module):
    def __init__(self, num_tags):
        super(CRF, self).__init__()
        self.num_tags = num_tags
        self.start_tag = num_tags
        self.end_tag = num_tags + 1
        self.transitions = nn.Parameter(torch.randn(num_tags + 2, num_tags + 2))
        self.init_transitions()

    def init_transitions(self):
        self.transitions.data[self.end_tag, :] = -10000.0
        self.transitions.data[:, self.start_tag] = -10000.0
        self.transitions.data[self.end_tag, self.start_tag] = -10000.0
        self.transitions.data[self.start_tag, self.end_tag] = -10000.0

    def forward(self, emissions, masks, labels):
        gold_score = self._compute_gold_score(emissions, masks, labels)
        forward_score = self._compute_forward_score(emissions, masks)
        return gold_score - forward_score

    def _compute_gold_score(self, emissions, masks, labels):
        batch_size, seq_len, num_tags = emissions.size()
        score = torch.zeros(batch_size, device=emissions.device)
        current_tags = torch.full((batch_size,), self.start_tag, dtype=torch.long, device=emissions.device)

        for i in range(seq_len):
            mask = masks[:, i]
            current_labels = labels[:, i].clamp(0, self.num_tags - 1)
            emit_score = emissions[:, i].gather(1, current_labels.unsqueeze(1)).squeeze(1)
            trans_score = self.transitions[
                current_tags.clamp(0, self.num_tags + 1),
                current_labels.clamp(0, self.num_tags + 1)
            ]
            score += (emit_score + trans_score) * mask.float()
            current_tags = torch.where(mask, current_labels, current_tags)

        trans_score = self.transitions[
            current_tags.clamp(0, self.num_tags + 1),
            self.end_tag
        ]
        score += trans_score
        return score

    def _compute_forward_score(self, emissions, masks):
        batch_size, seq_len, num_tags = emissions.size()
        alpha = torch.full((batch_size, self.num_tags + 2), -10000.0, device=emissions.device)
        alpha[:, self.start_tag] = 0.0

        for i in range(seq_len):
            mask = masks[:, i].unsqueeze(1)
            emit_score = emissions[:, i].unsqueeze(1)
            emit_score = torch.cat([
                emit_score,
                torch.full((batch_size, 1, 2), -10000.0, device=emissions.device)
            ], dim=2)

            alpha_t = alpha.unsqueeze(2) + self.transitions.unsqueeze(0) + emit_score
            alpha_t = torch.logsumexp(alpha_t, dim=1)
            alpha = torch.where(mask, alpha_t, alpha)

        alpha = alpha + self.transitions[:, self.end_tag].unsqueeze(0)
        return torch.logsumexp(alpha, dim=1)

    def decode(self, emissions, masks):
        batch_size, seq_len, num_tags = emissions.size()
        viterbi = torch.full((batch_size, self.num_tags + 2), -10000.0, device=emissions.device)
        viterbi[:, self.start_tag] = 0.0
        backpointers = torch.zeros((batch_size, seq_len, self.num_tags + 2), dtype=torch.long, device=emissions.device)

        for i in range(seq_len):
            mask = masks[:, i].unsqueeze(1)
            emit_score = emissions[:, i].unsqueeze(1)
            emit_score = torch.cat([
                emit_score,
                torch.full((batch_size, 1, 2), -10000.0, device=emissions.device)
            ], dim=2)

            viterbi_t = viterbi.unsqueeze(2) + self.transitions.unsqueeze(0)
            viterbi_t, bptrs_t = torch.max(viterbi_t, dim=1)
            viterbi_t += emit_score.squeeze(1)
            viterbi = torch.where(mask, viterbi_t, viterbi)
            backpointers[:, i, :] = bptrs_t

        viterbi += self.transitions[:, self.end_tag].unsqueeze(0)
        best_tags_list = []

        for i in range(batch_size):
            best_last_tag = viterbi[i].argmax().item()
            best_tags = [best_last_tag]

            for j in range(seq_len - 1, -1, -1):
                best_tag = backpointers[i, j, best_tags[-1]].item()
                best_tags.append(best_tag)

            best_tags = best_tags[::-1]
            seq_len_i = masks[i].sum().item()
            best_tags = best_tags[:seq_len_i]
            best_tags = [tag for tag in best_tags if 0 <= tag < self.num_tags]
            best_tags_list.append(best_tags)

        return best_tags_list


# LSTM+CRF模型
class LSTMCRF(nn.Module):
    def __init__(self, vocab_size, tag_to_idx, embed_dim, hidden_dim, num_layers=1, dropout=0.5):
        super(LSTMCRF, self).__init__()
        self.vocab_size = vocab_size
        self.tag_to_idx = tag_to_idx
        self.num_tags = len(tag_to_idx)
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.hidden2tag = nn.Linear(hidden_dim, self.num_tags)
        self.crf = CRF(self.num_tags)
        self.dropout = nn.Dropout(dropout)

    def forward(self, sentences, masks, labels):
        features = self._get_lstm_features(sentences, masks)
        return self.crf(features, masks, labels)

    def decode(self, sentences, masks):
        features = self._get_lstm_features(sentences, masks)
        return self.crf.decode(features, masks)

    def _get_lstm_features(self, sentences, masks):
        batch_size, seq_len = sentences.size()
        embeds = self.embedding(sentences)
        embeds = self.dropout(embeds)

        lengths = masks.sum(dim=1).cpu()
        packed_embeds = nn.utils.rnn.pack_padded_sequence(embeds, lengths, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_embeds)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        output = output * masks.unsqueeze(-1).float()
        features = self.hidden2tag(output)
        return features


# 评估函数
def evaluate(model, data_loader, idx_to_tag, device):
    model.eval()
    total_correct = 0
    total_tokens = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for sentences, labels, masks in data_loader:
            sentences, labels, masks = sentences.to(device), labels.to(device), masks.to(device)
            predicted_tags = model.decode(sentences, masks)

            for i in range(len(predicted_tags)):
                seq_len = masks[i].sum().item()
                preds = predicted_tags[i]
                golds = labels[i][:seq_len].tolist()

                if len(preds) < seq_len:
                    preds += [0] * (seq_len - len(preds))
                elif len(preds) > seq_len:
                    preds = preds[:seq_len]

                pred_tags = [idx_to_tag[idx] for idx in preds]
                gold_tags = [idx_to_tag[idx] for idx in golds]

                all_preds.extend(pred_tags)
                all_labels.extend(gold_tags)
                total_correct += sum([1 for p, g in zip(pred_tags, gold_tags) if p == g])
                total_tokens += len(pred_tags)

    accuracy = total_correct / total_tokens if total_tokens > 0 else 0
    tags = list(set(all_labels))
    metrics = {}

    for tag in tags:
        if tag == '<PAD>':
            continue

        tp = sum([1 for p, g in zip(all_preds, all_labels) if p == g and p == tag])
        fp = sum([1 for p, g in zip(all_preds, all_labels) if p == tag and g != tag])
        fn = sum([1 for p, g in zip(all_preds, all_labels) if p != tag and g == tag])

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        metrics[tag] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': tp + fn
        }

    micro_tp = sum([metrics[tag]['support'] for tag in metrics if tag.startswith('B-') or tag.startswith('I-')])
    micro_fp = sum([sum([1 for p, g in zip(all_preds, all_labels) if p == tag and g != tag])
                    for tag in metrics if tag.startswith('B-') or tag.startswith('I-')])
    micro_fn = sum([sum([1 for p, g in zip(all_preds, all_labels) if p != tag and g == tag])
                    for tag in metrics if tag.startswith('B-') or tag.startswith('I-')])

    micro_precision = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) > 0 else 0
    micro_recall = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) > 0 else 0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (
                                                                                                    micro_precision + micro_recall) > 0 else 0

    return accuracy, metrics, micro_f1


# 可视化训练历史
def plot_training_history(history, model_name="LSTM+CRF"):
    plt.figure(figsize=(12, 5))

    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='训练损失')
    plt.xlabel('轮次', fontproperties=font)
    plt.ylabel('损失值', fontproperties=font)
    plt.title(f'{model_name} 训练损失曲线', fontproperties=font)
    plt.legend(prop=font)

    # F1分数曲线
    plt.subplot(1, 2, 2)
    plt.plot(history['val_f1'], label='验证集Micro F1')
    plt.xlabel('轮次', fontproperties=font)
    plt.ylabel('F1分数', fontproperties=font)
    plt.title(f'{model_name} 验证集性能曲线', fontproperties=font)
    plt.legend(prop=font)

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300)
    plt.close()
    print("训练历史可视化已保存为 training_history.png")


# 可视化实体类型性能对比
def plot_entity_metrics(metrics):
    # 筛选实体标签（排除'O'和'<PAD>'）
    entity_tags = [tag for tag in metrics.keys() if
                   tag not in ['O', '<PAD>'] and (tag.startswith('B-') or tag.startswith('I-'))]
    entity_tags.sort()

    # 提取指标
    precisions = [metrics[tag]['precision'] for tag in entity_tags]
    recalls = [metrics[tag]['recall'] for tag in entity_tags]
    f1_scores = [metrics[tag]['f1'] for tag in entity_tags]

    # 绘制条形图
    x = np.arange(len(entity_tags))
    width = 0.25

    plt.figure(figsize=(12, 6))
    plt.bar(x - width, precisions, width, label='精确率')
    plt.bar(x, recalls, width, label='召回率')
    plt.bar(x + width, f1_scores, width, label='F1分数')

    plt.xlabel('实体类型', fontproperties=font)
    plt.ylabel('分数', fontproperties=font)
    plt.title('不同实体类型的性能指标对比', fontproperties=font)
    plt.xticks(x, entity_tags, rotation=45, ha='right', fontproperties=font)
    plt.ylim(0, 1.0)
    plt.legend(prop=font)
    plt.tight_layout()
    plt.savefig('entity_metrics.png', dpi=300)
    plt.close()
    print("实体类型性能对比已保存为 entity_metrics.png")


# 可视化实体标注示例
def visualize_ner_example(words, gold_tags, pred_tags, idx):
    # 定义颜色映射
    color_map = {
        'B-PER': '#FFAAAA', 'I-PER': '#FFAAAA',
        'B-ORG': '#AAFFAA', 'I-ORG': '#AAFFAA',
        'B-LOC': '#AAAAFF', 'I-LOC': '#AAAAFF',
        'B-MISC': '#FFFFAA', 'I-MISC': '#FFFFAA',
        'O': '#FFFFFF'
    }

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('off')

    # 绘制表格
    table_data = [[word, gold, pred] for word, gold, pred in zip(words, gold_tags, pred_tags)]
    table = ax.table(cellText=table_data, colLabels=['词语', '真实标签', '预测标签'],
                     cellLoc='center', loc='center', bbox=[0, 0, 1, 1])

    # 设置单元格样式
    for i in range(len(words)):
        # 词语单元格背景色根据真实标签设置
        gold_tag = gold_tags[i]
        table[(i + 1, 0)].set_facecolor(color_map.get(gold_tag, '#FFFFFF'))
        # 高亮预测错误的单元格
        if gold_tags[i] != pred_tags[i]:
            table[(i + 1, 2)].set_facecolor('#FFCCCC')

    plt.title(f'实体识别示例 {idx + 1}', fontproperties=font)
    plt.tight_layout()
    plt.savefig(f'ner_example_{idx}.png', dpi=300)
    plt.close()
    print(f"实体识别示例 {idx + 1} 已保存为 ner_example_{idx}.png")


# 训练函数
def train_model(model, train_loader, val_loader, optimizer, num_epochs, idx_to_tag, device):
    model = model.to(device)
    best_f1 = 0.0
    history = {'train_loss': [], 'val_f1': []}

    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        total_loss = 0.0

        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
        for i, (sentences, labels, masks) in progress_bar:
            sentences, labels, masks = sentences.to(device), labels.to(device), masks.to(device)
            optimizer.zero_grad()
            loss = -model(sentences, masks, labels).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item()
            progress_bar.set_description(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}')

        avg_train_loss = total_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        accuracy, metrics, val_f1 = evaluate(model, val_loader, idx_to_tag, device)
        history['val_f1'].append(val_f1)

        end_time = time.time()
        epoch_time = end_time - start_time
        print(f'Epoch {epoch + 1}/{num_epochs} | 时间: {epoch_time:.2f}s | '
              f'训练损失: {avg_train_loss:.4f} | 验证集Micro F1: {val_f1:.4f} | 准确率: {accuracy:.4f}')

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), 'best_lstm_crf_model.pt')
            print(f"最佳模型已保存，验证集Micro F1: {val_f1:.4f}")

        print("-" * 80)

    return history


# 主函数
def main():
    # 加载数据
    print("加载数据...")
    data_dir = r"data\nlp4\Conll-2003"

    train_sentences, train_labels = load_conll_data(data_dir, 'train')
    val_sentences, val_labels = load_conll_data(data_dir, 'dev')
    test_sentences, test_labels = load_conll_data(data_dir, 'test')

    print(f"训练集样本数: {len(train_sentences)}")
    print(f"验证集样本数: {len(val_sentences)}")
    print(f"测试集样本数: {len(test_sentences)}")

    # 构建词汇表
    print("构建词汇表...")
    word_to_idx, label_to_idx = build_vocab(train_sentences, train_labels)
    idx_to_word = {v: k for k, v in word_to_idx.items()}
    idx_to_label = {v: k for k, v in label_to_idx.items()}

    print(f"词汇表大小: {len(word_to_idx)}")
    print(f"标签数量: {len(label_to_idx)}")

    # 数据验证
    max_label_idx = max(label_to_idx.values())
    print("\n验证训练数据...")
    train_dataset = NERDataset(train_sentences, train_labels, word_to_idx, label_to_idx)
    validate_data(train_dataset, max_label_idx)

    val_dataset = NERDataset(val_sentences, val_labels, word_to_idx, label_to_idx)
    test_dataset = NERDataset(test_sentences, test_labels, word_to_idx, label_to_idx)

    # 创建数据加载器
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)

    # 初始化模型
    model = LSTMCRF(
        vocab_size=len(word_to_idx),
        tag_to_idx=label_to_idx,
        embed_dim=100,
        hidden_dim=200,
        num_layers=1,
        dropout=0.5
    )

    # 训练模型
    print("开始训练模型...")
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 5
    history = train_model(model, train_loader, val_loader, optimizer, num_epochs, idx_to_label, device)

    # 可视化训练历史
    plot_training_history(history)

    # 测试集评估
    print("在测试集上评估...")
    model.load_state_dict(torch.load('best_lstm_crf_model.pt', map_location=device))
    accuracy, metrics, micro_f1 = evaluate(model, test_loader, idx_to_label, device)

    print(f"\n测试集Micro F1: {micro_f1:.4f}")
    print(f"测试集准确率: {accuracy:.4f}")

    # 可视化实体类型性能
    plot_entity_metrics(metrics)

    # 可视化实体标注示例
    model.eval()
    sample_indices = [0, 5, 10]  # 选择3个示例
    for idx in sample_indices:
        sentence_idx, label_idx = test_dataset[idx]
        sentence_tensor = torch.tensor([sentence_idx], dtype=torch.long)
        mask_tensor = torch.tensor([[1] * len(sentence_idx)], dtype=torch.bool)

        with torch.no_grad():
            predicted_tags = model.decode(sentence_tensor, mask_tensor)[0]

        words = [idx_to_word[i] for i in sentence_idx]
        gold_tags = [idx_to_label[i] for i in label_idx]
        pred_tags = [idx_to_label[i] for i in predicted_tags]

        visualize_ner_example(words, gold_tags, pred_tags, idx)


if __name__ == "__main__":
    main()