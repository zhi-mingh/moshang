import numpy as np
import os
import re
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from collections import defaultdict, Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import random
from tqdm import tqdm
from scipy.sparse import csr_matrix, vstack


# 设置随机种子以确保结果可复现
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)


set_seed(42)


# 配置matplotlib支持中文显示
def configure_matplotlib():
    """配置matplotlib以支持中文显示"""
    # 获取系统中所有可用字体
    fonts = fm.findSystemFonts()

    # 查找支持中文的字体
    chinese_fonts = []
    for font in fonts:
        try:
            font_name = fm.FontProperties(fname=font).get_name()
            if "Heiti" in font_name or "SimHei" in font_name or "WenQuanYi" in font_name:
                chinese_fonts.append(font_name)
        except:
            continue

    # 如果找到中文字体，则使用第一个找到的中文字体
    if chinese_fonts:
        plt.rcParams["font.family"] = chinese_fonts[0]
        print(f"已配置matplotlib使用中文字体: {chinese_fonts[0]}")
    else:
        print("未找到中文字体，图表中的中文可能无法正常显示。")


# 配置matplotlib
configure_matplotlib()


class TextClassifier:
    def __init__(self, feature_type='bow', n_gram=1, learning_rate=0.01, num_epochs=100,
                 batch_size=32, optimizer='sgd', regularization=0.0, max_vocab_size=5000,
                 progress_update_freq=10):
        """初始化文本分类器"""
        self.feature_type = feature_type
        self.n_gram = n_gram
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.regularization = regularization
        self.max_vocab_size = max_vocab_size  # 新增：限制词汇表最大规模
        self.progress_update_freq = progress_update_freq
        self.vocab = None
        self.label_dict = None
        self.W = None
        self.b = None
        self.history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    def preprocess_text(self, text):
        """预处理文本：转为小写并移除特殊字符"""
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        return text

    def tokenize(self, text):
        """将文本分词"""
        return text.split()

    def build_vocab(self, texts, min_freq=5):
        """构建词汇表，增加最大规模限制"""
        if self.feature_type == 'bow':
            # 词袋模型
            word_freq = defaultdict(int)
            for text in texts:
                tokens = self.tokenize(text)
                for token in tokens:
                    word_freq[token] += 1

            # 过滤低频词并限制最大词汇量
            filtered = [word for word, freq in word_freq.items() if freq >= min_freq]
            # 按词频排序，取前max_vocab_size个
            sorted_words = sorted(filtered, key=lambda x: word_freq[x], reverse=True)[:self.max_vocab_size]
            self.vocab = {word: idx for idx, word in enumerate(sorted_words)}

        elif self.feature_type == 'ngram':
            # N-gram模型
            ngram_freq = defaultdict(int)
            for text in texts:
                tokens = self.tokenize(text)
                ngrams = []
                for i in range(len(tokens) - self.n_gram + 1):
                    ngram = ' '.join(tokens[i:i + self.n_gram])
                    ngrams.append(ngram)

                for ngram in ngrams:
                    ngram_freq[ngram] += 1

            # 过滤低频N-gram并限制最大规模
            filtered = [ngram for ngram, freq in ngram_freq.items() if freq >= min_freq]
            sorted_ngrams = sorted(filtered, key=lambda x: ngram_freq[x], reverse=True)[:self.max_vocab_size]
            self.vocab = {ngram: idx for idx, ngram in enumerate(sorted_ngrams)}

        else:
            raise ValueError(f"不支持的特征类型: {self.feature_type}")

    def text_to_feature(self, text):
        """将文本转换为稀疏特征向量（关键优化）"""
        if not self.vocab:
            raise ValueError("词汇表尚未构建，请先调用build_vocab方法")

        indices = []
        if self.feature_type == 'bow':
            tokens = self.tokenize(text)
            for token in tokens:
                if token in self.vocab:
                    indices.append(self.vocab[token])

        elif self.feature_type == 'ngram':
            tokens = self.tokenize(text)
            for i in range(len(tokens) - self.n_gram + 1):
                ngram = ' '.join(tokens[i:i + self.n_gram])
                if ngram in self.vocab:
                    indices.append(self.vocab[ngram])

        # 统计词频
        if indices:
            counts = Counter(indices)
            return csr_matrix(
                (list(counts.values()), ([0] * len(counts), list(counts.keys()))),
                shape=(1, len(self.vocab)),
                dtype=np.float32  # 使用float32减少内存
            )
        else:
            return csr_matrix((1, len(self.vocab)), dtype=np.float32)

    def prepare_data(self, texts, labels=None):
        """准备特征矩阵（稀疏矩阵）和标签向量"""
        # 批量处理文本为稀疏矩阵
        matrices = []
        for text in texts:
            matrices.append(self.text_to_feature(text))

        # 垂直堆叠稀疏矩阵
        X = vstack(matrices, format='csr')

        if labels is not None:
            # 构建标签字典
            if self.label_dict is None:
                unique_labels = sorted(list(set(labels)))
                self.label_dict = {label: idx for idx, label in enumerate(unique_labels)}
                self.reverse_label_dict = {idx: label for label, idx in self.label_dict.items()}

            # 转换标签为one-hot编码
            y = np.zeros((len(labels), len(self.label_dict)), dtype=np.float32)
            for i, label in enumerate(labels):
                y[i, self.label_dict[label]] = 1

            return X, y
        else:
            return X

    def softmax(self, z):
        """Softmax函数"""
        z = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def cross_entropy_loss(self, y_pred, y_true):
        """计算交叉熵损失"""
        epsilon = 1e-10
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

        if self.regularization > 0 and self.W is not None:
            loss += (self.regularization / 2) * np.sum(self.W ** 2)

        return loss

    def predict(self, X):
        """预测类别"""
        # 稀疏矩阵乘法
        z = X.dot(self.W) + self.b
        y_pred = self.softmax(z)
        return np.argmax(y_pred, axis=1)

    def predict_proba(self, X):
        """预测类别概率"""
        z = X.dot(self.W) + self.b
        return self.softmax(z)

    def evaluate(self, X, y_true):
        """评估模型性能"""
        y_pred = self.predict(X)
        y_true_labels = np.argmax(y_true, axis=1)

        # 设置zero_division=1避免警告
        accuracy = accuracy_score(y_true_labels, y_pred)
        precision = precision_score(y_true_labels, y_pred, average='macro', zero_division=1)
        recall = recall_score(y_true_labels, y_pred, average='macro', zero_division=1)
        f1 = f1_score(y_true_labels, y_pred, average='macro', zero_division=1)

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """训练模型，适配稀疏矩阵"""
        num_samples, num_features = X_train.shape
        num_classes = y_train.shape[1]

        # 初始化权重
        self.W = np.random.randn(num_features, num_classes).astype(np.float32) * 0.01
        self.b = np.zeros(num_classes, dtype=np.float32)

        # 计算批次数
        num_batches = (num_samples + self.batch_size - 1) // self.batch_size

        # 创建进度条
        epoch_range = tqdm(range(self.num_epochs), desc="训练进度", unit="epoch", leave=True)

        for epoch in epoch_range:
            if self.optimizer == 'gd':
                # 批量梯度下降（适配稀疏矩阵）
                z = X_train.dot(self.W) + self.b
                y_pred = self.softmax(z)

                # 计算梯度（稀疏矩阵转置乘法）
                dW = X_train.T.dot(y_pred - y_train) / num_samples
                db = np.sum(y_pred - y_train, axis=0) / num_samples

                if self.regularization > 0:
                    dW += self.regularization * self.W

                self.W -= self.learning_rate * dW
                self.b -= self.learning_rate * db

            elif self.optimizer == 'sgd':
                # 随机梯度下降
                indices = np.arange(num_samples)
                np.random.shuffle(indices)

                dW = np.zeros_like(self.W)
                db = np.zeros_like(self.b)
                epoch_loss = 0

                for i in range(num_batches):
                    start_idx = i * self.batch_size
                    end_idx = min(start_idx + self.batch_size, num_samples)
                    batch_indices = indices[start_idx:end_idx]

                    # 稀疏矩阵按索引取批次
                    X_batch = X_train[batch_indices]
                    y_batch = y_train[batch_indices]
                    batch_size = len(batch_indices)

                    z = X_batch.dot(self.W) + self.b
                    y_pred = self.softmax(z)

                    # 计算梯度
                    batch_grad = X_batch.T.dot(y_pred - y_batch) / batch_size
                    dW += batch_grad
                    db += np.sum(y_pred - y_batch, axis=0) / batch_size

                    # 累积损失
                    epoch_loss += self.cross_entropy_loss(y_pred, y_batch) * batch_size

                # 平均梯度和损失
                dW /= num_batches
                db /= num_batches
                epoch_loss /= num_samples

                if self.regularization > 0:
                    dW += self.regularization * self.W

                self.W -= self.learning_rate * dW
                self.b -= self.learning_rate * db

            else:
                raise ValueError(f"不支持的优化器: {self.optimizer}")

            # 计算训练损失和准确率
            train_loss = self.cross_entropy_loss(self.softmax(X_train.dot(self.W) + self.b), y_train)
            train_acc = self.evaluate(X_train, y_train)['accuracy']

            # 计算验证损失和准确率
            val_loss = None
            val_acc = None
            if X_val is not None and y_val is not None:
                val_loss = self.cross_entropy_loss(self.softmax(X_val.dot(self.W) + self.b), y_val)
                val_acc = self.evaluate(X_val, y_val)['accuracy']

            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            if val_loss is not None:
                self.history['val_loss'].append(val_loss)
            if val_acc is not None:
                self.history['val_acc'].append(val_acc)

            # 更新进度条信息
            if (epoch + 1) % self.progress_update_freq == 0 or epoch == self.num_epochs - 1:
                if X_val is not None:
                    epoch_range.set_postfix({
                        'train_loss': f"{train_loss:.4f}",
                        'train_acc': f"{train_acc:.4f}",
                        'val_loss': f"{val_loss:.4f}",
                        'val_acc': f"{val_acc:.4f}"
                    })
                else:
                    epoch_range.set_postfix({
                        'train_loss': f"{train_loss:.4f}",
                        'train_acc': f"{train_acc:.4f}"
                    })

    def plot_training_history(self, save_path='training_history.png'):
        """绘制训练历史"""
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(self.history['train_loss'], label='训练损失')
        if self.history['val_loss']:
            plt.plot(self.history['val_loss'], label='验证损失')
        plt.xlabel('轮次')
        plt.ylabel('损失')
        plt.title('训练和验证损失')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(self.history['train_acc'], label='训练准确率')
        if self.history['val_acc']:
            plt.plot(self.history['val_acc'], label='验证准确率')
        plt.xlabel('轮次')
        plt.ylabel('准确率')
        plt.title('训练和验证准确率')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def compare_features(self, train_texts, train_labels, val_texts, val_labels, feature_types=['bow', 'ngram']):
        """比较不同特征类型的性能"""
        results = {}
        original_feature_type = self.feature_type

        for feature_type in feature_types:
            # 重置模型状态
            self.feature_type = feature_type
            self.vocab = None
            self.W = None
            self.b = None
            self.label_dict = None
            self.history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

            print(f"\n使用 {feature_type} 特征训练模型...")

            # 预处理文本并构建词汇表
            preprocessed_train = [self.preprocess_text(text) for text in train_texts]
            preprocessed_val = [self.preprocess_text(text) for text in val_texts]

            self.build_vocab(preprocessed_train)
            print(f"{feature_type} 词汇表大小: {len(self.vocab)}")

            # 准备数据
            X_train, y_train = self.prepare_data(preprocessed_train, train_labels)
            X_val, y_val = self.prepare_data(preprocessed_val, val_labels)

            # 训练模型
            self.train(X_train, y_train, X_val, y_val)

            # 评估模型
            metrics = self.evaluate(X_val, y_val)
            results[feature_type] = metrics

            print(f"{feature_type} 特征验证集性能:")
            for key, value in metrics.items():
                print(f"  {key}: {value:.4f}")

        # 恢复原始特征类型
        self.feature_type = original_feature_type

        # 可视化比较结果
        self._plot_feature_comparison(results)

        return results

    def _plot_feature_comparison(self, results):
        """可视化不同特征类型的性能比较"""
        metrics = list(results[list(results.keys())[0]].keys())
        feature_types = list(results.keys())

        x = np.arange(len(metrics))
        width = 0.8 / len(feature_types)

        plt.figure(figsize=(10, 6))

        for i, feature_type in enumerate(feature_types):
            values = [results[feature_type][metric] for metric in metrics]
            plt.bar(x + i * width - width * (len(feature_types) - 1) / 2, values, width, label=feature_type)

        plt.ylabel('分数')
        plt.title('不同特征类型的性能比较')
        plt.xticks(x, metrics)
        plt.legend()
        plt.grid(axis='y')

        plt.tight_layout()
        plt.savefig('feature_comparison.png')
        plt.close()


def load_rotten_tomatoes_data(train_path, test_path=None):
    """加载Rotten Tomatoes数据集"""
    # 加载训练集
    train_texts = []
    train_labels = []
    with open(train_path, 'r', encoding='utf-8') as f:
        header = next(f).strip().split('\t')
        text_idx = header.index('Phrase')
        label_idx = header.index('Sentiment') if 'Sentiment' in header else None

        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < text_idx + 1:
                continue

            text = parts[text_idx]
            train_texts.append(text)

            if label_idx is not None:
                label = parts[label_idx]
                train_labels.append(label)

    # 加载测试集（如果提供）
    test_texts = []
    test_ids = []
    if test_path:
        with open(test_path, 'r', encoding='utf-8') as f:
            header = next(f).strip().split('\t')
            text_idx = header.index('Phrase')
            id_idx = header.index('PhraseId') if 'PhraseId' in header else None

            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < text_idx + 1:
                    continue

                text = parts[text_idx]
                test_texts.append(text)

                if id_idx is not None:
                    test_ids.append(parts[id_idx])

    if test_path:
        return train_texts, train_labels, test_texts, test_ids
    else:
        return train_texts, train_labels


def main():
    # 加载数据
    print("加载数据...")
    data_dir = "data/nlp1"
    train_path = os.path.join(data_dir, "train.tsv")
    test_path = os.path.join(data_dir, "test.tsv")

    # 检查文件是否存在
    if not os.path.exists(train_path):
        print(f"错误：训练数据文件 {train_path} 不存在")
        return

    train_texts, train_labels, test_texts, test_ids = load_rotten_tomatoes_data(train_path, test_path)

    # 划分数据集
    print("划分数据集...")
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts, train_labels, test_size=0.2, random_state=42)

    print(f"训练集大小: {len(train_texts)}")
    print(f"验证集大小: {len(val_texts)}")
    print(f"测试集大小: {len(test_texts)}")

    # 检查类别分布
    print("\n类别分布:")
    label_counts = Counter(train_labels)
    for label, count in label_counts.items():
        print(f"  {label}: {count} ({count / len(train_labels):.4f})")

    # 初始化分类器（优化参数）
    print("初始化分类器...")
    clf = TextClassifier(
        feature_type='bow',
        n_gram=1,
        learning_rate=0.01,
        num_epochs=50,  # 适当减少轮次加速训练
        batch_size=32,
        optimizer='sgd',
        regularization=0.001,
        max_vocab_size=5000,  # 限制词汇表大小
        progress_update_freq=5
    )

    # 构建词汇表
    print("构建词汇表...")
    preprocessed_train_texts = [clf.preprocess_text(text) for text in train_texts]
    clf.build_vocab(preprocessed_train_texts, min_freq=3)  # 提高最低频率过滤
    print(f"词汇表大小: {len(clf.vocab)}")

    # 准备数据（稀疏矩阵）
    print("准备数据...")
    X_train, y_train = clf.prepare_data(preprocessed_train_texts, train_labels)
    X_val, y_val = clf.prepare_data(
        [clf.preprocess_text(text) for text in val_texts], val_labels)
    X_test = clf.prepare_data(
        [clf.preprocess_text(text) for text in test_texts])

    # 训练模型
    print("训练模型...")
    clf.train(X_train, y_train, X_val, y_val)

    # 评估模型
    print("\n评估模型...")
    train_metrics = clf.evaluate(X_train, y_train)
    val_metrics = clf.evaluate(X_val, y_val)

    print("训练集性能:")
    for key, value in train_metrics.items():
        print(f"  {key}: {value:.4f}")

    print("\n验证集性能:")
    for key, value in val_metrics.items():
        print(f"  {key}: {value:.4f}")

    # 预测测试集
    print("\n预测测试集...")
    test_preds = clf.predict(X_test)
    test_pred_labels = [clf.reverse_label_dict[pred] for pred in test_preds]

    # 保存预测结果
    output_path = "predictions.csv"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("PhraseId,Sentiment\n")
        for phrase_id, pred in zip(test_ids, test_pred_labels):
            f.write(f"{phrase_id},{pred}\n")

    print(f"预测结果已保存至 {output_path}")

    # 绘制训练历史
    clf.plot_training_history()
    print("训练历史图表已保存为 training_history.png")

    # 比较不同特征类型
    print("\n比较不同特征类型...")
    clf.compare_features(train_texts, train_labels, val_texts, val_labels,
                         feature_types=['bow', 'ngram'])
    print("特征比较图表已保存为 feature_comparison.png")


if __name__ == "__main__":
    main()