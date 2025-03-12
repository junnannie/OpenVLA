import numpy as np
import gensim.downloader as api
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 加载预训练的 Word2Vec 模型（Google News）
model = api.load("word2vec-google-news-300")

# 要分析的单词
words = ["king", "queen", "man", "woman", "apple", "banana", "paris", "london", "china", "india"]

# 提取这些单词的词向量
word_vectors = [model[word] for word in words]

# 打印每个词的词向量
print("Word Vectors (Before PCA):")
for word, vec in zip(words, word_vectors):
    print(f"{word}: {vec[:10]}...")  # 仅显示前10个维度，避免打印过长
    print(f"Shape of {word} vector: {vec.shape}")

# 打印词向量的一些统计信息
vectors_array = np.array(word_vectors)
print("\nStatistical Summary of Word Vectors (Before PCA):")
print(f"Shape of word vectors: {vectors_array.shape}")
print(f"Mean of word vectors: {np.mean(vectors_array, axis=0)}")
print(f"Standard deviation of word vectors: {np.std(vectors_array, axis=0)}")
print(f"Min value in word vectors: {np.min(vectors_array, axis=0)}")
print(f"Max value in word vectors: {np.max(vectors_array, axis=0)}")

# 使用PCA将词向量降到2D
pca = PCA(n_components=2)
reduced_vectors = pca.fit_transform(word_vectors)

# 可视化
plt.figure(figsize=(10, 8))
plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1])

# 为每个点添加标签
for i, word in enumerate(words):
    plt.text(reduced_vectors[i, 0], reduced_vectors[i, 1], word, fontsize=12)

plt.title("2D PCA of Word2Vec Word Embeddings")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid(True)
plt.show()
