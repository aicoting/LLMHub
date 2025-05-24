## Transformer
### **问题分析**
Transformer 是 Google 2017 年提出的模型架构，特点是不依赖循环（RNN），完全基于**注意力机制**，同时并行计算效率高，广泛应用于自然语言处理和计算机视觉。

Transformer 主要分为**编码器 (Encoder)** 和**解码器 (Decoder)** 两部分，整体由多个堆叠的子层组成。

#### **Encoder 结构**
每个 Encoder Layer 包含：

1. **多头自注意力机制 (Multi-Head Self Attention)**
2. **前向全连接层 (Feed-Forward Network, FFN)**
3. **残差连接 + 层归一化 (Residual + LayerNorm)**

#### **Decoder 结构**
每个 Decoder Layer 多了一个模块，包含：

1. **Masked 多头自注意力机制 (Masked Multi-Head Self Attention)**
2. **Encoder-Decoder Attention**（用于关注 Encoder 输出）
3. **前向全连接层 (FFN)**
4. **残差连接 + 层归一化**

#### **核心原理 — 注意力机制 (Attention)**
注意力机制本质上通过**Query-Key-Value** 计算序列内部各个词之间的关系权重：

```plain
Attention(Q, K, V) = softmax(QKᵀ / √d_k) * V
```

这使得模型可以动态关注输入序列中与当前词最相关的信息。

#### **多头机制 (Multi-Head)**
将注意力机制并行分成多个子空间（多个头），增强模型捕捉多粒度特征的能力，最后拼接融合。

#### **位置编码 (Positional Encoding)**
因为 Transformer 不像 RNN 有顺序性，位置编码 (Positional Encoding) 用于引入序列位置信息，通常采用正余弦函数。

### **面试回答**
> Transformer 通过堆叠的注意力机制和前向网络，结合残差与归一化，能高效捕捉长距离依赖，广泛用于 NLP 和 CV 任务中。Transformer 后续启发了 BERT（只用 Encoder）、GPT（只用 Decoder）、Vision Transformer (ViT) 等模型，成为现代 AI 基础架构之一。
>

## Transformer为什么可以并行
### 问题分析
#### 摒弃 RNN 中的串行依赖
+ RNN/LSTM 必须按时间步一个一个处理序列，后一个位置依赖前一个位置的隐藏状态，**无法并行**。
+ Transformer 完全去除了这种时间步依赖，**所有位置可以同时进行计算**。

#### 自注意力机制是“全局并行”的核心
+ 在 Transformer 中，每一个 token 的表示通过 **对整句中的所有 token 进行加权求和（Self-Attention）** 得到。
+ 这些注意力权重的计算可以通过矩阵操作一次性完成。

📌 举例：

> 输入序列长度为 n，那么 Self-Attention 的计算是通过 Q dot K^T 实现 n  的权重矩阵，这可以使用矩阵乘法一次性完成，天然适合 GPU 并行。
>

#### 位置编码用于补充顺序信息
+ Transformer 没有顺序结构，因此使用位置编码（如正余弦位置编码或 RoPE）提供序列顺序信息。
+ 这样既保持了并行计算，又保留了序列建模能力。

#### 全部采用矩阵操作，适合 GPU/TPU 加速
+ Transformer 中的注意力机制、前馈网络（FFN）、LayerNorm 等模块都是矩阵操作。
+ 矩阵乘法是 **现代硬件并行加速的最优场景**（比如 CUDA 核心可以大规模并发执行）。

### 面试回答
> Transformer 之所以能够很好地并行，是因为它摒弃了循环结构，采用了自注意力机制（Self-Attention），可以让每个位置同时看到全局信息，因此训练时可以用矩阵操作进行完全并行，非常适合 GPU 加速，可以同时处理整个序列中所有位置的输入，从而大大提升了训练和推理的并行效率。
>

