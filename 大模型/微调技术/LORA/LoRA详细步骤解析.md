# 📚 微调系列文章
[一文了解微调技术的发展与演进](https://zhuanlan.zhihu.com/p/1939080284374022103)  
[一文搞懂 LoRA 如何高效微调大模型](https://zhuanlan.zhihu.com/p/1939447022114567022)

随着大规模 Transformer 模型（如 GPT、LLaMA、ViT）的广泛应用，微调大模型的计算和存储成本成为制约因素。  
LoRA 作为一种参数高效微调（PEFT）技术，通过低秩矩阵分解，仅微调增量部分，有效降低了资源消耗。

本文将分步骤解析 LoRA 的训练原理及优势，帮助你快速掌握 LoRA 的核心机制。

<font style="color:rgb(25, 27, 31);">所有相关源码示例、流程图、模型配置与知识库构建技巧，我也将持续更新在Github：</font>[**<font style="color:rgb(25, 27, 31);">LLMHub</font>**](https://github.com/aicoting/LLMHub)<font style="color:rgb(25, 27, 31);">，欢迎关注收藏！</font>

---

## 一、LoRA 简介
![](https://cdn.nlark.com/yuque/0/2025/png/28454971/1754983348216-9c00e71e-39c8-473d-8486-31157df578ef.png)

LoRA（Low-Rank Adaptation）是一种针对大模型的参数高效微调方法，核心思想是：

+ **冻结大部分预训练模型参数，只在低秩矩阵上进行增量训练**，显著降低训练和存储成本；
+ 适用于大语言模型（LLM）、视觉 Transformer（ViT）及其他大规模深度模型。

---

## 二、LoRA 训练详细步骤
### 冻结预训练模型参数
传统微调需调整整个 Transformer 权重，而 LoRA 只冻结原模型参数，避免全量反向传播开销。

```python
for param in model.parameters():
    param.requires_grad = False  # 冻结所有参数
```

优点：显著降低训练计算和显存需求。

---

### 替换 Transformer 注意力层的全连接层
Transformer 中，查询（Q）、键（K）、值（V）计算通常通过线性层实现：

![image](https://cdn.nlark.com/yuque/__latex/c9c465ad77b490bd7af8ebfe38bafbc8.svg)

LoRA 不直接训练原始权重 ![image](https://cdn.nlark.com/yuque/__latex/60b59ab950cf235a6c25eb186a35ee5d.svg)，而是对其增量进行低秩分解：

![image](https://cdn.nlark.com/yuque/__latex/a5cdcffd6669ef95358de22c7d0e6eac.svg)

其中：

+ ![image](https://cdn.nlark.com/yuque/__latex/de951302f41d4707b9d80ca1af34dd0f.svg) 大小为 ![image](https://cdn.nlark.com/yuque/__latex/c5a0dfe09e71eaea8635ffd525bf3f56.svg)（低秩矩阵），
+ ![image](https://cdn.nlark.com/yuque/__latex/54f5fb1b07a88521e7b036e3bc7a5e33.svg) 大小为 ![image](https://cdn.nlark.com/yuque/__latex/a3c057578ca804946db284d512f2653b.svg)，
+ ![image](https://cdn.nlark.com/yuque/__latex/02348d660d6f257d77d8965fffa03b34.svg)，大幅减少训练参数。

代码示例：

```python
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank=4, alpha=32):
        super().__init__()
        self.rank = rank
        self.alpha = alpha

        self.W = nn.Linear(in_features, out_features, bias=False)
        self.W.requires_grad_(False)  # 冻结原权重

        self.A = nn.Linear(in_features, rank, bias=False)  # d × r
        self.B = nn.Linear(rank, out_features, bias=False)  # r × d

        nn.init.kaiming_uniform_(self.A.weight, a=5**0.5)
        nn.init.zeros_(self.B.weight)  # B 零初始化，防止扰动原始权重

    def forward(self, x):
        return self.W(x) + self.alpha * self.B(self.A(x))
```

---

### 只训练低秩矩阵参数
```python
optimizer = torch.optim.AdamW([
    {'params': model.lora_A.parameters()},
    {'params': model.lora_B.parameters()}
], lr=1e-4)
```

仅训练 ![image](https://cdn.nlark.com/yuque/__latex/de951302f41d4707b9d80ca1af34dd0f.svg) 和 ![image](https://cdn.nlark.com/yuque/__latex/54f5fb1b07a88521e7b036e3bc7a5e33.svg)，冻结原模型所有参数，显著降低计算量。

---

### 训练完成后权重合并
训练完成后，可将增量权重直接加到原始权重：

![image](https://cdn.nlark.com/yuque/__latex/9f7d5e07025987e0d8fb8e1f3b69c16f.svg)

合并优势：

+ 推理时无额外计算开销；
+ 轻松部署，无需保留额外参数结构。

---

### 推理阶段选择
+ **保持 LoRA 结构**：适合多任务动态切换，节省存储；
+ **合并权重**：适合单一任务高效推理，避免额外计算。

示例合并代码：

```python
model.W_Q.weight.data += model.B.weight @ model.A.weight
```

---

## 三、LoRA 与传统微调对比
| 对比项 | 传统微调（Full Fine-Tuning） | LoRA 微调 |
| --- | --- | --- |
| 参数更新 | 全部权重 | 仅低秩矩阵 A 和 B |
| 训练开销 | 高（数十亿参数） | 低（百万参数级别） |
| 存储需求 | 大 | 小 |
| 推理效率 | 可能受影响 | 几乎无额外负担 |


LoRA 特别适合：

+ 大规模 Transformer 微调（如 GPT、LLaMA、ViT）；
+ 多任务模型快速切换与存储优化；
+ 算力受限环境，如移动端和边缘计算。

---

## 四、LoRA 在 Transformer 中的作用位置
LoRA 主要作用于 Multi-Head Attention 的查询（Q）、键（K）、值（V）线性层，是微调最关键的参数部分。

---

## 五、LoRA 训练示例代码
```python
import torch
import torch.nn as nn
import torch.optim as optim

class LoRAModel(nn.Module):
    def __init__(self, d, r=4, alpha=32):
        super().__init__()
        self.W = nn.Linear(d, d, bias=False)
        self.W.requires_grad_(False)

        self.A = nn.Linear(d, r, bias=False)
        self.B = nn.Linear(r, d, bias=False)

        nn.init.kaiming_uniform_(self.A.weight, a=5**0.5)
        nn.init.zeros_(self.B.weight)

        self.alpha = alpha

    def forward(self, x):
        return self.W(x) + self.alpha * self.B(self.A(x))

model = LoRAModel(d=512, r=4).cuda()

optimizer = optim.AdamW([
    {'params': model.A.parameters()},
    {'params': model.B.parameters()}
], lr=1e-4)

for epoch in range(10):
    x = torch.randn(32, 512).cuda()
    y = model(x).sum()
    y.backward()
    optimizer.step()
    optimizer.zero_grad()
    print(f"Epoch {epoch}: Loss={y.item()}")
```

---

## 六、总结
LoRA 通过对 Transformer 注意力层权重增量进行低秩分解，有效减少训练参数量和计算资源消耗。其采用冻结大模型参数，仅训练低秩矩阵，降低存储和计算开销。并且LORA支持大语言模型和视觉 Transformer 的高效微调，可以兼顾多任务和快速推理。



<font style="color:rgb(25, 27, 31);">关于深度学习和大模型相关的知识和前沿技术更新，请关注公众号</font><font style="color:rgb(25, 27, 31);background-color:rgb(246, 246, 246);">coting</font><font style="color:rgb(25, 27, 31);">！</font>



