本篇文章是Transformer系列的第七篇。

Transformer系列文章：

● [一览Transformer整体架构](https://zhuanlan.zhihu.com/p/1918047303597552480)  
● [Transformer——Attention怎么实现集中注意力](https://zhuanlan.zhihu.com/p/1918049072331362469)  
● [Transformer——FeedForward模块在干什么？](https://zhuanlan.zhihu.com/p/1918050616376301224)  
● [从0开始实现Transformer](https://zhuanlan.zhihu.com/p/1918357249883105145)  
● [什么是KV-Cache](https://zhuanlan.zhihu.com/p/1919338888536756837)  
● [Transformer注意力机制——MHA&MQA&GQA](https://zhuanlan.zhihu.com/p/1919500956946655189)



<font style="color:rgb(25, 27, 31);">所有相关源码示例、流程图、模型配置与知识库构建技巧，我也将持续更新在Github：</font>[**<font style="color:rgb(25, 27, 31);">LLMHub</font>**](https://github.com/zhangting-hit/LLMHub)<font style="color:rgb(25, 27, 31);">，欢迎关注收藏！</font>

希望大家带着下面的问题来学习，我会在文末给出答案。

1.  传统 Attention 的主要性能瓶颈在哪里？为什么需要 FlashAttention？  
2.  FlashAttention 是如何利用 shared memory 降低显存占用并提高速度的？     
3.  FlashAttention 在实际应用中还有哪些不足或限制？    

## 一、引言
Transformer 模型自诞生以来，已成为自然语言处理、计算机视觉、语音等领域的核心架构。而 Attention 机制作为 Transformer 的核心计算模块，其计算复杂度和显存占用在处理长序列时常常成为性能瓶颈。传统 Attention 的时间和空间复杂度为 O(n^2)，这在大规模模型或长文本输入中表现为效率低下、显存不足。

FlashAttention 是由 Stanford Hazy Research 团队提出的一种高效实现方式，专为 GPU 设计，通过 I/O 感知`（IO-aware）`的优化策略，在不损失精度的前提下显著加速 Attention 计算并降低显存占用。

## 二、Attention回顾
在标准 Transformer 中，Attention 计算如下：

![image](https://cdn.nlark.com/yuque/__latex/5fafaa19c7c5a81b97b8ef07dae68b68.svg)

其中：

+ Q（Query）、K（Key）、V（Value）为输入的线性变换结果；
+ QK^T生成的是 Attention Score 矩阵，大小为 (n, n)，即每个 token 对所有 token 的相关性。

计算流程如下：

1. 计算 QK^T：需存储一个n*n 矩阵；
2. 计算 softmax；
3. 乘以 V 得到最终输出。

**瓶颈分析：**

+ 需要在 GPU 上存储整个 QK^T 结果 → 显存开销大；
+ 需要频繁访问 HBM（高带宽显存）→ 带宽受限 → 性能下降。

## 三、FlashAttention 的核心思想（核心原理）
FlashAttention 的目标是：**降低显存占用，同时提升速度**，具体做法包括：

### 1. Tile-Based 计算
将 Q, K, V 分块为小块（tile），每次仅处理一小块：

+ 利用 GPU 的片上 SRAM（Shared Memory）完成 QK^T 和 softmax
+ 避免中间结果写入 HBM

![](https://cdn.nlark.com/yuque/0/2025/png/28454971/1751333049874-29600be6-4c19-4140-8280-3666d217ba9e.png)

上面图中左半部分是计算机的内存分布， HBM 是 “`High Bandwidth Memory`” 的缩写，也就是**高带宽显存**，是一种专为**高性能计算和显存密集型任务**（如 GPU、AI 加速、图形渲染等）设计的**下一代显存技术**。 <font style="color:rgb(25, 27, 31);">SRAM是一种静态随机访问存储器，用于高速缓存等内部存储器，具有更快的访问速度和更低的延迟，但成本更高且占用更多芯片空间。</font>

<font style="color:rgb(25, 27, 31);">标准Attention的计算算法如下：</font>

![](https://cdn.nlark.com/yuque/0/2025/png/28454971/1751333337519-f404f96f-34e6-4e86-be25-c034e38b7ff4.png)

<font style="color:rgb(25, 27, 31);">可以看到，</font>**标准 Attention 实现大量中间结果需频繁访问 HBM**<font style="color:rgb(25, 27, 31);">，而 HBM 的访问速度远远低于 GPU 的SRAM。因此 FlashAttention 通过“</font>**tile 计算+显存访问优化**<font style="color:rgb(25, 27, 31);">”方案，</font>**减少了对 HBM 的依赖，提高了整体执行效率**<font style="color:rgb(25, 27, 31);">。  </font>

<font style="color:rgb(25, 27, 31);">softmax计算公式如下：</font>

![](https://cdn.nlark.com/yuque/0/2025/png/28454971/1751333858781-d77f2803-3b55-4a8c-aca3-b000e644b7f6.png)

为了数值稳定性，FlashAttention采用Safe Softmax，对于向量`x`

![](https://cdn.nlark.com/yuque/0/2025/png/28454971/1751333995086-5250addd-318f-46b0-a9ac-a813c2c94b9b.png)同理，对于向量x =  [ x1,x2]，softmax可以分解计算：

![](https://cdn.nlark.com/yuque/0/2025/png/28454971/1751333955097-3715f4ff-c975-4aec-a093-d92c4bbd714a.png)

这就说明即使Q,K,V被分成块也是可以计算softmax的。

### 2. Recomputation Strategy
为了节省存储中间的 softmax 权重，FlashAttention 在需要时**重新计算部分内容**，避免保存完整矩阵。

标准Attention的反向传播算法如下，其中P代表**Softmax(QKᵀ / √dₖ)，也就是注意力权重矩阵。**

结合着Attention的计算公式更好理解

![image](https://cdn.nlark.com/yuque/__latex/5fafaa19c7c5a81b97b8ef07dae68b68.svg)

![](https://cdn.nlark.com/yuque/0/2025/png/28454971/1751334861591-c197f8b5-d69b-462c-8a29-467e8ca6420c.png)

在标准 Attention 实现中，为了完成前向传播和反向传播，我们通常需要保存如下中间结果：

+ QKᵀ（Attention Score 矩阵）
+ softmax 权重
+ Attention output（最终结果）

这些矩阵很大，尤其是在处理长序列时，显存消耗会非常高。

FlashAttention 为了**降低显存占用**，采取了一种策略：

> 在前向传播时 **不保留中间矩阵**，而是到了反向传播阶段 **再把它们重新计算出来**。
>



以 softmax 的 attention score 为例：

+ **标准方法：**

```latex
QKᵀ → softmax → 缓存在显存中 → 用于乘V和反向传播
```

+ **FlashAttention 方法：**

```latex
QKᵀ → softmax → 直接用于乘V，不缓存
...
后面反向传播需要用到 softmax → 再重新计算一次 QKᵀ 和 softmax
```

这就节省了存 softmax 的显存开销，尤其在长序列上非常可观。

FlashAttention的前向传播算法如下：

![](https://cdn.nlark.com/yuque/0/2025/png/28454971/1751334603300-602b20a2-5e95-407d-a4b2-a986d3d938b5.png)

FlashAttention的反向传播的过程如下：

![](https://cdn.nlark.com/yuque/0/2025/png/28454971/1751335193836-91fc30b6-0991-4d43-8c3b-13c457d52f4a.png)

可以看到其中没有存储，反向传播的过程中需要的数据都是重新计算的，这种“以算代存”的方式是一种典型的**时间换空间**（compute vs. memory）策略。虽然多计算一次会略微增加一点时间，但显存节省得非常明显，反而提升了整体性能，因为：

+ 显存访问慢，限制吞吐；
+ GPU 有大量计算资源，计算冗余可以承受；
+ 避免了 HBM 带宽瓶颈。

### 3.Block Sparse FlashAttention 
传统 Attention 是 **全连接的**：每个 token 都和所有其他 token 交互，计算量为 O(n^2)。

而 **Sparse Attention** 只计算部分 token 对的关系，常见稀疏模式包括：

+ **Sliding Window**：每个 token 只关注它前后几个邻居；
+ **Block Sparse**：将 Q、K、V 分成若干块（block），只计算特定 block 对之间的 attention；
+ **Global + Local**：大部分是局部 attention，少数 token 有全局连接（如 Longformer）。

**在FlashAttention 的基础上，为了进一步提升处理超长序列的性能和可扩展性，Block Sparse FlashAttention** 结合了 FlashAttention 的 IO-aware 高效计算方式和 **block-sparse attention mask** 的稀疏结构，从而实现 **更少计算 + 更少显存占用** 的 attention 操作。

Block Sparse FlashAttention 的关键是在 FlashAttention 高效计算的基础上，**只计算被稀疏掩码指定的 QK 块对**，算法如下：

![](https://cdn.nlark.com/yuque/0/2025/png/28454971/1751335842571-528a907c-1115-4c4a-b500-09753521f56c.png)

1. 输入：Q、K、V 被划分为若干 block；
2. 依据稀疏掩码（mask）决定哪些 Q-block 要与哪些 K-block 交互；
3. 对每个有效块对，执行 FlashAttention 核心流程（QKᵀ → softmax → ×V）；
4. 将结果整合，拼接为完整输出。

## 四、FlashAttention vs 标准Attention
| 项目 | 原始 Attention | FlashAttention |
| --- | --- | --- |
| 时间复杂度 | O(n^2 d) | O(n^2 d)，但更快 |
| 显存消耗 | 高（存储中间矩阵） | 低（tile重计算） |
| 速度表现 | 慢（受限于显存读写） | 快（高效访存） |
| 精度控制 | float32 为主 | 支持 fp16 / bf16 |


在长序列任务中，FlashAttention 可将显存减少 2-4 倍，速度提升达 2-4 倍。

## 五、从0手撸FlashAttention
```python
for i in range(0, N, block_size):
    q_block = q[:, i:i+block_size]  # [B, Bq, D]
    max_score = None
    row_sum_exp = None
    acc = torch.zeros_like(q_block)

    for j in range(0, N, block_size):
        k_block = k[:, j:j+block_size]  # [B, Bk, D]
        v_block = v[:, j:j+block_size]  # [B, Bk, D]

        # 1. Attention logits
        scores = torch.bmm(q_block, k_block.transpose(1, 2)) * scale  # [B, Bq, Bk]

        # 2. Numerical stability
        block_max = scores.max(dim=-1, keepdim=True).values  # [B, Bq, 1]
        scores = scores - block_max
        exp_scores = scores.exp()  # [B, Bq, Bk]

        # 3. Dropout (可选)
        if dropout_p > 0.0:
            exp_scores = F.dropout(exp_scores, p=dropout_p, training=True)

        # 4. Weighted sum
        acc += torch.bmm(exp_scores, v_block)  # [B, Bq, D]

        # 5. Softmax normalization (log-sum-exp trick)
        block_sum = exp_scores.sum(dim=-1, keepdim=True)  # [B, Bq, 1]
        if row_sum_exp is None:
            row_sum_exp = block_sum
            max_score = block_max
        else:
            row_sum_exp += block_sum
            max_score = torch.max(max_score, block_max)

    # Normalize accumulated result
    output[:, i:i+block_size] = acc / (row_sum_exp + 1e-6)

return output
```

**要注意的是 上面的PyTorch 实现并没有用到 Shared Memory**，它只是演示了 FlashAttention 的思想流程。  
真正利用了 SRAM 的，是 **FlashAttention 的 CUDA kernel 或 Triton kernel 实现**。  

如果想要测试效率，可以直接调用torch封装好的flashattention

```python
from flash_attn.modules.mha import FlashMHA
import torch

x = torch.randn(8, 512, 512, device='cuda')  # batch, seq_len, dim
mha = FlashMHA(embed_dim=512, num_heads=8, device='cuda')
output = mha(x)
print(output.shape)  # [8, 512, 512]
```



## 六、总结
FlashAttention 提供了一种高效、低显存的 Attention 实现方式，极大地缓解了 Transformer 模型在长序列处理中的性能瓶颈。在当前大模型时代，FlashAttention 成为高效训练与部署的关键组件之一。





<font style="color:rgb(25, 27, 31);">最后，我们回答一下文章开头提出的问题。</font>

1. <font style="color:rgb(25, 27, 31);">传统 Attention 的主要性能瓶颈在哪里？为什么需要 FlashAttention？  
</font><font style="color:rgb(25, 27, 31);">标准的 Attention 实现存在两个严重问题：</font>
+ **显存占用高**<font style="color:rgb(25, 27, 31);">：完整计算 attention 需要构造形如 </font>`<font style="color:rgb(25, 27, 31);">[Batch, Head, SeqLen, SeqLen]</font>`<font style="color:rgb(25, 27, 31);"> 的 score 矩阵 QKᵀ，即 O(n^2) 的显存需求；</font>
+ **访存带宽瓶颈**<font style="color:rgb(25, 27, 31);">：计算过程中，Q、K、V、score、softmax 权重、输出 O 都需要多次读写 global memory（HBM），而 GPU 的计算能力往往无法完全发挥出来。</font>

<font style="color:rgb(25, 27, 31);">FlashAttention 被提出，目标就是通过“在 shared memory 中块级 tile 化 attention 计算”，</font>**避免 score 的 materialization 和重复访存**<font style="color:rgb(25, 27, 31);">，从而提升效率、减少内存压力。</font>



2. <font style="color:rgb(25, 27, 31);">FlashAttention 是如何利用 shared memory 降低显存占用并提高速度的？  
</font><font style="color:rgb(25, 27, 31);">FlashAttention 的关键设计是：</font>
+ <font style="color:rgb(25, 27, 31);">将 Q/K/V 分为小块（tiles），在 shared memory（即 SRAM）中进行 attention 的计算；</font>
+ <font style="color:rgb(25, 27, 31);">在计算 softmax 的过程中使用 log-sum-exp 技巧，确保数值稳定；</font>
+ <font style="color:rgb(25, 27, 31);">将 softmax 后与 V 的乘法也集成进 tile 内的计算流程，避免生成大矩阵；</font>
+ <font style="color:rgb(25, 27, 31);">利用 recomputation：</font>**不存储 softmax 权重 P，而是在反向传播时重算 QKᵀ**<font style="color:rgb(25, 27, 31);">，换取显存节省。</font>



3. <font style="color:rgb(25, 27, 31);">FlashAttention 在实际应用中还有哪些不足或限制？  
</font><font style="color:rgb(25, 27, 31);">尽管 FlashAttention 在性能和显存方面带来显著改善，但也存在一些实际问题：</font>
+ **线程并行效率不高**<font style="color:rgb(25, 27, 31);">：使用的是 “1 warp 对应 1 Q 行” 的划分方式，warp 内线程空闲率高；</font>
+ **split-K 导致频繁 HBM 读写**<font style="color:rgb(25, 27, 31);">：每次 tile 操作都要访问 Q 和 O，存在冗余累加；</font>
+ **不支持 MQA / GQA 等高效注意力结构**<font style="color:rgb(25, 27, 31);">：仅适用于标准 MHA；</font>
+ **实现依赖 Triton 编译器**<font style="color:rgb(25, 27, 31);">：对部署平台要求高，难以在 PyTorch、TensorFlow 等框架中原生集成；</font>
+ **反向传播内核较少优化**<font style="color:rgb(25, 27, 31);">：精度和性能兼顾方面还有改进空间。</font>

 

<font style="color:rgb(25, 27, 31);">关于深度学习和大模型相关的知识和前沿技术更新，</font>请关注公众号`<font style="color:rgb(51, 51, 51);">算法coting</font>`<font style="color:rgb(51, 51, 51);">!</font>

<font style="color:rgb(25, 27, 31);"></font>

以上内容部分参考了

[FlashAttention:Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/pdf/2205.14135)

[Flash Attention原理详解(含代码讲解)](https://zhuanlan.zhihu.com/p/676655352)

非常感谢，如有侵权请联系删除！



