本篇文章是Transformer系列的第八篇。

Transformer系列文章：

● [一览Transformer整体架构](https://zhuanlan.zhihu.com/p/1918047303597552480)  
● [Transformer——Attention怎么实现集中注意力](https://zhuanlan.zhihu.com/p/1918049072331362469)  
● [Transformer——FeedForward模块在干什么？](https://zhuanlan.zhihu.com/p/1918050616376301224)  
● [从0开始实现Transformer](https://zhuanlan.zhihu.com/p/1918357249883105145)  
● [什么是KV-Cache](https://zhuanlan.zhihu.com/p/1919338888536756837)  
● [Transformer注意力机制——MHA&MQA&GQA](https://zhuanlan.zhihu.com/p/1919500956946655189)  
● [FlashAttention怎么提升速度的？](https://zhuanlan.zhihu.com/p/1923328314241704991)



<font style="color:rgb(25, 27, 31);">所有相关源码示例、流程图、模型配置与知识库构建技巧，我也将持续更新在Github：</font>[**<font style="color:rgb(25, 27, 31);">LLMHub</font>**](https://github.com/zhangting-hit/LLMHub)<font style="color:rgb(25, 27, 31);">，欢迎关注收藏！</font>

希望大家带着下面的问题来学习，我会在文末给出答案。

1.  FlashAttention 1 已经 tile 化并优化了 softmax，为什么在实际训练中仍然不是最优？  
2.  FlashAttention 2 的内核重写带来了哪些底层机制变革？它如何真正“解锁 GPU 潜力”？  
3. <font style="color:rgb(25, 27, 31);"> FlashAttention 2 是如何在兼顾高效和精度的同时，支持多精度训练与反向传播优化的？  </font>

## 一、引言
Transformer 架构中，注意力机制是性能瓶颈所在。虽然 FlashAttention 1 已大幅降低了显存和加速计算，但在实际大模型训练中，它仍存在几个重要问题：

+ **线程并行效率不高**：FlashAttention 1 的并行粒度设计不当，导致 warp 内线程使用率不足；
+ **支持模式受限**：不支持如 Multi-Query Attention (MQA)、Grouped Query Attention (GQA) 等优化注意力结构；
+ **兼容性差**：基于 Triton 的实现对部署环境要求较高，难以集成到大多数框架中。

为此，Dao AI Lab 在 2023 年提出 FlashAttention 2，通过全新的 CUDA kernel 设计，实现了更快、更稳、更通用的注意力计算。

## 二、Attention 与 FlashAttention 1 简要回顾
### 标准 Attention 计算回顾
给定输入序列经过线性映射得到 Q, K, V，注意力输出为：

![image](https://cdn.nlark.com/yuque/__latex/5fafaa19c7c5a81b97b8ef07dae68b68.svg)

在实际实现中，该公式存在两个瓶颈：

1. **中间结果庞大**：需要构造和保存 $QK^T$ 矩阵，占用大量显存；
2. **访存带宽受限**：Q、K、V 多次从 HBM 读写，耗时大。

### FlashAttention 1 的优化策略
![](https://cdn.nlark.com/yuque/0/2025/png/28454971/1751421813974-ebdc7702-0fab-4c14-96a4-b96a1e0f3440.png)

+ **Tile-based 计算**：将序列划分为小块，用 shared memory（SRAM）处理每块 attention；
+ **Softmax 内核优化**：采用 log-sum-exp 技巧，避免精度损失；
+ **Recomputation 策略**：中间不存储 softmax 权重，反向传播时重算，节省显存。

这些优化让 FlashAttention 1 在长序列任务上显著降低显存并提高速度。

### FlashAttention 1 的局限
![](https://cdn.nlark.com/yuque/0/2025/webp/28454971/1751422021830-ee35bf5a-b4c6-4963-9505-5f821d9373f6.webp)

+ Warp 内线程使用率不高；
+ 仅支持标准 MHA 模式；
+ Triton 实现对底层环境要求较高，编译困难。

## 三、FlashAttention 2 的核心原理
### 3.1.减少non-matmul FLOPs(非矩阵乘法操作)
在注意力机制中，除了主计算 `Q × K^T` 和 `P × V`（属于矩阵乘法，matmul FLOPs）外，还包括很多非矩阵乘法操作：

+ 归一化（softmax）相关操作，如 `exp`、`sum`、`divide`
+ 维度缩放（scaling）`QK / sqrt(d_k)`
+ 累加和重归一化（为防止数值溢出）
+ 求最大值（用于 log-sum-exp 的 softmax）
+ Recompute softmax during backward（如果不保存中间结果）
+ 多次读写 memory（带来的 memory-bound 计算）

这些操作虽不是 matmul，但占据了大量运算开销，尤其在反向传播阶段尤为明显。

FlashAttention的计算方式如下：

![](https://cdn.nlark.com/yuque/0/2025/png/28454971/1751424909779-7f638bb4-3efc-46c2-b8ba-c7e4970ba265.png)

FA1 在计算 QK^T、softmax 和乘 V 的过程中分别调用多个 kernel，导致中间结果写入 global memory。

FlashAttention2的计算方式如下：

![](https://cdn.nlark.com/yuque/0/2025/png/28454971/1751425255878-4fd9414b-00bc-4228-a198-f3dce218da7f.png)

FlashAttention-2 通过以下 **三种关键方式减少 non-matmul FLOPs**：

前向传播算法如下

![](https://cdn.nlark.com/yuque/0/2025/png/28454971/1751425789535-980f78bd-5e3e-45e0-ba0a-9555e82be8b2.png)

可以看到不同于FlashAttention K，V在外循环，Q在内循环，FlashAttention2将Q移到了外循环，K，V在内循环。

#### 1. 精简 softmax 实现（带 log-sum-exp 变体）
FlashAttention-2 使用 numerically stable softmax：

![image](https://cdn.nlark.com/yuque/__latex/41140ba2d87d4228767356b06ffebbdb.svg)

但它只做一次 log-sum-exp 归一化，**没有每一次都算一次分母，只有最后才除了一次**，并且将 softmax、缩放、加权累加一步完成，避免以下重复：

+ 不再多次遍历 QK 值找最大值；
+ softmax 权重无需单独保存和归一化；
+ 一些操作可 fuse 到 QK 计算和 `×V` 中完成。

因此减少了 `exp`、`sum`、`divide`、`max` 等操作的 FLOPs 数量。

#### 2. Fused kernel 将非 matmul 操作融合进主计算流
传统做法中：

```latex
QK^T → softmax → matmul with V
```

需要执行多个 kernel，每一步都有中间结果写入内存且包含大量非 matmul FLOPs。

FA2 的做法：

```latex
QK^T + softmax + ×V 全部在一个 kernel 内完成
```

这个 fused kernel 带来的优势是：

+ 不再有中间 buffer 存储 softmax 权重（节省带宽和不必要的 op）；
+ softmax 与 V 的乘法过程同步进行，不额外调用算子；
+ **避免冗余计算和加载**，比如重复的 scale 和 divide 操作。

总之，大量非 matmul 操作被“压缩”到主流程中执行。



#### 3. Backward 阶段 recomputation 与 minimal FLOP backward kernel
![](https://cdn.nlark.com/yuque/0/2025/png/28454971/1751427994824-9f7c73f4-7687-4f07-9ef5-2d2efc7715b1.png)

FlashAttention-2 提供了多个反向传播 kernel，其中：

+ `minimal FLOPs` 版本最极致减少了反向传播中的非 matmul FLOPs；
+ softmax 权重不存储，recompute 过程中仅保留极小 subset；
+ 使用简化的逻辑仅重建需要反向链条的部分。

### 3.2 Thread Block 划分改进（Work Partitioning）
在理解 FlashAttention 2 的设计前，我们需要先了解一些 GPU 的基础概念。

![](https://cdn.nlark.com/yuque/0/2025/png/28454971/1751422488216-beb7189f-fd27-489f-93b8-d741be5bc600.png)

GPU 是高度并行的处理器，其核心由多个 Streaming Multiprocessors（SM）组成，每个 SM 又包含多个 CUDA 核心。GPU 的并行粒度如下：

+ **线程（Thread）**：最小的执行单位；
+ **线程束（Warp）**：32 个线程组成一个 warp，是 GPU 并行调度的基本单位；
+ **线程块（Thread Block）**：由多个 warp 构成；
+ **网格（Grid）**：包含所有线程块。

每个 warp 的 32 个线程会被同步执行同一段指令（SIMD 模式），如果 warp 内线程分支不同，则会出现 **线程发散（warp divergence）**，降低效率。

因此，高效利用 warp 的并行能力，是优化 GPU kernel 的关键。

FlashAttention 2 的一个核心优化就是：**提升 warp 的利用率，使得 32 个线程同时高效工作，避免资源浪费。**

![](https://cdn.nlark.com/yuque/0/2025/png/28454971/1751425997072-d6f045d8-4538-4119-90dc-dbe99882ba78.png)

FlashAttention 2 引入新的线程组织方式：

+ 将每个 attention row 映射到多个 warp，而不是 1 warp 处理 1 row；
+ 每个 warp 内线程共同处理 QK × softmax × V 的多个 tile；
+ 显著提高并行度，使得 **warp utilization 接近 100%**。

在 FlashAttention 的算法中，K 和 V 是在外部循环中逐块（block）加载的。每次处理一个 K/V 块时，会将该块划分为 4 个 warp 进行并行计算，同时所有 warp 都可以访问对应的 Q。每个 K 的 warp 参与计算 `S_ij = Q × K_j^T`，然后对得到的 `S_ij` 执行局部 softmax，并与对应的 V_j 相乘，得到中间输出贡献 `O_ij`。

然而，为了获得最终的输出 O，每当处理完一个 K/V 块（即外层循环中的 j 增加），都需要将当前 `O_ij` 与之前累积的结果进行合并。这个过程涉及对上一个 `O` 进行缩放（rescale）后加上当前结果。这种做法意味着每个 warp 都必须频繁地从 HBM（高带宽内存）中读取和写入对应的 Q 和 O，以完成中间结果的累加。

这种策略通常被称为“split-K”方案。由于多个 warp 需要不断访问和更新同一输出区域，它会导致频繁的全局内存访问，从而成为性能瓶颈，是一种效率较低的实现方式。

![](https://cdn.nlark.com/yuque/0/2025/png/28454971/1751426527126-6e6f7064-93e7-4d3b-8a81-40929b772553.png)



在 FlashAttention-2 的前向传播中，将对 **Q 的遍历移动到外循环**，而将 **K/V 的遍历移动到内循环**。此外，FlashAttention-2 将每个 Q 向量的处理任务划分给一个线程块，并将该块内的计算进一步拆分为 4 个 warp 并行执行，所有 warp 均可访问当前的 Q、K 和 V。

这种调度方式的核心优势在于：在 FlashAttention 中，内循环每次处理一个新的 K/V 块时，也需要重新加载 Q 并对中间结果 O 执行累加，导致频繁的 HBM（高带宽内存）访问。而 FlashAttention-2 中，**Q 保持不变，存储在高速的 shared memory（SRAM）中**，只在开始时加载一次。随后每个内循环步仅处理新的 K/V tile，避免了对 Q 和中间输出 O 的重复 HBM 读写，从而显著降低了内存访问开销并提升计算效率。

![](https://cdn.nlark.com/yuque/0/2025/png/28454971/1751426464017-27eb258a-76b9-4a4a-9798-6055f03e35bb.png)

### 3.3 MQA / GQA 的原生支持
FlashAttention 2 原生支持 MQA 和 GQA，是第一个能在 **训练阶段** 高效支持这两种注意力结构的通用 kernel，实现上具有如下特点：

#### 1. 多头共享 KV 的并行处理
+ 在 MQA 和 GQA 中，多个 query head 会共享相同的 key/value；
+ FlashAttention 2 对这种共享结构进行了 **kernel 级别的融合**，使多个 query 在访问相同 K/V 时不重复加载，而是复用缓存的数据；
+ 显著提升了计算密度，减少了访存压力。

####  2. 灵活的 head 分组策略（支持任意 GQA 配置）
+ GQA 中，假设有 16 个 query head 被分为 4 组，那么每组对应一组 KV head；
+ FlashAttention 2 在 kernel 启动时就按照 GQA 分组划分好 warp/block，对每组 KV tile 进行处理；
+ 并通过优化后的 warp mapping 保证每个 group 的 query 都能并行处理，**保持 warp 利用率和 tile 复用率**。



<font style="color:rgb(25, 27, 31);">最后，我们回答一下文章开头提出的问题。</font>

1. FlashAttention 1 已经 tile 化并优化了 softmax，为什么在实际训练中仍然不是最优？ 

FlashAttention 1 虽然在理论上减少了显存并提升了性能，但它仍存在几个关键瓶颈：

+ **Split-K 模式导致频繁 HBM 访问**：每处理一个 K/V block 就需要更新输出 O，需要不断读写 Q 和 O，这在大模型中代价极高；
+ **线程块划分低效**：一个 warp 处理一个 Q 行，warp 内并行度低，造成线程资源浪费；
+ **缺乏对高效结构的支持**：如 MQA/GQA 是生成模型中广泛使用的结构，但 FA1 无法高效支持；

这些问题在长序列、预训练阶段尤为明显，导致 FA1 的实用性有限。 

2.  FlashAttention 2 的内核重写带来了哪些底层机制变革？它如何真正“解锁 GPU 潜力”？  

FlashAttention 2 不仅仅是算法上的优化，而是对 CUDA 内核的全面重构，核心变革包括：

+ **完整 fused kernel**：将 QKᵀ → softmax → ×V 三步全部融合在一个 CUDA kernel 中完成，无需中间写入 global memory；
+ **重新定义线程划分策略**：以“一个 block 负责一个 Q 行”的方式，多个 warp 分工处理不同 tile，warp 利用率几乎达 100%，避免线程浪费；
+ **数据驻留在 SRAM（shared memory + register）中**：Q 和中间值不再频繁读写 HBM，极大减少了 memory I/O；
+ **内核支持多种注意力结构（如 causal、MQA、GQA）**：使其能适配 decoder-only 结构或混合架构；
+ **更少非矩阵乘法 FLOPs**：精简 softmax、消除多余 scale/rescale 操作，提升整体运算密度。



3. <font style="color:rgb(25, 27, 31);"> FlashAttention 2 是如何在兼顾高效和精度的同时，支持多精度训练与反向传播优化的？</font>  
FlashAttention 2 不仅关注 forward 性能，还对 **反向传播和数值精度** 做了系统设计：
+ **支持 float16、bfloat16、float32 等多种精度**；
+ **采用 numerically stable softmax**，避免低精度下的数值不稳定；
+ **recomputation 策略**：不保存 softmax 权重矩阵，而在 backward 中重算，节省显存；
+ **三种 backward kernel 模式**：可选择最小 FLOP 路径（minimal-FLOP kernel）、标准路径等，适配不同硬件与任务；
+ **内核融合在 backward 中同样成立**，例如 dQ, dK, dV 的计算与反向 softmax 合并执行，进一步提升速度。



以上就是FlashAttention的全部内容啦，建议大家结合原论文看，理解的会更加深入！



<font style="color:rgb(25, 27, 31);">关于深度学习和大模型相关的知识和前沿技术更新，</font>请关注公众号`<font style="color:rgb(51, 51, 51);">算法coting</font>`<font style="color:rgb(51, 51, 51);">!</font>

<font style="color:rgb(25, 27, 31);"></font>

以上内容部分参考了

[https://arxiv.org/pdf/2307.08691](https://arxiv.org/pdf/2307.08691)

[FlashAttention2详解（性能比FlashAttention提升200%）](https://zhuanlan.zhihu.com/p/645376942)

[FlashAttenion-V3: Flash Decoding详解](https://zhuanlan.zhihu.com/p/661478232)

非常感谢，如有侵权请联系删除！



