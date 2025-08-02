本篇文章是Transformer系列的第九篇。

Transformer系列文章：

● [一览Transformer整体架构](https://zhuanlan.zhihu.com/p/1918047303597552480)  
● [Transformer——Attention怎么实现集中注意力](https://zhuanlan.zhihu.com/p/1918049072331362469)  
● [Transformer——FeedForward模块在干什么？](https://zhuanlan.zhihu.com/p/1918050616376301224)  
● [从0开始实现Transformer](https://zhuanlan.zhihu.com/p/1918357249883105145)  
● [什么是KV-Cache](https://zhuanlan.zhihu.com/p/1919338888536756837)  
● [Transformer注意力机制——MHA&MQA&GQA](https://zhuanlan.zhihu.com/p/1919500956946655189)  
● [FlashAttention怎么提升速度的？](https://zhuanlan.zhihu.com/p/1923328314241704991)  
● [FlashAttention2：更快的注意力机制，更好的并行效率](https://zhuanlan.zhihu.com/p/1923714840653993730)



<font style="color:rgb(25, 27, 31);">所有相关源码示例、流程图、模型配置与知识库构建技巧，我也将持续更新在Github：</font>[**<font style="color:rgb(25, 27, 31);">LLMHub</font>**](https://github.com/zhangting-hit/LLMHub)<font style="color:rgb(25, 27, 31);">，欢迎关注收藏！</font>

希望大家带着下面的问题来学习，我会在文末给出答案。

1. FlashAttention3和FlashAttention1和FlashAttention2有什么区别？
2. FlashAttention3的核心机制是什么？ 
3. FlashAttention3还有什么可以改进的地方？  

## 1. 引言
自 Transformer 在《Attention is All You Need》中问世以来，Attention 模块迅速成为了深度学习模型的核心组件。尤其在大模型（如 GPT、LLaMA、BERT）中，Attention 不仅承担信息整合的关键角色，也成为了计算资源消耗最多的模块之一。

然而，原始的 Attention 计算存在两个关键瓶颈：

+ **内存开销大**：需要在显存中存储整个 attention matrix（尤其是长序列时）。
+ **计算效率低**：依赖低效的访存模式，严重受限于 GPU 的内存带宽。

为此，论文《FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness》提出了 FlashAttention 系列，通过精细设计的 **kernel fusion** 和 **tile-based 计算策略**，显著提升了效率，极大地缓解了显存压力。

如今，**FlashAttention-3** 进一步将这些优化推向极致，在加速 Transformer 模型推理和训练方面，展现出前所未有的性能优势。

## 2. FlashAttention 回顾
### 2.1 FlashAttention-1：重计算换内存的经典范式
![](https://cdn.nlark.com/yuque/0/2025/png/28454971/1751421813974-ebdc7702-0fab-4c14-96a4-b96a1e0f3440.png?x-oss-process=image%2Fformat%2Cwebp)

FlashAttention 的初代版本（2022）引入了 “tile-based 重计算策略”：

+ 将输入序列划分为小块（tiles），分批次计算 attention 分数。
+ 在计算 softmax 过程中使用数值稳定的 **分段归一化策略**，无需保存完整 attention matrix。
+ 虽然会带来一些重复计算（recompute），但极大节省了显存（显存开销从 O(n^2) 降为 O(n)）。

效果：训练速度提升 2-4 倍，显存占用显著降低。

### 2.2 FlashAttention-2：kernel 融合与高效反向传播
![](https://cdn.nlark.com/yuque/0/2025/png/28454971/1751425997072-d6f045d8-4538-4119-90dc-dbe99882ba78.png?x-oss-process=image%2Fformat%2Cwebp)

FlashAttention-2（2023）更进一步，将 forward 和 backward 统一为单个 CUDA kernel：

+ **统一的 tiled kernel**：使前向和反向传播共享 memory access 和数据加载逻辑。
+ **Warp-level pipeline 并行**：充分利用 CUDA warp 和 GPU 流水线，减少

效果：相比 FA1，在长序列任务中加速进一步提升 20%-30%。

## 3. FlashAttention3 的核心技术架构
在 Transformer 的 Attention 计算中，两个关键步骤贯穿始终：**GEMM（通用矩阵乘法）** 和 **softmax**。

其中，GEMM 主要用于计算注意力权重 `Q × Kᵀ` 和最终输出 `softmax(QKᵀ) × V`，其本质是高吞吐、结构化的矩阵乘法操作，**可以完全由 GPU 的 Tensor Core 高速执行**，在现代硬件上能达到上万亿次浮点运算每秒（TFLOPs）级别。

而 softmax 则是一种逐元素归一化操作，涉及指数函数和除法，虽然计算量不大，但**难以在 Tensor Core 上并行提速**。在 H100 上测得的结果显示，**softmax 的吞吐量仅为 GEMM 的 1/250 左右**，在流水线中极易成为瓶颈。

正是因为 **GEMM 快而 softmax 慢**，FlashAttention-3 为了解决这两者之间的速度鸿沟，设计了一种 **异步执行机制**：通过将不同的 warp group 分别指定为「计算 GEMM」和「执行 softmax」，并让它们交替运行、互不等待，从而实现真正意义上的流水线并行。这种方式极大地减少了软操作的阻塞，显著提升了整体执行效率。

FlashAttention-3 的核心目标是：**在不牺牲精度的前提下，最大程度榨干 GPU 的算力**，特别是针对 NVIDIA Hopper 架构（如 H100）。论文提出了三项关键技术，我们一项项来拆解。

### 3.1 Producer-Consumer 异步机制（Warp Specialization + TMA）
传统 GPU kernel 是同步执行的：一个线程块中的所有线程要一起等待数据加载，然后一起计算。这导致大量时间浪费在 **等待内存读取** 上，尤其当 Attention 操作牵涉大量 Q/K/V 数据时尤为明显。

#### 3.1.1 FlashAttention-3 的方案：分工协作 + 异步加载
它借助 Hopper GPU 提供的两个特性：

1. **TMA（Tensor Memory Accelerator）**：可以在后台异步从 HBM（显存）加载数据到共享内存（SMEM）； 它根据设置好的 tile 参数，从内存中**一块块地搬运矩阵数据**到共享内存中，而计算线程则在另一边执行 GEMM。

![](https://cdn.nlark.com/yuque/0/2025/png/28454971/1751532974659-9f151985-4182-47e8-8139-4e263c93ccb6.png)

2. **Warp Specialization（warp专用化）**：将线程块中的 warp 分成两类：
    - **Producer Warp**：专门负责数据搬运，从 HBM 异步加载 Q/K/V；
    - **Consumer Warp**：专门负责计算，如 QK^T、softmax、乘 V。

这就像一个流水线：**前面的人（Producer）不停搬砖，后面的人（Consumer）拼命砌墙。**

#### 3.1.2 技术细节
+ 使用一个 **环形共享内存缓存（circular SMEM buffer）** 存放 tile 数据；
+ 通过 barrier 指令实现 Producer 和 Consumer 的同步协作；
+ 支持多个 tile 的并发调度，确保计算不会因数据等待而停滞。

![](https://cdn.nlark.com/yuque/0/2025/png/28454971/1751531351688-8151901e-d9e1-43b2-986b-6628c3a9a996.png)

这是从一个 **CTA（Cooperative Thread Array，通常是一个 CUDA Block）** 的角度描述整个计算过程的主调度逻辑。它把 attention 的输入序列分成若干个 tile，每个 tile 的数据通过 producer warp 加载后，被 consumer warp 执行具体的 attention 计算。

主要流程：

1. **初始化**：
    - 设置线程块中哪些 warp 是 producer，哪些是 consumer；
    - 为每个 tile 分配缓冲区（`smem_qkv`, `smem_s`, 等）；
    - 使用 barrier（`barr`) 进行同步。
2. **tile 级迭代**（外部循环）：
    - Producer warp：
        * 从 global memory 加载 Q/K/V block；
        * 通过 TMA 异步搬到 shared memory；
        * 使用 `barrier` 通知 Consumer warp；
    - Consumer warp：
        * 等待 barrier；
        * 调用 `CONSUMER_WARPGROUP_FORWARD_PASS`（即 Algorithm 2）进行计算；
        * 等待 write barrier，再处理下一个 tile。

本质上，它是 tile-level 的数据流调度者，管理了：

+ Producer 和 Consumer 的交替；
+ 各 tile 的加载和处理顺序；
+ 多 warp group 间的同步协调。

**本质突破**：让计算线程在数据还没加载完成时，也能提前干别的活，**隐藏 memory latency。**



### 3.2 交错执行 GEMM 与 Softmax（Pingpong Scheduling）
#### **<font style="color:rgb(25, 27, 31);">3.2.1 </font>**背景问题
在 Attention 中，softmax 虽然 FLOPs 很少，但它涉及指数函数、除法等操作，**吞吐量远低于矩阵乘法**，在 H100 上甚至只有 1/250 的吞吐。

这就像高速公路上的一辆慢车——哪怕只有 10 米，也能严重阻塞整个车流。

#### **<font style="color:rgb(25, 27, 31);">3.2.2 </font>**FlashAttention3 的方案：GEMM 和 Softmax「交叉执行」
+ 利用 Hopper 的异步 WGMMA（Warpgroup Matrix Multiply Accumulate）指令；
+ 将 warp group 分成两批：
    - warp group A：当前执行 softmax；
    - warp group B：当前执行 GEMM；
+ 下一轮 A 和 B 对调，形成“乒乓”节奏。

#### **<font style="color:rgb(25, 27, 31);">3.2.3 </font>**举例说明
![](https://cdn.nlark.com/yuque/0/2025/png/28454971/1751513069081-dd09cf7b-a378-4448-8f28-9351b78e8056.png)

+ 第 1 步：warp group A 执行 softmax(S1)，warp group B 执行 QK^T(S2)；
+ 第 2 步：warp group A 执行 QK^T(S3)，warp group B 执行 softmax(S2)；
+ ...

这就构建了一个**跨 warp 的指令流水线**，让慢操作“藏”在快操作的阴影中。

![](https://cdn.nlark.com/yuque/0/2025/png/28454971/1751530014461-90fde1ed-1c97-4936-8b1d-38572dcf8efb.png)

在一个warpgroup中，同样将softmax中的一些指令和GEMM中的一些指令重叠，在上面的算法中，内部循环内的操作具有顺序依赖性，这阻碍了单词迭代内的并行化，例如19行softmax操作依赖于第一个GEMM结果S，21行的GEMM操作依赖19行的softmax结果P。这就造成了softmax和GEMM的串行。

flashattention3具体优化算法如下：

![](https://cdn.nlark.com/yuque/0/2025/png/28454971/1751530648286-7ffcf6df-d0bf-4dd2-b512-a2a0ef696d35.png)



其中WGMMA也可以理解为GEMM，可以看到，迭代的第二个人WGMAA操作（第11行）和第二次迭代的softmax（第十三行）操作重叠，如下图

![](https://cdn.nlark.com/yuque/0/2025/png/28454971/1751530883028-b0caa0c3-d6e2-48f0-aea1-ac2e757f5aee.png)

**本质突破**：在 Tensor Core 忙碌计算的时候，不浪费其它硬件单元的资源（如多功能单元执行 exp），**最大化利用硬件并发性。**

 Algorithm 1 从线程块（CTA）的全局视角出发，负责调度 producer 和 consumer warp group 的协同工作，组织 tile 级别的 attention 计算流程；而 Algorithm 2 则细化了其中 consumer warp group 的具体执行逻辑，定义了在处理每个 tile 时如何分块计算 QK^T、执行 softmax 并累加输出。因此，Algorithm 2 实质上是 Algorithm 1 中 consumer 部分的具体展开和核心计算实现，两者构成了 FlashAttention-3 前向传播的整体框架。  

```swift
Algorithm 1 调度结构（CTA 层级）
└── Producer Warp
│   └── 加载 Q/K/V → 写入共享内存
└── Consumer Warp Group
    └── ← Algorithm 2（计算 Attention）
         └── 多个 subtile：
             └── QK^T → softmax → 乘 V → 累加

```

### 3.3 FP8 低精度支持（高性能 × 高精度）
#### 3.3.1 背景问题
使用 FP16 能加速计算，但在 H100 上，使用 **FP8 精度可以再提速 2×**。然而：

+ FP8 只有 3 位尾数，量化误差很大；
+ Attention 中容易出现 **outlier 特征值**，被量化后信息丢失严重；
+ Hopper 对 FP8 的 **数据排布要求严格**，不满足就不能运行 WGMMA。

#### 3.3.2 FlashAttention3 的方案：结合三项技术，兼顾速度与精度
![](https://cdn.nlark.com/yuque/0/2025/png/28454971/1751513291363-700aa10e-9c4e-47fd-9ad6-eadecce8702d.png)

##### <font style="color:rgb(25, 27, 31);">3.3.2.1</font> 内核内布局转换（in-kernel transpose）
+ FP8 要求的 V 必须按 k-major（行主序）存储，而模型输出通常是 head-major。
+ 使用 warp 别的 **LDSM / STSM 指令**（加载/存储 8x8 block）：
    - 实现 **在共享内存中边加载边转置**；
    - 可隐藏在前后 GEMM 的执行间隙中，**几乎无额外开销**。

##### <font style="color:rgb(25, 27, 31);">3.3.2.2</font> Block Quantization（块量化）
+ 将 Q/K/V 划分为多个 tile，每个 tile 单独使用一个 scale：
    - 比 per-tensor scale 精度更高；
    - 可与 rotary embedding 等前置模块融合，无额外代价。

##### <font style="color:rgb(25, 27, 31);">3.3.2.3 </font>Incoherent Processing（正交扰动）
+ Q/K 在量化前先乘一个**正交扰动矩阵 M（Hadamard + ±1 随机对角矩阵）**：
    - 不改变最终 Attention 输出（QK^T 不变）；
    - 能“打散” outlier，**让每个 tile 的数值分布更均匀、可量化。**

效果：FP8 模式下精度比传统 FP8 attention 提高 **2.6×**，运行速度达到 **1.2 PFLOPs/s**。



<font style="color:rgb(25, 27, 31);">最后，我们回答一下文章开头提出的问题。</font>

1. FlashAttention3和FlashAttention1和FlashAttention2有什么区别？

FlashAttention 系列的发展路径可以总结为一次次**瓶颈突破**：

| 版本 | 核心提升 | 技术特点 |
| --- | --- | --- |
| **FA-1** | 显存优化 | 使用 tiling + recompute softmax，显著节省显存，但没有处理计算瓶颈 |
| **FA-2** | 速度提升 | 使用 fused kernel，支持 dropout、causal mask，速度更快，代码更工程化 |
| **FA-3** | **极致性能释放** | 引入 **异步 Producer/Consumer warp group 调度 + 2-stage softmax-GEMM pipeline + TMA 加载机制**，几乎跑满 H100，达到理论上限的 77-90% TFLOPs 使用率 |


FlashAttention-3 并非简单优化，而是重构了整个执行模型，从线程粒度到 warp group 级别进行了彻底的调度机制革新，真正意义上将 memory latency「隐藏」在计算背后。

---

2. ** FlashAttention-3 的核心机制是什么？**

FlashAttention-3 的三大核心机制是：

> 1. **Warp Specialization（Producer/Consumer 解耦）**
>     - 将 warp 分成搬运数据的 Producer warp group 和专注计算的 Consumer warp group，实现异步加载与执行。
> 2. **TMA（Tensor Memory Accelerator）异步加载**
>     - 利用 `tma.load_async` 预取 Q/K/V tiles 到共享内存，搬运不再阻塞主计算流程。
> 3. **2-Stage Consumer Pipeline（见 Figure 2）**
>     - Consumer warp group 内部将 softmax 和 GEMM 解耦，通过交叉流水执行，**隐藏 slow op，放大 fast op 的吞吐优势**。
>

这套机制组合后，实现了 GPU 计算资源的最大化利用，尤其在软操作（如 softmax）难加速的背景下，依然保持整体高效。

---

3. FlashAttention-3 还有哪些可以改进的地方？

尽管 FA-3 已经非常接近硬件性能上限，但仍存在一些值得探索的方向：

> 1. **跨层 KV 缓存重用优化**
>     - 多层 Attention 共用 KV 时仍存在冗余计算和加载，可考虑引入跨层 prefetch 或 KV compression。
> 2. **支持更广泛的序列模式**
>     - 当前主要优化 dense attention，对于 sparse attention（如 long context RAG、prefix routing）优化尚有限。
> 3. **更灵活的硬件适配**
>     - FA-3 针对 Hopper 架构（如 H100）设计，其他架构（如 A100、L4）效果下降，仍需软硬件协同优化版本。
> 4. **训练优化方向**
>     - 当前重点在推理速度，未来版本可尝试将这些技术推广到训练阶段，特别是大模型 pretraining 时的效率优化。
>





<font style="color:rgb(25, 27, 31);">关于深度学习和大模型相关的知识和前沿技术更新，</font>请关注公众号`<font style="color:rgb(51, 51, 51);">算法coting</font>`<font style="color:rgb(51, 51, 51);">!</font>

<font style="color:rgb(51, 51, 51);"></font>

以上内容部分参考了

[Flash Attention 3 深度解析](https://zhuanlan.zhihu.com/p/17533058076)

[https://arxiv.org/pdf/2407.08608](https://arxiv.org/pdf/2407.08608)

非常感谢，如有侵权请联系删除！

