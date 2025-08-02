本篇文章是Transformer系列的第十篇。

Transformer系列文章：

● [一览Transformer整体架构](https://zhuanlan.zhihu.com/p/1918047303597552480)  
● [Transformer——Attention怎么实现集中注意力](https://zhuanlan.zhihu.com/p/1918049072331362469)  
● [Transformer——FeedForward模块在干什么？](https://zhuanlan.zhihu.com/p/1918050616376301224)  
● [从0开始实现Transformer](https://zhuanlan.zhihu.com/p/1918357249883105145)  
● [什么是KV-Cache](https://zhuanlan.zhihu.com/p/1919338888536756837)  
● [Transformer注意力机制——MHA&MQA&GQA](https://zhuanlan.zhihu.com/p/1919500956946655189)  
● [FlashAttention怎么提升速度的？](https://zhuanlan.zhihu.com/p/1923328314241704991)  
● [FlashAttention2：更快的注意力机制，更好的并行效率](https://zhuanlan.zhihu.com/p/1923714840653993730)  
● [FlashAttention3 全解析：速度、精度、显存的再平衡](https://zhuanlan.zhihu.com/p/1924154277082961318)



<font style="color:rgb(25, 27, 31);">所有相关源码示例、流程图、模型配置与知识库构建技巧，我也将持续更新在Github：</font>[**<font style="color:rgb(25, 27, 31);">LLMHub</font>**](https://github.com/zhangting-hit/LLMHub)<font style="color:rgb(25, 27, 31);">，欢迎关注收藏！</font>

<font style="color:rgb(25, 27, 31);"></font>

随着大语言模型（LLM）如 ChatGPT、LLaMA 等广泛应用于文本生成、问答、代码自动补全等任务，人们对其推理效率提出了更高要求。尽管生成一个回答的成本可能仅为几分钱，但在拥有亿级用户、日均多次交互的背景下，算力消耗呈指数级上升。尤其是在如代码补全这类高频推理任务中，模型效率直接影响服务响应速度与云端部署成本。

为了解决这些问题，研究团队提出了一种面向解码场景的新型注意力加速方案 —— **Flash-Decoding**，该方法在保持准确率的前提下，显著提升了推理速度，特别适用于长上下文和小批量的推理任务。



希望大家带着下面的问题来学习，我会在文末给出答案。

1.  Flash-Decoding 为什么能在 batch size 很小时仍然充分利用 GPU？  
2.  Flash-Decoding 对短序列或大批量推理是否仍然有优势？  
3. <font style="color:rgb(25, 27, 31);"> FlashDecoding 和 FlashAttention 的主要区别是什么？</font>

##  解码中的注意力瓶颈
大语言模型的生成过程是逐 token 的迭代推理，每生成一个新 token，模型就需执行一次前向传播。虽然可以缓存历史的 Key 和 Value，避免重复计算，但每次推理仍必须执行一次完整的注意力机制，即：

```plain
softmax(Q @ K^T) @ V
```

这一步骤的计算量随着上下文长度（序列长度）线性增长，成为当前推理过程中的主要瓶颈。尤其是在使用长上下文（如 32k、64k token）或小批量推理（batch size = 1）时，GPU 资源难以充分利用，导致计算效率低下。

![](https://cdn.nlark.com/yuque/0/2025/gif/28454971/1751635583280-3a079447-3a31-4c95-a704-38f11326b2a4.gif)

## 现有方案的局限
在训练阶段，FlashAttention 通过高效并行优化了注意力计算，显著减少了内存访问成本。然而该方法在推理阶段并不适用。原因在于推理时每次仅有一个 Query token，如果批大小也较小（如在部署阶段常见的 batch=1），GPU 的计算单元将大量闲置。

而传统的 PyTorch 实现或 FasterTransformer 虽然能充分占用 GPU 资源，但中间数据频繁读写，效率依然不高。



## Flash-Decoding 的核心思想
![](https://cdn.nlark.com/yuque/0/2025/gif/28454971/1751635554527-04d9bb0c-fc42-481d-8131-c0034b6f224d.gif)

Flash-Decoding 的关键创新，在于**引入了对 Key/Value 长度维度的并行化处理**，在 Attention 计算中引入了“**分块处理 + log-sum-exp 融合**”的新流程，具体包括以下三步：

1. **分块处理 KV 缓存**：将 Key/Value 张量按长度划分为若干小块，不涉及 GPU 操作，仅为视图切片。
2. **并行计算注意力子输出**：每个 Query 分别与多个 KV 块并行计算注意力，生成局部输出，同时记录 log-sum-exp。
3. **归约融合最终输出**：利用 log-sum-exp 规则，将各个子块输出整合为最终的 Attention 输出。

由于 softmax 函数支持分布式迭代计算，因此这一过程既可高效并行，也无需存储多余中间数据，显著提升了 GPU 利用率。

##  性能验证与实测结果
研究人员在 **CodeLLaMA-34B** 模型上验证了 Flash-Decoding 的效果，测试场景为解码推理，比较方案包括：

+ 原生 PyTorch 注意力计算
+ FlashAttention v2（v2.2 之前）
+ FasterTransformer
+ Flash-Decoding（新方法）

![](https://cdn.nlark.com/yuque/0/2025/png/28454971/1751635411870-9f2438e8-3d68-4cf7-b092-57954188b139.png)

结果显示：

+ 在上下文较短（<1k tokens）时，各方案性能相近；
+ 在上下文极长（例如 32k 或 64k tokens）时，**Flash-Decoding 可实现最高 8 倍整体推理速度提升**；
+ Attention 子模块本身可比 FlashAttention 快 **30～50 倍**，并且几乎不受序列长度增加的影响。

这意味着 Flash-Decoding 成功解决了当前 LLM 推理中最关键的计算瓶颈。

## FlashAttention VS FlashDecoding
我们设定一个背景，假设有一群小朋友（代表 GPU 里的计算单元），他们的任务是帮你把一堆糖果（代表 Key 和 Value）从储物柜里拿出来，然后根据你的喜好（Query）计算出该给你哪些糖果（Attention 结果）。

### 5.1 FlashAttention：训练时的方案
**适用于糖果不多、来拿糖果的小朋友很多的场景。**

在训练阶段，大家一起训练模型，这时有很多 Query（即输入 token 的序列长度长，或者 batch size 很大）。所以每个小朋友都分配到了任务，**并行处理很多 Query**。

FlashAttention 的做法是：

+ 让小朋友按块从柜子里拿出糖果（Key/Value），
+ 每人处理一部分任务（一个 Query 块 + 一个 Batch），
+ 所有小朋友分头行动，再集中汇报。

优点：适合任务多、人多的情况（即训练）。  
缺点：**如果只有你一个人要糖果**（推理时只有一个 Query），大部分小朋友都闲着！



### 5.2 Flash-Decoding：推理时的新方案
**适用于只有一个人来拿糖果，但柜子特别大、糖果很多的场景。**

在推理阶段，比如你用 ChatGPT 生成一个回答时：

+ 你是唯一的 Query（一个 token）
+ 但你的上下文（之前说过的话）特别长（比如 64k token）

Flash-Decoding 的做法是：

+ 把储物柜 **按区域分块**（将 Key/Value 分成很多小块）
+ 每个小朋友负责柜子的一块，**并行从不同区域取糖果**、做局部判断
+ 最后把这些小块的结果 **合并起来，得出最终推荐结果**。

优点：即使只有你一个 Query，仍然能让所有小朋友都上阵工作  
对长上下文特别友好，不受 token 数量影响  
对短上下文（柜子小）可能就没啥优势了

## 总结
Flash-Decoding 为长上下文、小批量推理中的注意力计算带来了实质性的效率飞跃。它在不增加内存占用的前提下，通过结构级优化充分利用 GPU 资源，是未来 LLM 推理部署不可忽视的关键技术。



<font style="color:rgb(25, 27, 31);">最后，我们回答一下文章开头提出的问题。</font>

1. <font style="color:rgb(25, 27, 31);">Flash-Decoding 为什么能在 batch size 很小时仍然充分利用 GPU？</font><font style="color:rgb(25, 27, 31);">  
</font><font style="color:rgb(25, 27, 31);">因为 Flash-Decoding 引入了一个新的并行维度 —— </font>**Key/Value 序列长度的分块**<font style="color:rgb(25, 27, 31);">。  
</font><font style="color:rgb(25, 27, 31);">传统的 FlashAttention 在推理时只有一个 Query（当前要生成的 token），这导致 GPU 上的很多计算单元闲置。而 Flash-Decoding 将 KV 缓存切成多个小块，让 GPU 中的多个线程同时对这些分块并行计算注意力得分，最后再通过一次轻量的归约操作（log-sum-exp）整合结果。  
</font><font style="color:rgb(25, 27, 31);">这种方式即使在 batch size = 1 的情况下，也能做到 </font>**“人少但活多”**<font style="color:rgb(25, 27, 31);">，从而把 GPU 的计算能力榨干。</font>

---

2. <font style="color:rgb(25, 27, 31);">Flash-Decoding 对短序列或大批量推理是否仍然有优势？</font>

<font style="color:rgb(25, 27, 31);">在短上下文（如 <1k tokens）或大 batch 推理场景下，</font>**Flash-Decoding 并不一定比 FlashAttention 更快**<font style="color:rgb(25, 27, 31);">。这是因为 Flash-Decoding 的优势来源于将长 KV 序列切分并并行处理，只有当上下文够长（例如 16k、32k、64k）时，这种分块才带来可观的并行性。而在 batch size 很大时，FlashAttention 已经可以利用 Query/Batched 并行本身实现高效运行，此时 Flash-Decoding 的额外分块与归约过程可能带来一定开销。因此在实际应用中，很多框架（如 xFormers）会根据上下文长度、batch 大小等因素</font>**自动切换使用 FlashAttention 或 Flash-Decoding**<font style="color:rgb(25, 27, 31);">，从而在不同场景下获得最优性能。</font>

---

3.  FlashDecoding 和 FlashAttention 的主要区别是什么？

在于并行方式不同：**FlashAttention** 适用于训练阶段，它通过在 Query 长度和 Batch 大小上并行化，在多 token、多样本的场景中高效利用 GPU；而 **FlashDecoding** 专为推理优化，特别是在只有一个 Query（如生成一个 token）但上下文很长时，它通过在 Key/Value 长度维度上分块并行计算注意力，使得即使在 batch size = 1 的情况下也能充分占用 GPU 资源，显著提升长序列推理效率。  

<font style="color:rgb(25, 27, 31);"></font>

<font style="color:rgb(25, 27, 31);">关于深度学习和大模型相关的知识和前沿技术更新，</font>请关注公众号`<font style="color:rgb(51, 51, 51);background-color:rgb(243, 244, 244);">算法coting</font>`<font style="color:rgb(51, 51, 51);">！</font>

<font style="color:rgb(25, 27, 31);"></font>

以上内容部分参考了

[https://crfm.stanford.edu/2023/10/12/flashdecoding.html](https://crfm.stanford.edu/2023/10/12/flashdecoding.html)

非常感谢，如有侵权请联系删除！





