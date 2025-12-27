前几天面试的时候，面试官问我知道什么是KV-Cache吗？我愣在了原地，所以回来赶紧搞懂，把我所理解的和大家一起学习一下。也作为Transformer系列的第五篇。

Transformer系列文章：

● [一览Transformer整体架构](https://zhuanlan.zhihu.com/p/1918047303597552480)  
● [Transformer——Attention怎么实现集中注意力](https://zhuanlan.zhihu.com/p/1918049072331362469)  
● [Transformer——FeedForward模块在干什么？](https://zhuanlan.zhihu.com/p/1918050616376301224)  
● [从0开始实现Transformer](https://zhuanlan.zhihu.com/p/1918357249883105145)

<font style="color:rgb(25, 27, 31);">所有相关源码示例、流程图、模型配置与知识库构建技巧，我也将持续更新在Github：</font>[**<font style="color:rgb(25, 27, 31);">LLMHub</font>**](https://github.com/zhangting-hit/LLMHub)<font style="color:rgb(25, 27, 31);">，欢迎关注收藏！</font>

<font style="color:rgb(25, 27, 31);"></font>

希望大家带着下面的问题来学习，我会在文末给出答案。

1. <font style="color:rgb(25, 27, 31);">KV Cache节省了</font>[<font style="color:rgb(9, 64, 142);">Self-Attention</font>](https://zhida.zhihu.com/search?content_id=228316421&content_type=Article&match_order=1&q=Self-Attention&zhida_source=entity)<font style="color:rgb(25, 27, 31);">层中哪部分的计算？</font>
2. <font style="color:rgb(25, 27, 31);">KV Cache对</font>[<font style="color:rgb(9, 64, 142);">MLP</font>](https://zhida.zhihu.com/search?content_id=228316421&content_type=Article&match_order=1&q=MLP&zhida_source=entity)<font style="color:rgb(25, 27, 31);">层的计算量有影响吗？</font>
3. <font style="color:rgb(25, 27, 31);">KV Cache对block间的数据传输量有影响吗？</font>

在推理阶段，KV-Cache是Transformer加速推理的常用策略。

我们都知道，Transformer的解码器是**自回归**的架构，所谓自回归，就是前面的输出会加到现在的输入里面进行不断生成，所以理解了Attention的同学就会意识到这个里面有很多重复性的计算，如果不了解Attention，可以去看一下我之前的文章[Transformer——Attention怎么实现集中注意力](https://zhuanlan.zhihu.com/p/1918049072331362469)。

那么为什么会有重复性计算呢，我们来看一下

![](https://cdn.nlark.com/yuque/0/2025/png/28454971/1749981826692-4cb20e60-5b5a-4588-8c2b-54a267c1b2c2.png?x-oss-process=image%2Fformat%2Cwebp)

![](https://cdn.nlark.com/yuque/0/2025/webp/28454971/1750383945014-62e8ee36-d58b-4656-8e4a-722852929208.webp)

可以看到当前的Attention计算只与几个数据有关：

1. 当前的query加入的新token，也就是来自模型前一轮的输出，图中的第一轮的“你”，第二轮的“是”，第三轮的“谁”。
2. 历史K矩阵：每个Q向量都会依次和K矩阵中的每一行进行计算
3. 历史V矩阵：Q*K得到的矩阵每一行要与V矩阵进行计算



传统Transformer在进行计算时是在每一轮中将Q,K,V乘以对应的W权重，进行计算Attention的过程，但其实这个计算过程中每一轮新增的向量只是**Q中最后一行向量，K中最后一列向量，V中最后一行向量**，可以把之前K,V计算的结果进行缓存，当前一轮只利用新加入的Q向量和新的K向量和V向量进行计算，最后将K向量和V向量与原始的向量进行拼接，来大大减少冗余的计算量。



当然KV-Cache会增加内存的使用，是典型的**空间换时间**操作，所以当序列特别长的时候，KV-Cache的显存开销甚至会超过模型本身，很容易爆显存，<font style="color:rgb(25, 27, 31);">比如batch_size=32, head=32, layer=32, dim_size=4096, seq_length=2048, float32类型，则需要占用的显存为 2 * 32 * 4096 * 2048 * 32 * 4 / 1024/1024/1024 /1024 = 64G。</font>

<font style="color:rgb(25, 27, 31);"></font>

<font style="color:rgb(25, 27, 31);">最后，我们回答一下文章开头提出的问题。</font>

1. <font style="color:rgb(25, 27, 31);">KV Cache节省了</font>[<font style="color:rgb(9, 64, 142);">Self-Attention</font>](https://zhida.zhihu.com/search?content_id=228316421&content_type=Article&match_order=1&q=Self-Attention&zhida_source=entity)<font style="color:rgb(25, 27, 31);">层中哪部分的计算？</font>

<font style="color:rgb(25, 27, 31);">  节省的是历史 token 的 Key 和 Value 的重新计算 ，把 </font>**历史 token 的 Key/Value**<font style="color:rgb(25, 27, 31);"> 缓存在缓存中。每次只需计算当前 token 的 Q，历史的 K/V 可直接复用。</font>**无需重新前向计算 K,V 的线性变换和位置编码**<font style="color:rgb(25, 27, 31);">，从而节省了大量计算。</font>

2. <font style="color:rgb(25, 27, 31);">KV Cache对</font>[<font style="color:rgb(9, 64, 142);">MLP</font>](https://zhida.zhihu.com/search?content_id=228316421&content_type=Article&match_order=1&q=MLP&zhida_source=entity)<font style="color:rgb(25, 27, 31);">层的计算量有影响吗？</font>

没有影响，MLP 层（即 FFN）是每个 token 独立计算的，不依赖历史上下文。所以每个生成的 token 无论如何都要进行一次完整的 MLP 前向传播，KV Cache 只作用于 Self-Attention 层的 Key 和 Value，不涉及 MLP 层。

3. <font style="color:rgb(25, 27, 31);">KV Cache对block间的数据传输量有影响吗？</font>

**有影响，通常会减少 block 间传输量（尤其在多卡/分布式环境中）**。<font style="color:rgb(25, 27, 31);">如果每一步都重新计算历史 Key/Value，就要不断在 block 之间传输所有 token 的 KV 表征。使用 KV Cache 后，每步只需传输当前 token 的 Q（给当前层使用）以及缓存的 KV（已经存储，不重复传）</font>

<font style="color:rgb(25, 27, 31);"></font>

<font style="color:rgb(25, 27, 31);">关于深度学习和大模型相关的知识和前沿技术更新，</font>请关注公众号`<font style="color:rgb(51, 51, 51);background-color:rgb(243, 244, 244);">算法coting</font>`<font style="color:rgb(51, 51, 51);">！</font>

<font style="color:rgb(51, 51, 51);"></font>

上内容部分参考了

[动图看懂什么是KV Cache](https://zhuanlan.zhihu.com/p/19489285169)

[LLM(20)：漫谈 KV Cache 优化方法，深度理解 StreamingLLM](https://zhuanlan.zhihu.com/p/659770503)

[大模型推理加速：看图学KV Cache](https://zhuanlan.zhihu.com/p/662498827)

非常感谢，如有侵权请联系删除！



