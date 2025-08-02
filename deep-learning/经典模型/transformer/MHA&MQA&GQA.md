本篇文章是Transformer系列的第六篇。

Transformer系列文章：

● [一览Transformer整体架构](https://zhuanlan.zhihu.com/p/1918047303597552480)  
● [Transformer——Attention怎么实现集中注意力](https://zhuanlan.zhihu.com/p/1918049072331362469)  
● [Transformer——FeedForward模块在干什么？](https://zhuanlan.zhihu.com/p/1918050616376301224)  
● [从0开始实现Transformer](https://zhuanlan.zhihu.com/p/1918357249883105145)  
● [什么是KV-Cache](https://zhuanlan.zhihu.com/p/1919338888536756837)



<font style="color:rgb(25, 27, 31);">所有相关源码示例、流程图、模型配置与知识库构建技巧，我也将持续更新在Github：</font>[**<font style="color:rgb(25, 27, 31);">LLMHub</font>**](https://github.com/zhangting-hit/LLMHub)<font style="color:rgb(25, 27, 31);">，欢迎关注收藏！</font>

希望大家带着下面的问题来学习，我会在文末给出答案。

1. 为什么KV-Cache虽然可以减少重复计算，但在上下文长度变长时却会显存“爆炸”？
2. MQA 是怎么在牺牲部分表达能力的前提下减少显存占用的？     
3. GQA 相比于 MHA 和 MQA，有哪些优势？为什么说它是两者的折中方案？  



因为原始transformer中采用的是Multi-HeadAttention，要重复计算Key和Value的向量，导致消耗大量的计算资源，所以出现了前面我们介绍了transformer推理加速常用的方法**KV-Cache**，将相应的Key和Value进行缓存，大大减小了计算量。但是随着上下文长度的不断增大，KV-Cache需要的显存也就越来越大，最后直接爆炸。

所以，不同的研究就出现来去解决这个问题。



![](https://cdn.nlark.com/yuque/0/2025/png/28454971/1750408107853-f43f563d-3815-4ecb-9c8f-104a22e77add.png)

## MHA
MHA中有h个attention头，每个头的维度是d_head = d_model/h，对每个头，会用不同的线性变换矩阵将输入X映射为Query,Key和Value向量。

每个头的线性变换公式如下，对于第i个头

![](https://cdn.nlark.com/yuque/0/2025/png/28454971/1750421169811-aa7313d6-9216-4a30-bc99-56fd727f6c36.png)

其中

![](https://cdn.nlark.com/yuque/0/2025/png/28454971/1750421180742-5c46f578-9fb3-4c9f-a33c-596625747b97.png)

计算完每个头的注意力输出后，我们将它拼接成一个张量作为最后的输出：

![](https://cdn.nlark.com/yuque/0/2025/png/28454971/1750421231153-1728e682-23db-4b06-b240-b5c51b1fa77e.png)

![](https://cdn.nlark.com/yuque/0/2025/png/28454971/1750416882124-b2f92eba-f1d3-4f23-9e45-d87f24206001.png)

MHA中有n个head，每个head可以捕捉不同的特征，使模型更全面的理解输入数据，每个head都有一个对应的Key和Value的向量，所以n个head就会有n个Key和n个Value，对应也就会有n个KV-Cache。

## MQA
为了解决多头注意力（MHA）的显存占用过大问题，多查询注意力（MQA）通过让所有的head共享Key和Value向量，大大减少了显存的需求。

![](https://cdn.nlark.com/yuque/0/2025/png/28454971/1750421814479-70fd5657-2f58-4854-9e1b-a2e7dfad88d4.png)

具体的，我们将MHA中的所有Kh和Vh做平均：  
![](https://cdn.nlark.com/yuque/0/2025/png/28454971/1750418381353-341975f3-a86b-448a-9986-50a02e65d860.png)

其中H表示head的数量，Kh和Vh分别表示第h个头对应的Key和Value，在推理过程中每个头共享K*和V*，但是Q是不一样的，最后将所有头的输出拼接并映射回输出空间

![](https://cdn.nlark.com/yuque/0/2025/png/28454971/1750418570400-dbc1d151-7cd0-4a76-9864-8fcffd07b20b.png)

虽然显存的压力大大减小了，但是对所有head的不同查询，键和值都是相同的，模型的表达能力受到了一定的限制。

## GQA
GQA是MHA和MQA的折中方案，全称是分组查询注意力，通过将head分为若干组，每一组里面的head共享Key和Value，在推理速度和模型性能之间取得平衡。

![](https://cdn.nlark.com/yuque/0/2025/png/28454971/1750421804276-4e7935e9-4e81-4d80-9b05-9ca9ab41e4c7.png)

每组包含H/G个查询头，每组共享一个Key和Value，首先将输入通过线性变换投影为Q,K,V，将Q划分到h个头中后，再将这些头进一步划分为G组，同时将K和V划分为G组，每组共享一个K和V，对每组的Q和各组共享的K,V进行注意力计算，将各组的注意力结果拼接，最终投影得到输出。



建议大家结合实现的代码一起看一下：[LLMHub/code/transformer/attention.ipynb at main · zhangting-hit/LLMHub](https://github.com/zhangting-hit/LLMHub/blob/main/code/transformer/attention.ipynb)，相信大家一定会有更加深入的理解。



<font style="color:rgb(25, 27, 31);">最后，我们回答一下文章开头提出的问题。</font>

1. 为什么KV-Cache虽然可以减少重复计算，但在上下文长度变长时却会显存“爆炸”？<font style="color:rgb(25, 27, 31);">  
</font><font style="color:rgb(25, 27, 31);">	KV-Cache 的本质是将每一层的 Key 和 Value 保存下来，避免重复计算。但随着上下文变长（例如从几百个token到几万个token），每一步生成都要将当前的 Query 与所有历史的 Key 做注意力计算。因此，Key 和 Value 的缓存随着时间步增长线性增加，显存占用不断增大，最终可能超过GPU的内存限制。</font>
2. MQA 是怎么在牺牲部分表达能力的前提下减少显存占用的？<font style="color:rgb(25, 27, 31);">  
</font><font style="color:rgb(25, 27, 31);">	MQA（Multi-Query Attention）中，所有 attention head </font>**<font style="color:rgb(25, 27, 31);">共享同一个 Key 和 Value 向量</font>**<font style="color:rgb(25, 27, 31);">，只保留各自独立的 Query 向量。由于 Key 和 Value 不再为每个 head 分别维护，而是只保存一次，因此显存需求大幅下降。但这种共享也意味着不同 head 没法感知不同的上下文表示，导致表达能力下降。</font>
3. GQA 相比于 MHA 和 MQA，有哪些优势？为什么说它是两者的折中方案？<font style="color:rgb(25, 27, 31);">  
</font><font style="color:rgb(25, 27, 31);">	GQA（Grouped-Query Attention）将所有 head 分成若干组，每组共享 Key 和 Value，而每个 head 保留独立的 Query。</font>**<font style="color:rgb(25, 27, 31);">这种结构在保持部分表示多样性的同时减少了 Key 和 Value 的数量，因此在性能和效率之间取得平衡。</font>**<font style="color:rgb(25, 27, 31);">相较 MHA，它更节省显存；相较 MQA，它表达能力更强。</font>



<font style="color:rgb(25, 27, 31);">关于深度学习和大模型相关的知识和前沿技术更新，</font>请关注公众号`<font style="color:rgb(51, 51, 51);background-color:rgb(243, 244, 244);">算法coting</font>`<font style="color:rgb(51, 51, 51);">！</font>

<font style="color:rgb(51, 51, 51);"></font>

上内容部分参考了

[GQA:Training Generalized Multi-Query Transformer Models from  Multi-Head Checkpoints](https://arxiv.org/pdf/2305.13245)

[Transformer注意力机制：MHA、MQA与GQA的对比](https://syhya.github.io/zh/posts/2025-01-16-group-query-attention/)

非常感谢，如有侵权请联系删除！





