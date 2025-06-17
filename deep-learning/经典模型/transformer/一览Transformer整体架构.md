研究人工智能的人应该都知道Transformer，Transformer可以说是人工智能的基石，无论是自然语言处理经典模型Bert，还是现在火的一塌糊涂的GPT，他们的核心正是Transformer，说实话，虽然研究方向是深度学习，但是我对Transformer一直没有透彻的理解，所以这次痛定思痛，决定一定要把Transformer搞明白。但是网上大部分的材料要么草草讲一下Transformer架构，要么讲的过于理论，所以我想把我自己学习的、理解的记录下来，和大家一起学习，正好作为我的一个新的系列。

该系列从以下几个方面由浅入深的深入剖析一下Transformer：

● [一览Transformer整体架构](https://zhuanlan.zhihu.com/p/1918047303597552480)  
● [Transformer——Attention怎么实现集中注意力](https://zhuanlan.zhihu.com/p/1918049072331362469)  
● [Transformer——FeedForward模块在干什么？](https://zhuanlan.zhihu.com/p/1918050616376301224)  
● [从0开始实现Transformer](https://zhuanlan.zhihu.com/p/1918357249883105145)

<font style="color:rgb(25, 27, 31);">所有相关源码示例、流程图、模型配置与知识库构建技巧，我也将持续更新在Github：</font>[**<font style="color:rgb(25, 27, 31);">LLMHub</font>**](https://github.com/zhangting-hit/LLMHub)<font style="color:rgb(25, 27, 31);">，欢迎关注收藏！</font>



本文章是该系列的第一篇文章，我们来看一下Transformer的整体架构，数据怎么流通以及各个模块起到了什么作用。

Transformer是2017年Google提出的模型架构，最开始论文中的任务是翻译任务，之后扩展到文本预测、图像、语音等各个研究领域。



![](https://cdn.nlark.com/yuque/0/2025/png/28454971/1749974907375-0e2e7a34-295e-467a-a99d-60e666b0acff.png)

## tokenize
首先，输入序列会被分解为一堆小片段，这些片段被称为token，在文本中，他们通常是单词、单词的小片段或者其他的组合，如果涉及图像或者声音，那么token可以是图像的小块或者声音的片段。

![](https://cdn.nlark.com/yuque/0/2025/png/28454971/1749975116655-c9f4cc65-4cbe-4f21-aeab-dac8d191fe0f.png)

每个token都会被编码成一个embbedding向量，对于embedding可以参考我之前的文章[RAG-embedding篇](https://zhuanlan.zhihu.com/p/1912910452339484544)，这些向量是在高维空间下的坐标，具有相似含义的单词往往会落在这个空间中靠近的向量上，这些步骤是在数据进入Transformer中之前的预处理步骤。

## Position Embedding
在我们得到embedding之后，每个向量只包含了每个token的含义，但是没有考虑到词向量的顺序，也就会丢失原来文本的顺序信息。为了保留原始输入的顺序关系，我们需要加入位置编码，Transformer原文中位置编码的计算方式如下：

![](https://cdn.nlark.com/yuque/0/2025/png/28454971/1750074631357-fe3e9897-4b33-4f44-8d6d-d5c906c55242.png)

其中pos为位置序号，dmodel为特征的维度，i表示特征的第i维。

## Encoder
Transformer有编码器和解码器两个部分，编码器的输入是固定一次性的，而解码器输入是自回归累加的。他们之间的区别是Decoder可以选择masking，并且Decoder中的注意力是交叉注意力机制，将Encoder的输入作为Key和Value，其余部分都相同。但是他们的核心都是Attention，一个叫注意力机制的东西。

Encoder的作用是处理输入的向量，目标是得到不仅包含原输入单词的embedding，并且希望将和原输入单词相关性高的单词的信息融合。

实现融合信息的核心就是注意力（Attention）模块，在这里他们相互通信根据上下文进行更新，例如，“我爱你”中的爱和“i love you”里面的love相关性很高，注意力模块负责弄清楚上下文中哪些单词的含义之间的相关性更高，以及这些单词的含义应该如何更新。Attention模块的具体讲解在[Transformer——Attention怎么实现集中注意力](https://zhuanlan.zhihu.com/p/1918049072331362469)

在注意力模块之后，这些向量通过多层感知机或者前馈网络，增强网络的非线性表达能力，这里向量彼此之间不进行通信，他们并行执行相同的操作，这也是Transformer能够很好的并行训练的原因。从计算上讲，两个块中所有运算都是矩阵乘法。前馈神经网络的具体讲解在[Transformer——FeedForward模块在干什么？](https://zhuanlan.zhihu.com/p/1918050616376301224)

## Decoder
Decoder中的模块和Encoder中的模块类似，都是注意力模块和前馈神经网络以及融合归一化层。不同的是，Decoder是**自回归**，多了mask机制和交叉注意力机制。

Decoder每一次预测得到一个词，预测下一个单词所需要的信息都需要编码到序列的最后一个向量中，这个向量再经过一次运算产生下一个可能出现的文本的概率分布。预测出来的这个词和Decoder原始输入合在一起再次输入到Decoder，这也就是所谓的自回归。重复这个过程不断生成直至完毕。



以上就是Transformer整体的结构。

<font style="color:rgb(25, 27, 31);">关于深度学习和大模型相关的知识和前沿技术更新，</font>请关注公众号`<font style="color:rgb(51, 51, 51);background-color:rgb(243, 244, 244);">算法coting</font>`<font style="color:rgb(51, 51, 51);">！</font>

<font style="color:rgb(51, 51, 51);"></font>

以上内容部分参考了

[Transformer 模型 | 菜鸟教程](https://www.runoob.com/pytorch/transformer-model.html)



[But what is a GPT? Visual intro to Transformers](https://www.3blue1brown.com/lessons/gpt)

[https://zhuanlan.zhihu.com/p/264468193](https://zhuanlan.zhihu.com/p/264468193)

非常感谢，如有侵权请联系删除！

