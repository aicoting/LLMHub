研究人工智能的人应该都知道Transformer，Transformer可以说是人工智能的基石，无论是自然语言处理经典模型Bert，还是现在火的一塌糊涂的GPT，他们的核心正是Transformer，说实话，虽然研究方向是深度学习，但是我对Transformer一直没有透彻的理解，所以这次痛定思痛，决定一定要把Transformer搞明白。但是网上大部分的材料要么草草讲一下Transformer架构，要么讲的过于理论，所以我想把我自己学习的、理解的记录下来，和大家一起学习，正好作为我的一个新的系列。

该系列从以下几个方面由浅入深的深入剖析一下Transformer：

● [一览Transformer整体架构](https://zhuanlan.zhihu.com/p/1918047303597552480)  
● [Transformer——Attention怎么实现集中注意力](https://zhuanlan.zhihu.com/p/1918049072331362469)  
● [Transformer——FeedForward模块在干什么？](https://zhuanlan.zhihu.com/p/1918050616376301224)  
● [从0开始实现Transformer](https://zhuanlan.zhihu.com/p/1918357249883105145)

<font style="color:rgb(25, 27, 31);">所有相关源码示例、流程图、模型配置与知识库构建技巧，我也将持续更新在Github：</font>[**<font style="color:rgb(25, 27, 31);">LLMHub</font>**](https://github.com/zhangting-hit/LLMHub)<font style="color:rgb(25, 27, 31);">，欢迎关注收藏！</font>



本文章是该系列的第二篇文章，在介绍Attention Block之前，我先介绍一下点积在衡量向量间的对齐度的作用和softmax。

从计算上看，点积是将所有的对齐分量相乘并累加；从几何上看，当向量指向相似方向时，点积为正，如果向量垂直，点积为0，当向量方向相反时则为负数。

![](https://cdn.nlark.com/yuque/0/2025/png/28454971/1749976695274-e1011672-ffb5-4af2-8f9d-cb41979dc198.png)

假设我们用embedding(cats)-embedding(cat)，我们主观的会认为得到了可以表示复数的向量，为了测试这一点，把这个结果向量和各种单复数名词进行点积，可以看到和复数的点积结果要比和单数的点积结果要高，并且在和one,two,three,four进行点积得到的结果是递增的。

![](https://cdn.nlark.com/yuque/0/2025/png/28454971/1749976907574-e0f439fc-35ad-4aa9-8734-2715b571ad6a.png)

<font style="color:rgb(0, 0, 0);">transformer block最终得到一个数字序列作为概率分布，代表所有可能的下一个单词的分布，所有值都应该在 0 到 1 之间，并且加起来都应该为 1。然而，在深度学习中，我们所做的其实是矩阵向量积，得到的输出并不满足上面的条件。</font>

![](https://cdn.nlark.com/yuque/0/2025/png/28454971/1749977766305-a2e0cadc-3c09-421d-8f12-c7008057b0d0.png)

**<font style="color:rgb(0, 0, 0);">Softmax</font>**<font style="color:rgb(0, 0, 0);"> </font><font style="color:rgb(0, 0, 0);">将任意数字列表转换为有效分布，这样最大值最终最接近 1，较小的值最终更接近 0。</font>

<font style="color:rgb(0, 0, 0);">它的工作原理是首先将输出向量中的每个数字进行e地幂次方，这会将所有的值变为正值，然后把所有这些新项的总和除以这个总和，这样就会把它归一化成一个总和为 1 的列表。</font>

![](https://cdn.nlark.com/yuque/0/2025/png/28454971/1749977893454-050224f5-3e7d-4e16-8eb5-5378f3a84489.png)

<font style="color:rgb(0, 0, 0);">之所以称它为 </font>_<font style="color:rgb(0, 0, 0);">softmax</font>_<font style="color:rgb(0, 0, 0);">，是因为它不是简单地提取出最大值，而是生成一个分布，该分布为所有相对较大的值提供权重，与它们的大小相称，相对于max更加的soft。</font>

<font style="color:rgb(0, 0, 0);"></font>

<font style="color:rgb(0, 0, 0);">下面我们就开始注意力的学习。</font>

<font style="color:rgb(0, 0, 0);">Transformer的论文名称叫做</font>_**<font style="color:rgb(0, 0, 0);">Attention</font>**__<font style="color:rgb(0, 0, 0);"> is All You Need,</font>_<font style="color:rgb(0, 0, 0);"> 所以可以看出注意力机制起到了至关重要的作用。</font>

<font style="color:rgb(0, 0, 0);">我们看一下下面三个短句：</font>

+ <font style="color:rgb(0, 0, 0);">American shrew</font><font style="color:rgb(0, 0, 0);"> </font>**<font style="color:rgb(0, 0, 0);">mole</font>**<font style="color:rgb(0, 0, 0);">.</font>
+ <font style="color:rgb(0, 0, 0);">One</font><font style="color:rgb(0, 0, 0);"> </font>**<font style="color:rgb(0, 0, 0);">mole</font>**<font style="color:rgb(0, 0, 0);"> </font><font style="color:rgb(0, 0, 0);">of carbon dioxide.</font>
+ <font style="color:rgb(0, 0, 0);">Take a biopsy of the </font>**<font style="color:rgb(0, 0, 0);">mole</font>**<font style="color:rgb(0, 0, 0);">.</font>

<font style="color:rgb(0, 0, 0);">其中的mole意思都不相同，他们的含义取决于上下文，分别代表</font>**<font style="color:rgb(0, 0, 0);">鼹鼠、摩尔和痣</font>**<font style="color:rgb(0, 0, 0);">，但是在transformer的第一步将每个token变成embedding后他们的向量是相同的，只有在transformer的下一步中，也就是attention中，周围的embedding才会将信息进行传递并更新mole的embedding的值。</font>

<font style="color:rgb(0, 0, 0);"></font>

<font style="color:rgb(0, 0, 0);">举一个例子，考虑单词 </font>**<font style="color:rgb(0, 0, 0);">tower</font>**<font style="color:rgb(0, 0, 0);"> 的embedding。表示的大概率是空间中一些非常通用的、非特定的方向，与许多又高又大的名词相关联。</font>

<font style="color:rgb(0, 0, 0);">如果tower 前面还有Eiffel，那么向量应该指向一个更具体地表示埃菲尔铁塔的方向，可能与巴黎和法国相关的向量以及铁制的东西相关。</font>

<font style="color:rgb(0, 0, 0);">如果它前面还有单词 </font>**<font style="color:rgb(0, 0, 0);">miniature</font>**<font style="color:rgb(0, 0, 0);">，那么向量应该进一步更新，使它不再与大而高的事物相关。</font>

![](https://cdn.nlark.com/yuque/0/2025/png/28454971/1749979476805-efd8fc11-383e-4bb1-afd9-c5d8e16f6071.png)

<font style="color:rgb(0, 0, 0);"></font>

<font style="color:rgb(0, 0, 0);">我们先描述一个注意力头，之后再介绍注意力块是如何由并行运行的不同的头组成的。</font>

<font style="color:rgb(0, 0, 0);">同样我们还是举一个例子，我们的输入是</font>

_<font style="color:rgb(0, 0, 0);">A fluffy blue creature roamed the verdant forest.</font>_

<font style="color:rgb(0, 0, 0);">我们目前想让根据形容词调整相应的名词的embedding。</font>

![](https://cdn.nlark.com/yuque/0/2025/png/28454971/1749980158003-0d97c36b-60ee-49a7-ad64-480b75e15630.png)

每个单词的初始embedding是一些高维向量，包含单词的编码和位置编码。我们用E表示这些向量。

![](https://cdn.nlark.com/yuque/0/2025/png/28454971/1749980416739-6984797e-0b66-4dca-90cb-b275837f4e5c.png)

如上图所示，我们的目标是经过计算产生一组新的embedding，这些新的embedding从前面的形容词中提取了相应的含义。

对于注意力块的第一步，我们可以想象每个名词都在问一个问题：“我前面有什么形容词吗？”，这些问题被编码为另一个向量，称为Query，这个Query是每个token的embedding和一个矩阵W<sub>Q </sub>相乘得到的，查询向量记作Q。其中的W<sub>Q </sub>是模型的参数，是在数据中学习的。

![](https://cdn.nlark.com/yuque/0/2025/png/28454971/1749980728014-607de564-255a-41ba-92f8-0afcfa3956a6.png)



同时，每个embedding乘以矩阵W<sub>k</sub>,得到我们成为Key的向量，这些Key是那些Query的潜在答案。

![](https://cdn.nlark.com/yuque/0/2025/png/28454971/1749981048622-092d974e-1324-40bb-b7cb-48f96b9d8e92.png)

我们希望Query和Key对齐时进行匹配，<font style="color:rgb(0, 0, 0);">我们衡量给定的一对 key 和 query 向量的对齐程度的方法就是我们在文章开头介绍的点积。较大的点积对应于更强的对齐。</font>

![](https://cdn.nlark.com/yuque/0/2025/png/28454971/1749981234301-34a56f1f-cb8c-4b46-9f2a-f58c2e97988c.png)

在我们举的例子中，_<font style="color:rgb(0, 0, 0);">fluffy</font>_<font style="color:rgb(0, 0, 0);"> 和 </font>_<font style="color:rgb(0, 0, 0);">blue</font>_<font style="color:rgb(0, 0, 0);"> 生成的key与 </font>_<font style="color:rgb(0, 0, 0);">creature</font>_<font style="color:rgb(0, 0, 0);"> 生成的query点积较大，所以他们的相关性更高。相比之下，其他单词（如 </font>_<font style="color:rgb(0, 0, 0);">the</font>_<font style="color:rgb(0, 0, 0);">）的key与单词 </font>_<font style="color:rgb(0, 0, 0);">creature</font>_<font style="color:rgb(0, 0, 0);"> 的query之间的点积将是一些小值或负值，这反映出这些单词彼此无关。</font>

<font style="color:rgb(0, 0, 0);">在计算所有的key和value的点积之后，我们得到一个矩阵，其中的数值表示每个单词和其他的单词的相关性的大小。</font>

![](https://cdn.nlark.com/yuque/0/2025/png/28454971/1749981709687-90dfbd66-121c-4077-a492-23224efe251f.png)

接下来我们就要用到文章开头介绍的softmax函数了，<font style="color:rgb(0, 0, 0);">将 softmax 应用于所有列，得到最终的相关性矩阵。</font>

![](https://cdn.nlark.com/yuque/0/2025/png/28454971/1749981792499-f0a58387-0118-4cd2-b4fe-4b7be3f3aca3.png)

transformer论文中用了一个公式来表示注意力机制，<font style="color:rgb(0, 0, 0);">在这里，变量Q和K分别表示查询向量和键向量的完整数组，即通过将embedding乘以W</font><sub>Q</sub><font style="color:rgb(0, 0, 0);">和W</font><sub>k</sub><font style="color:rgb(0, 0, 0);">获得的较小向量。</font>

![](https://cdn.nlark.com/yuque/0/2025/png/28454971/1749981826692-4cb20e60-5b5a-4588-8c2b-54a267c1b2c2.png)

<font style="color:rgb(0, 0, 0);">在等式的右侧，softmax 函数中的分子K</font><sup><font style="color:rgb(0, 0, 0);">T</font></sup>_<font style="color:rgb(0, 0, 0);">Q</font>_<font style="color:rgb(0, 0, 0);">是所有key-value对之间的点积。其中除以√</font>_<font style="color:rgb(0, 0, 0);">d</font>_<sub>_<font style="color:rgb(0, 0, 0);">k</font>_</sub><font style="color:rgb(0, 0, 0);">是为了数值稳定性。</font>

<font style="color:rgb(0, 0, 0);"></font>

<font style="color:rgb(0, 0, 0);">对于注意力机制，我们希望在进行计算时后面的单词不会影响前面的单词，因为在进行单词预测时前面的单词是看不到后面的单词的，为了防止这种情况，强制相关性矩阵下半部分为负无穷大。这样在应用 softmax 之后，所有这些点都变为零。这个过程成为masking。</font>

![](https://cdn.nlark.com/yuque/0/2025/png/28454971/1750037113975-4fb16832-9de0-491d-8b1f-d30fbf1c251b.png)

至此我们就得到了描述哪些词与更新哪些其他词相关的注意力矩阵，下一步就要进行更新了。

这时候就引入了除了Query和Key之外的第三个向量，Value向量，同样我们用到了一个矩阵W<sub>V</sub>,我们将这个矩阵乘以前面单词的embedding，得到的就是Value向量，也就是我们更新第二个单词要用到的向量，在这个例子中，我们就要把fluffy编码到creature中。

![](https://cdn.nlark.com/yuque/0/2025/png/28454971/1750037514664-45bd4cda-1799-4879-b82b-0f3b4556f7b6.png)

<font style="color:rgb(0, 0, 0);">W</font><sub><font style="color:rgb(0, 0, 0);">V</font></sub><font style="color:rgb(0, 0, 0);">乘以每一个embedding，产生一个Value向量集合，我们将每个Value向量乘以我们前面得到的目标列的相关性权重，累加得到我们想要进行的更改ΔE，添加ΔE到原始embedding中，得到一个优化的向量E′编码了上下文更丰富的含义。</font>

<font style="color:rgb(0, 0, 0);"></font>

![](https://cdn.nlark.com/yuque/0/2025/png/28454971/1750037795772-1d591fc2-bd25-44a1-905a-6f9d6ed8d9ca.png)

这个操作应用于每一列生成优化的embedding。至此，我们的注意力模块中进行的操作就到此结束，可以看到注意力机制核心是三个不同的矩阵参数化，W<sub>Q</sub>,W<sub>K</sub>,W<sub>V</sub>

![](https://cdn.nlark.com/yuque/0/2025/png/28454971/1750038052194-94f484f6-c241-4f68-aa40-60f0bd5e2262.png)

在实际应用中使用的是多头自注意力机制，多头自注意力中每个head都有自己的Query,Key和Value，并且并行运行

![](https://cdn.nlark.com/yuque/0/2025/png/28454971/1750038854495-0434b3e9-5d5f-48af-9ddf-b7177fdf3258.png)

<font style="color:rgb(0, 0, 0);">注意力机制的总体思路是，通过并行运行许多不同的 head，使模型能够学习上下文改变含义的许多不同方式。</font>

<font style="color:rgb(0, 0, 0);"></font>

<font style="color:rgb(0, 0, 0);">以上就是Attention机制的所有学习内容啦，码字不易，点个关注吧~</font>

<font style="color:rgb(25, 27, 31);">关于深度学习和大模型相关的知识和前沿技术更新，</font>请关注公众号`<font style="color:rgb(51, 51, 51);background-color:rgb(243, 244, 244);">算法coting</font>`<font style="color:rgb(51, 51, 51);">！</font>

<font style="color:rgb(25, 27, 31);"></font>

以上内容部分参考了

[Visualizing Attention, a Transformer's Heart](https://www.3blue1brown.com/lessons/attention)

非常感谢，如有侵权请联系删除！

