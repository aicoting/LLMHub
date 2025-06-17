研究人工智能的人应该都知道Transformer，Transformer可以说是人工智能的基石，无论是自然语言处理经典模型Bert，还是现在火的一塌糊涂的GPT，他们的核心正是Transformer，说实话，虽然研究方向是深度学习，但是我对Transformer一直没有透彻的理解，所以这次痛定思痛，决定一定要把Transformer搞明白。但是网上大部分的材料要么草草讲一下Transformer架构，要么讲的过于理论，所以我想把我自己学习的、理解的记录下来，和大家一起学习，正好作为我的一个新的系列。

该系列从以下几个方面由浅入深的深入剖析一下Transformer：

● [一览Transformer整体架构](https://zhuanlan.zhihu.com/p/1918047303597552480)  
● [Transformer——Attention怎么实现集中注意力](https://zhuanlan.zhihu.com/p/1918049072331362469)  
● [Transformer——FeedForward模块在干什么？](https://zhuanlan.zhihu.com/p/1918050616376301224)  
● [从0开始实现Transformer](https://zhuanlan.zhihu.com/p/1918357249883105145)

<font style="color:rgb(25, 27, 31);">所有相关源码示例、流程图、模型配置与知识库构建技巧，我也将持续更新在Github：</font>[**<font style="color:rgb(25, 27, 31);">LLMHub</font>**](https://github.com/zhangting-hit/LLMHub)<font style="color:rgb(25, 27, 31);">，欢迎关注收藏！</font>



本文章是该系列的第三篇文章，上一篇文章我们讲解了Attention注意力模块，Transformer中在注意力模块之后紧跟着的是一个FeedForward前馈神经网络，实际上是一个MLP，他起到了一个相同重要的作用，但是很多人容易忽略它的作用，下面我们就一起来学习一下。

如果向大模型中输入“<font style="color:rgb(0, 0, 0);">Michael Jordan play the sport of ___</font>”，模型能够正确的预测出basketball，则表明在模型的数千亿参数的某些地方存储了乔丹和篮球之间的关系。

与Attention相比，MLP的计算相对简单，主要是由两个矩阵乘法和一个激活函数组成。

![](https://cdn.nlark.com/yuque/0/2025/png/28454971/1750057801917-37da49e8-d728-4642-a511-83bb7d9018d4.png)

形象的说，如果我们的输入是Michael Jordan的embedding，我们希望MLP块的输出是一个basketball方向的向量，我们加入到Michael Jordan的embedding中。

下面我们依次看看MLP中做了什么操作，首先，输入的embedding乘以一个矩阵，我们可以按行来看参数矩阵，可以理解为每一行都是一个询问，最后得到的点积就是对询问的回答，点积越大表示越肯定。

![](https://cdn.nlark.com/yuque/0/2025/png/28454971/1750058363908-d0680bac-9463-44fa-9aee-8fa49e1bf49e.png)

当输入embedding为Michael Jordan时，不仅询问为Michael Jordan？时会得到正的点积，当询问为<font style="color:rgb(0, 0, 0);">Michael Phelps？或者 Alexis Jordan？，点积也会为正，所以这个时候就需要加一个偏置，来保证当询问为Michael Phelps？或者 Alexis Jordan？，点积为负。</font>![](https://cdn.nlark.com/yuque/0/2025/png/28454971/1750059117610-3310c5c5-ecd4-406b-91a1-b87cbc2e6655.png)

在保证我们所要确定的询问点积全部为正之后，我们之前进行的所有操作是完全线性的，但是语言是非线性的，点积为负的时候我们不需要他们，所以之后引入非线性函数ReLu（<font style="color:rgb(0, 0, 0);">Rectified Linear Unit</font>），这个函数<font style="color:rgb(0, 0, 0);">如果传入负值，则返回零，如果传入正值，则保持不变。保留了我们需要的信息。</font>

<font style="color:rgb(0, 0, 0);">之后进行的流程就和刚开始非常像，从ReLu输出的向量将乘以一个矩阵并加上偏置，将输出向量的维度恢复到初始embedding大小。</font>

![](https://cdn.nlark.com/yuque/0/2025/png/28454971/1750059725199-5bf66e22-5d09-4b63-8714-eaf7969ffb0e.png)

我们可以按列来看参数矩阵，假如第一列是模型学到的basketball方向，那么ReLu输出的向量中n0应该为一个正数，我们就会将basketball添加到初始embedding中，如果为0则不产生影响。

![](https://cdn.nlark.com/yuque/0/2025/png/28454971/1750059857964-64d337e8-00f6-46ac-af70-12beed54750e.png)

得到basketball方向后，和输入向量相加产生一个将所有内容编码在一起的向量。

![](https://cdn.nlark.com/yuque/0/2025/png/28454971/1750071839083-a08227de-3d74-4202-989e-446d73e8cebc.png)



总的来说，输入乘以的<font style="color:rgb(0, 0, 0);">第一个矩阵的行可以被认为是embedding空间中的方向，代表着给定向量与某个特定方向的对齐程度。第二个矩阵的列告诉我们如果方向对齐，结果中将添加什么</font>_<font style="color:rgb(0, 0, 0);">。</font>_

<font style="color:rgb(25, 27, 31);">关于深度学习和大模型相关的知识和前沿技术更新，</font>请关注公众号`<font style="color:rgb(51, 51, 51);background-color:rgb(243, 244, 244);">算法coting</font>`<font style="color:rgb(51, 51, 51);">！</font>

<font style="color:rgb(25, 27, 31);"></font>

以上内容部分参考了

[How might LLMs store facts](https://www.3blue1brown.com/lessons/mlp)

非常感谢，如有侵权请联系删除！

_<font style="color:rgb(0, 0, 0);"></font>_

