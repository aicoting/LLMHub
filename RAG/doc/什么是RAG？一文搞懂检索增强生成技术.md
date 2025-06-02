# 什么是RAG？一文搞懂检索增强生成技术
在大语言模型（LLM）如ChatGPT、Claude、Gemini日益强大的今天，人们希望它们不仅能“生成”，还要“准确生成”。然而，LLM训练的数据往往是静态的、封闭的，这使得它们在面对**时效性强、专业性高、上下文复杂**的问题时，力不从心。

在有些时候，企业内部或者事业部门内部的数据是不允许公开上传的，那么也就没有办法享受到大模型的服务，生产力也得不到解放。

这时，RAG（Retrieval-Augmented Generation，检索增强生成）应运而生。它是连接“生成能力”与“外部知识”的桥梁，让LLM不再是“闭门造车”，而成为真正的知识型智能体。

## 一、RAG的基本原理
RAG是一种通过**“先检索、后生成”**的方式，是一个提升语言模型生成准确性的技术框架。其核心流程如下：

1. **Query输入**：用户提出一个问题或任务。比如我问“明天的天气怎么样”，大语言模型大概率不会知道明天的天气，因为训练数据时间范围是今天前。
2. **Retriever检索器**：从外部知识库（文档、数据库、网页等）中检索与问题相关的内容。我问“明天的天气怎么样”之后，假设知识库里面刚好就有明天的天气信息，那么就会经过检索得到对应的语料信息“明天气温**50度**（千万不要出门）”。
3. **Generator生成器**：将检索到的内容连同问题一起输入大语言模型，让它生成更加精准、上下文丰富的回答。还是拿上面的我问“明天的天气怎么样”，**Generator生成器**得到**Retriever检索器**检索到的**“明天气温50度（千万不要出门）”**和我问的问题**“明天的天气怎么样”**一起输入到LLM中，得到回答”**明天气温50度，达到历史新高，请您注意一定不要出门，不然容易晒伤**“。

简单来说，RAG把**“我说我知道的”**变成**“我先当自己不知道”**->**"看看我的背包里有什么知识"**->**"哎找到了"**->**"总结一下再说"**。

## 二、为什么需要RAG？
大模型有知识盲点、时间滞后，原因在于：

+ 训练数据是静态的，无法获取实时信息；
+ 在专业领域（如医疗、金融、法律）中，模型缺乏最新的、结构化的知识；
+ 模型生成易产生“幻觉”（hallucination），即编造事实。

RAG通过引入检索机制，可以实时接入外部信息，同时精准聚焦专业文档，显著降低模型幻觉率。

## 三、RAG的技术架构
![](https://cdn.nlark.com/yuque/0/2025/png/28454971/1748700260677-c6e5149b-7cac-4d11-a8af-3c65da8fb68f.png)

RAG的系统主要分为两个核心模块：**Retriever + Generator**，可进一步细化为以下几部分：

### 1. 数据预处理与切分
+ 文档按结构或语义切分成段（chunk），如按标题、段落、Token窗口等，不同的文档处理方法不同，比如有图片，pdf，word，txt等等，后续会专门出一篇文章详细介绍以下不同文档的处理方法；
+ 使用向量化技术（如BERT、GTE等）构建**向量索引库，**向量数据库是深度学习领域专门使用的数据库，具有极快的查询速度，其中能够查询向量之间的相似度的特性能够很好的为RAG服务。

### 2. 检索阶段（Retrieval）
+ **稀疏检索**：如 BM25，依赖关键词匹配；
+ **密集检索**：如DPR、ColBERT，基于语义相似度；
+ **混合检索（Hybrid）**：结合两者，提升覆盖率与精度。

### 3. 生成阶段（Generation）
+ 使用LLM（如ChatGPT、LLaMA、Mistral）输入“问题+检索结果”，生成高质量回答。



RAG是一项将“语言生成”与“知识检索”紧密结合的关键技术，正快速从实验室走向产业。无论是NLP工程师、产品经理还是AI应用开发者，理解并掌握RAG，都将为我们开启智能系统的新可能。

---

接下来我将深入拆解RAG系统全景图谱，带你了解数据处理、检索增强、生成优化、评估反馈、架构部署到行业应用的全链路流程！



![](https://cdn.nlark.com/yuque/0/2025/png/28454971/1748700475889-f296ea99-6b48-4cd5-997b-97f14c83d2b0.png)

文中图片来自

[一图了解RAG的基本流程 - 小红书](https://www.xiaohongshu.com/explore/67af37c3000000002900f907?app_platform=android&ignoreEngage=true&app_version=8.84.2&share_from_user_hidden=true&xsec_source=app_share&type=normal&xsec_token=CBXL6AwaRxPwsBEfHd-b8XkIXnLp5yeI7-4atEWq64LaM=&author_share=1&xhsshare=WeixinSession&shareRedId=N0s4MUQ5NT82NzUyOTgwNjY0OTdGNUxN&apptime=1748700028&share_id=cd5eaa214b4e4c4b99cd2cecadfc33e1&share_channel=wechat)

[RAG 方案体系介绍 - 小红书](https://www.xiaohongshu.com/explore/683728b1000000002102c116?app_platform=android&ignoreEngage=true&app_version=8.84.2&share_from_user_hidden=true&xsec_source=app_share&type=normal&xsec_token=CB3GY25A_q4w1GcrSqWTlS6elqol_3FVR_pJLNPnLW5NQ=&author_share=1&xhsshare=WeixinSession&shareRedId=N0s4MUQ5NT82NzUyOTgwNjY0OTdGNUxN&apptime=1748684044&share_id=0df411b85dd544cfb595808cd4988f52&share_channel=wechat)

非常感谢，如有侵权请联系删除！

