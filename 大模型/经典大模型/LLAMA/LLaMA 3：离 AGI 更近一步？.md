LLaMA系列文章：

[一文读懂LLaMA](https://zhuanlan.zhihu.com/p/1930326295067199098)

[LLaMA2-大模型开源了！](https://zhuanlan.zhihu.com/p/1930329872208760843)



<font style="color:rgb(25, 27, 31);">继 LLaMA 1 的“小而强”、LLaMA 2 的“对齐进化”之后，LLaMA 3 带着更大规模的数据、更强的推理能力以及完全开放的商用许可横空出世。Meta 声称：</font>**LLaMA 3-70B 在多个任务中已逼近甚至超越 GPT-3.5，并将在未来挑战 GPT-4 的王座。**

![](https://cdn.nlark.com/yuque/0/2025/png/28454971/1753106810538-861f5b3b-a44e-4374-af98-063739a04053.png)

本文将带你全面了解 LLaMA 3 的技术细节、性能表现和应用前景，看它如何成为“开源阵营的最强音”。**  **

<font style="color:rgb(25, 27, 31);">所有相关源码示例、流程图、模型配置与知识库构建技巧，我也将持续更新在Github：</font>[**<font style="color:rgb(25, 27, 31);">LLMHub</font>**](https://github.com/zhangting-hit/LLMHub)<font style="color:rgb(25, 27, 31);">，欢迎关注收藏！</font>

---

<font style="color:rgb(25, 27, 31);">可以带着下面三个问题阅读本文：</font>

1. <font style="color:rgb(25, 27, 31);">LLaMA 3 相较于 LLaMA 2 有哪些实质性突破？</font>
2. <font style="color:rgb(25, 27, 31);">它是如何在不开源训练集的情况下做到性能领先的？</font>
3. <font style="color:rgb(25, 27, 31);">相比 GPT-4、Claude 3，LLaMA 3 的开放策略有哪些优势？</font>

---

## <font style="color:rgb(25, 27, 31);">一、LLaMA 3 是什么？</font>
<font style="color:rgb(25, 27, 31);">LLaMA 3 是 Meta 于 2024 年 4 月发布的新一代基础大语言模型系列。</font>

![](https://cdn.nlark.com/yuque/0/2025/png/28454971/1753003849031-12ef0b82-aa59-4441-bf68-a336eba72060.png)

<font style="color:rgb(25, 27, 31);">首次推出了两个主力版本：</font>

+ **LLaMA 3-8B**
+ **LLaMA 3-70B**

<font style="color:rgb(25, 27, 31);">这些模型均为 </font>**全开源、商用免费**<font style="color:rgb(25, 27, 31);">，支持基础任务和对话任务（LLaMA 3-Instruct），是 Meta 在开源大模型道路上的又一次大步前行。</font>

Llama 3 旨在打造媲美现有闭源模型的最强开源大语言模型，同时吸收开发者反馈，提升模型的整体可用性与安全性。我们秉持“早发布、多发布”的开源理念，让社区在模型开发期间即可使用。首批发布的是文本模型，后续将推出多语言、多模态、更长上下文窗口与更强推理能力版本。  

---

## <font style="color:rgb(25, 27, 31);">二、技术亮点：真正的第三代基础模型</font>
LLaMA3的模型结构仍然是基于transformer的自回归预测。

![](https://cdn.nlark.com/yuque/0/2025/png/28454971/1753003884557-8f3acda4-d67a-47af-955c-26cc9cbe5f62.png)

### <font style="color:rgb(25, 27, 31);">2.1训练数据全面升级（虽然未开源）</font>
+ <font style="color:rgb(25, 27, 31);">总量高达 </font>**15T tokens**<font style="color:rgb(25, 27, 31);">，是 LLaMA 2 的 7.5 倍；</font>
+ <font style="color:rgb(25, 27, 31);">覆盖 30 多种语言，更具全球适应性；</font>
+ <font style="color:rgb(25, 27, 31);">加入 </font>**代码、数学、长文本文档、学术论文**<font style="color:rgb(25, 27, 31);"> 等多种复杂语料；</font>
+ <font style="color:rgb(25, 27, 31);">数据源仍未公开，但明确不包含用户私有数据，使用了过滤与质量评分机制。</font>

### 2.2<font style="color:rgb(25, 27, 31);"> 架构创新</font>
<font style="color:rgb(25, 27, 31);">虽然 LLaMA 3 沿用了 Transformer 架构，但进行了大量工程改进：</font>

+ **上下文长度默认 8K，未来支持最多 128K**<font style="color:rgb(25, 27, 31);">；</font>
+ <font style="color:rgb(25, 27, 31);">精细设计了 </font>**tokenizer（tiktoken 兼容）**<font style="color:rgb(25, 27, 31);">，压缩率更高；</font>
+ <font style="color:rgb(25, 27, 31);">使用了新的</font>**数据混合策略（data mixture strategy）**<font style="color:rgb(25, 27, 31);">，提升多任务泛化能力；</font>
+ <font style="color:rgb(25, 27, 31);">全面支持 FP16 / BF16 / INT8 推理，适配主流硬件部署。</font>

### 2.3 训练创新
+ **训练过程采用数据、模型、流水线三重并行**，在定制 24K GPU 集群上运行，最大 GPU 利用率超过 400 TFLOPS。

![](https://cdn.nlark.com/yuque/0/2025/png/28454971/1753005131768-59a27245-1b11-4300-9cbb-20260a3b2a6a.png)

<font style="color:rgba(0, 0, 0, 0.85);">GPU按照[TP（tensor parallelism）、CP（context parallelism）、PP（pipeline parallelism）、DP（data parallelism）]的顺序被划分为并行组。在此示例中，16个GPU被配置为组大小为|TP| =2，|CP| =2，|PP| =2和|DP| =2的值。GPU在4D并行性中的位置被表示为向量[D1，D2，D3，D4]，其中Di是第i个并行性维度上的索引。在该示例中，GPU0[TP0，CP0，PP0，DP0]和GPU1[TP1，CP0，PP0，DP0]在相同的TP组中，GPU0和GPU2在相同的CP组中，GPU0和GPU4在相同的PP组中，并且GPU0和GPU8在相同的DP组中。</font>

+ 新训练堆栈支持**自动错误检测与修复、存储优化、数据回滚等功能**，Llama 3 训练效率比 Llama 2 提高约 3 倍，GPU 利用率达 **95%+**。

---

## <font style="color:rgb(25, 27, 31);">三、对话模型 LLaMA 3-Instruct 的对齐策略</font>
![](https://cdn.nlark.com/yuque/0/2025/png/28454971/1753004022143-840d129c-4a36-481f-bbc2-ae8fd8d5db03.png)

<font style="color:rgb(25, 27, 31);">不同于 LLaMA 2，LLaMA 3-Instruct 结合了：</font>

+ **监督微调（SFT）**
+ **拒绝采样**
+ **PPO（近端策略优化）**
+ **DPO（直接偏好优化）  **
+ <font style="color:rgb(25, 27, 31);">安全性测试、拒答机制和红队评估，并采用了 Meta 自研的 </font>**自我验证机制（Reflexion）**

<font style="color:rgba(0, 0, 0, 0.85);">LLaMa 3执行多步规划、推理和工具调用以解决任务的步骤如下图：</font>

![](https://cdn.nlark.com/yuque/0/2025/png/28454971/1753004322629-b92cd685-4d54-4ce8-af9f-95697e29d533.png)

<font style="color:rgb(25, 27, 31);">这些机制共同确保了 </font>**内容连贯性、回答风格友好、安全性增强**<font style="color:rgb(25, 27, 31);">，并可在开源中自由部署到 RAG、Agent、文档问答等系统中。</font>

---

## <font style="color:rgb(25, 27, 31);">四、优点</font>
<font style="color:rgb(25, 27, 31);">LLaMA 3 除了性能提升，更重视开放性：</font>

+ <font style="color:rgb(25, 27, 31);">完全免费开源，采用 Apache 2.0 协议；</font>
+ <font style="color:rgb(25, 27, 31);">可商用、可微调、可用于私有部署（无授权障碍）；</font>
+ <font style="color:rgb(25, 27, 31);">支持 HuggingFace、Torch、Transformers、vLLM 等主流平台和框架。</font>

<font style="color:rgb(25, 27, 31);">同时，Meta 与 AWS、Azure、Google Cloud、NVIDIA、Snowflake 等达成深度适配，</font>**从研发到生产一条龙**<font style="color:rgb(25, 27, 31);">。</font>



并且LLaMa3之后就支持多模态输入：

![](https://cdn.nlark.com/yuque/0/2025/png/28454971/1753004526235-20d5dede-bb78-4754-a613-cf1e0b07cde6.png)

+ <font style="color:rgb(25, 27, 31);">更强指令跟随模型</font>
+ <font style="color:rgb(25, 27, 31);">多模态输入（图文理解、PDF、音频）</font>
+ <font style="color:rgb(25, 27, 31);">更长上下文支持（最高至 128K）</font>
+ <font style="color:rgb(25, 27, 31);">可结合 RAG、工具使用、插件等 Agent 架构</font>

---

最后我们回答一下文章开头提出的问题：

1. **<font style="color:rgb(25, 27, 31);">LLaMA 3 相较于 LLaMA 2 有哪些实质性突破？</font>**

LLaMA 3 相较于 LLaMA 2 的实质性突破主要在于更优化的模型架构设计、更高质量和多样化的训练数据、以及更先进的对齐技术，使得模型在理解复杂任务和生成准确文本方面能力显著提升。

2. **<font style="color:rgb(25, 27, 31);">它是如何在不开源训练集的情况下做到性能领先的？</font>**

LLaMA 3 在不开源训练集的情况下，通过充分整合公开和授权数据，结合高效的训练技术和强化学习与人类反馈（RLHF）策略，实现了性能上的领先。

3. **<font style="color:rgb(25, 27, 31);">相比 GPT-4、Claude 3，LLaMA 3 的开放策略有哪些优势？</font>**

相比 GPT-4 和 Claude 3，LLaMA 3 的开放策略优势体现在模型权重和技术细节更为开放，支持本地部署和定制，降低使用门槛，促进社区创新与生态发展。



<font style="color:rgb(25, 27, 31);">关于深度学习和大模型相关的知识和前沿技术更新，请关注公众号</font><font style="color:rgb(25, 27, 31);background-color:rgb(246, 246, 246);">算法coting</font><font style="color:rgb(25, 27, 31);">！</font>

## 参考内容
+ [LLaMA 3 官方博客（Meta AI）](https://ai.meta.com/blog/meta-llama-3/)
+ [HuggingFace 上的 LLaMA 3 模型页](https://huggingface.co/meta-llama)
+ [https://arxiv.org/abs/2407.21783](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2407.21783)

