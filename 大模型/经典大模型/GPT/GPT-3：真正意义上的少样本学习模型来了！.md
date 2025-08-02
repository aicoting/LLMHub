GPT系列文章

[GPT1：通用语言理解模型的开端](https://zhuanlan.zhihu.com/p/1927401841815160673)

[GPT-2：让语言模型一统多任务学习江湖](https://zhuanlan.zhihu.com/p/1927841343062910920)

<font style="color:rgb(25, 27, 31);">所有相关源码示例、流程图、模型配置与知识库构建技巧，我也将持续更新在Github：</font>[**<font style="color:rgb(25, 27, 31);">LLMHub</font>**](https://github.com/zhangting-hit/LLMHub)<font style="color:rgb(25, 27, 31);">，欢迎关注收藏！</font>

<font style="color:rgb(25, 27, 31);"></font>

2020 年，OpenAI 发布了轰动一时的 GPT-3 论文《Language Models are Few-Shot Learners》，这是继 GPT-2 之后更大规模、更强泛化能力的语言模型。GPT-3 不仅在参数规模上实现了飞跃，还首次展示了**few-shot、one-shot 和 zero-shot 学习能力的统一提升**，从而彻底改变了自然语言处理的研究与应用格局。

---

在阅读这篇文章前，建议你先思考以下三个问题：

+ GPT-3 为什么不再需要微调（fine-tuning）就能在下游任务上达到很高精度？它是怎么做到 few-shot 学习的？
+ GPT-3 在模型结构上和 GPT-2 有什么区别？是结构创新带来的性能提升吗？
+ GPT-3 的成功是否仅靠“参数量堆叠”？模型变大后会不会过拟合或失去泛化能力？

---

### 模型设计
GPT-3 沿用了 GPT-2 的 Transformer Decoder 架构，没有引入任何新的机制（比如 attention 结构或新的 loss 函数），甚至训练目标也没有变化。

但其参数规模直接从 15 亿拓展到了 **1750 亿**，成为当时最大规模的语言模型。

![](https://cdn.nlark.com/yuque/0/2025/png/28454971/1752411964223-59ee559c-28c9-4353-9f6b-4b2f9ddcd2e7.png)

> GPT-3 没有结构创新，靠的是**规模**（scaling laws）+ 大数据 + 高计算量。
>

---

### 训练数据
GPT-3 使用了多个公开数据源，包括：

+ Common Crawl（过滤后高质量网页）
+ WebText2（改进版）
+ Books（多个书籍语料）
+ Wikipedia（英文版）
+ 其他文献语料

总计训练语料高达 **570GB（token 约 4990 亿个）**，使得模型掌握了丰富语言模式、知识和任务结构。

---

### 训练目标不变，但推理方式革新
训练目标仍是标准的语言建模（Language Modeling）：

![image](https://cdn.nlark.com/yuque/__latex/fbbbc7cbe302864ca116b3d1b05781f7.svg)

其中x1,x2,...,xi−1是当前 token 前面所有的词，作为上下文输入。

![](https://cdn.nlark.com/yuque/0/2025/png/28454971/1752411692154-71ea5ffd-699e-4a8d-9063-1ed1ec6c4795.png)

但 GPT-3 创新地在推理阶段（inference）使用自然语言 prompt 来实现任务指定，从而完成：

+ **Zero-shot learning**：只给任务描述
+ **One-shot learning**：给任务描述 + 一个示例
+ **Few-shot learning**：给任务描述 + 多个示例

![](https://cdn.nlark.com/yuque/0/2025/png/28454971/1752411930605-b4dd4268-4414-4e38-98e0-a4e3100438dc.png)

例如对于翻译任务，输入可能是：

```plain
Translate English to French:
English: The cat is on the table.
French:
```

GPT-3 会在没有微调的情况下直接生成合适翻译，准确率极高。

---

### 评估任务：涵盖语言理解、问答、数学、推理等
GPT-3 在多个任务上测试了 zero/one/few-shot 表现，包括：

+ 阅读理解（RACE, BoolQ）
+ 常识问答（OpenBookQA, PIQA）
+ 文本完形填空（LAMBADA）
+ 数学题（GSM8k）
+ 翻译（WMT）
+ 甚至编程任务（Python 代码生成）

**结果显示**：模型越大，few-shot 能力越强，且在多数任务上显著优于微调模型。

---

###  关键发现：few-shot 性能随规模对数线性增长
论文核心发现之一是：

> GPT-3 的 few-shot 表现几乎呈 **对数线性增长**，说明“更大 = 更聪明”。
>

而且：

+ 不容易过拟合
+ 泛化能力随规模增强
+ 不需要任务特定的结构改造

这给了后续 GPT-4、InstructGPT、ChatGPT 以极大的信心。

---

<font style="color:rgb(25, 27, 31);">最后，我们回答一下文章开头提出的问题。</font>

+ <font style="color:rgb(25, 27, 31);">GPT-3 为什么不再需要微调（fine-tuning）就能在下游任务上达到很高精度？它是怎么做到 few-shot 学习的？</font>

<font style="color:rgb(25, 27, 31);">GPT-3 实现 few-shot 学习是因为它在训练时见到了丰富多样的任务语言模式，加上巨大的模型容量，使得它能够通过自然语言 prompt 自动“识别”任务目标，不再需要传统的微调流程。</font>

---

+ <font style="color:rgb(25, 27, 31);">GPT-3 在模型结构上和 GPT-2 有什么区别？是结构创新带来的性能提升吗？</font>

<font style="color:rgb(25, 27, 31);">GPT-3 的结构与 GPT-2 相同，仍是纯 Transformer Decoder。但其规模扩大至 1750 亿参数，使模型具备更强的表达能力。这说明性能提升主要来源于“规模扩展”而非结构创新。</font>

---

+ <font style="color:rgb(25, 27, 31);">GPT-3 的成功是否仅靠“参数量堆叠”？模型变大后会不会过拟合或失去泛化能力？</font>

虽然 GPT-3 是一个超大模型，但实验发现它并没有过拟合。相反，它的泛化能力更强，few-shot 能力更稳健，证明了“**扩大模型规模+训练数据+训练步数**”是一条通往通用人工智能的重要路径。



<font style="color:rgb(25, 27, 31);">关于深度学习和大模型相关的知识和前沿技术更新，请关注公众号</font><font style="color:rgb(25, 27, 31);background-color:rgb(246, 246, 246);">算法coting</font><font style="color:rgb(25, 27, 31);">！</font>

