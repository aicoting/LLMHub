## BERT 模型
### **问题分析**
BERT，全称 **Bidirectional Encoder Representations from Transformers**，由 Google 2018 年提出，是一种基于 Transformer 编码器的**预训练语言模型**。

**主要特点有 3 点：**

1. **双向编码**：相比传统语言模型只从左到右，BERT 使用**Masked Language Model (MLM)**，可以同时利用左右上下文理解词语含义。
2. **预训练+微调**框架：BERT 在大规模语料（如 Wikipedia + BookCorpus）上预训练后，可以迁移到具体任务（如分类、问答）进行微调，效果优异。
3. **输入表示**：BERT 输入不仅有词向量，还有**Segment Embedding** 和 **Position Embedding**，可同时处理单句或句子对任务。

典型应用：文本分类、命名实体识别（NER）、阅读理解、句子匹配等。

### BERT训练任务
BERT 是通过自监督学习进行训练的，核心的训练任务有两个：**掩蔽语言模型 (MLM)** 和 **下一句预测 (NSP)**。

+ 在 MLM 中，随机掩蔽输入句子中的一些词，模型通过上下文预测这些被掩蔽的词。
+ 在 NSP 中，模型判断两个句子是否是连续的，从而学习句子级别的关系。  
BERT 的训练数据通过大规模语料库生成，并根据这些任务构建掩蔽和句子对，进行无监督的预训练。

### **面试回答**
> BERT（Bidirectional Encoder Representations from Transformers）是一种基于 Transformer 编码器的预训练语言模型，特点是**双向上下文建模**，可以更好理解句子语义。BERT 先用**大规模文本预训练**，再通过**下游任务微调**，广泛应用于文本分类、问答等任务。BERT 通过 Masked Language Model 和 Next Sentence Prediction 两个预训练目标，捕捉了词级别和句子级别的关系。续如 RoBERTa、ALBERT、DistilBERT 都在 BERT 基础上进一步优化模型规模、效率和性能。BERT 通过双向预训练和灵活微调，显著推动了 NLP 任务效果，属于预训练语言模型的里程碑工作。
>

### 
