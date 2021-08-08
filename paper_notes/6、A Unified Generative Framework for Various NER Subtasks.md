# **A Unified Generative Framework for Various NER Subtasks**

# 一、Abstract

**1、背景：**

 					NER是识别实体的跨度（spans ），NER分为  the flat NER, nested NER, and discontinuous NER子任务；

​					目前，主要通过token级别的序列标注或者span级别的分类来解决；

​					很难同时解决这三个子任务。

**2、提出**

​					提出**将 NER 子任务制定为实体跨度序列生成任务，可以通过统一的序列到序列 (Seq2Seq) 框架来解决**（we propose to formulate the NER subtasks as an entity span sequence generation task, which can be solved by a unified sequence-to-sequence (Seq2Seq) framework.）

**3、做法**

​					利用预训练的 Seq2Seq 模型来解决所有三种 NER 子任务，而无需特殊设计标记模式或枚举跨度的方法；（we can leverage the pre-trained Seq2Seq model to solve all three kinds of NER subtasks without the special design of the tagging schema or ways to enumerate spans. ）

​					利用三种类型的实体表示，将实体线性化为一个序列。（We exploit three types of entity representations to linearize entities into a sequence）

**3、模型优势**

​				容易实现

**4、实验表明**

​				8个英语NER数据集（2个 flat NER, 3个nested NER, 3个discontinuous NER），表现 SoTA or near SoTA

##  二、Conclusion

​				将  **NER子任务**  作为 **实体跨度序列生成** 问题--------》可以使用具有指针机制的统一 Seq2Seq 模型来处理扁平、嵌套和不连续的 NER 子任务。

​				结合预训练 Seq2Seq 模型 BART并提高性能。

​				为更好地利用 BART，测试了三种类型的实体表示方法，将实体跨度线性化为序列。

**长度更短、更类似于连续BPE序列的实体表示获得了更好的性能**

​		未来：

​			