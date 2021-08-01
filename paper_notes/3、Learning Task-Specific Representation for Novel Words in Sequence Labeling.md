# Learning Task-Specifific Representation for Novel Words in Sequence Labeling

## 一、Abstract

**1、背景：**

 				词表征是基于神经网络的序列标注系统的关键组成部分（Word representation is a key component in neural-network-based sequence labeling systems）。然而，在最终任务中训练的 **未见单词或稀有单词** 的表示通常在可欣赏的表现上 **很差**（representations of unseen or rare words trained on the end task are usually poor for appreciable performance）。这通常被称为词汇表外(OOV)问题。（ ）

**2、提出**

​			只使用任务的训练数据来解决序列标注中的OOV问题，提出了一种新的方法来预测OOV词的表示法(如字符序列)和上下文。（predict representations for OOV words from their surface-forms (e.g., character sequence) and contexts）

​			该方法是专门设计来避免错误传播问题的现有方法在同一范式（The method is specififically designed to avoid the error propagation problem suffered by existing approaches in the same paradigm. ）		

**3、实验表明**

​				为评估的有效性，我们对四个词性标注任务和四个命名实体识别任务进行了广泛的实证研究。

​				与现有方法相比，所提出的方法可以获得更好或更具竞争力的性能。（Experimental results show that the proposed method can achieve better or competitive performance on the OOV problem compared with existing state-of-the-art methods.）

##  二、Conclusion

​				只使用任务的训练数据来解决序列标注中的OOV问题，该方法被设计用来从OOV单词的表面形式和上下文生成它们的表示（It is designed to generate representations for OOV words from their surface-forms and contexts.），其设计是为了避免现有方法在同一范式下所遭受的误差传播问题（it is designed to avoid the error propagation problem suffered by existing methods in the same paradigm）。

​				实验表明，在词性标注(POS)和命名实体识别(NER)方面的大量实验研究表明，该方法在OOV问题上的性能优于现有方法

