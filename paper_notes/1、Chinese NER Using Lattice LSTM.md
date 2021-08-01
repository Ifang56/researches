# **Chinese NER Using Lattice LSTM**

## 一、Abstract

**1、提出：**

 				lattice-structured LSTM model  for Chinese NER，对输入字符序列以及所有匹配的潜在单词进行编码（encodes a sequence of input characters as well as all potential words that match a lexicon.）

**2、优点**

​	1）与  **character-based**  方法比较：

​				明确地利用单词和单词序列信息（ explicitly leverages word and word sequence information）

​	2）与  **word-based**  方法比较：

​				不受分割错误的影响（ not suffer from segmentation errors）

**3、Gated recurrent cells 在我们模型中**

​				从一个句子中选择最相关的字符和单词以获得更好NER结果（Gated recurrent cells allow our model to choose the most relevant characters and words from a sentence for better NER results.）

**4、实验表明**

​				不同的数据集上，与 word-based and character-based LSTM baselines 比较，该模型表现最好。

##  二、Conclusion

​				不同领域 **优于** 基于 词或字粒度 的 LSTM-CRF（finding that it gives consistently superior performance compared to word-based and character-based LSTM-CRF across different domains. ）

​				**lattice 方法 **完全独立于分词，因为在语境中，**从词库自由选择词进行NER消歧**，所以**更有效使用了词信息**（The lattice method is fully independent of word segmentation, yet more effective in using word information thanks to the freedom of choosing lexicon words in a context for NER disambiguation.）

## 三、Introduction

### 1、第一段：

​		NER一直都是当做序列标注任务解决，通过实体边界与分类标签联合进行预测（where entity boundary and category labels are jointly predicted. ）；

​		当前先进的 英语 NER 使用  LSTM-CRF models，将 字信息整合到词表征（with character information being integrated into word representations）

​		 (Lample et al., 2016;Ma and Hovy, 2016; Chiu and Nichols, 2016; Liu et al., 2018)



### 2、第二段

​		中文 NER 与 分词有关，特别地，命名实体边界也是词边界。

​		一种直观的方法是 **先进行分词，然后再进行词序列标注**（One intuitive way of performing Chinese NER is to perform word segmentation fifirst, before applying word sequence labeling）。

​		从 segmentation → NER ，可能会遇到错误传播的潜在问题（ can suffer the potential issue of error propagation），因为 命名实体（NEs） 的一个重要来源是 分词中的 oov，而不正确的分词导致 NER 错误	（since NEs are an important source of OOV in segmentation, and incorrectly segmented entity boundaries lead to NER errors.）

​		上述问题在开放域中可能很严重，因为跨域分词仍然是一个未解决的问题

​		 (Liu and Zhang, 2012; Jiang et al., 2013;Liu et al., 2014; Qiu and Zhang, 2015; Chen et al.,2017; Huang et al., 2017)

​		已经表明，对于中文 NER，基于字符的方法优于基于单词的方法

​		(He and Wang, 2008;Liu et al., 2010; Li et al., 2014).

### 3、第三段

​			character-based NER 的其中一个缺点：没有充分利用显式的单词和单词序列信息，这可能是有用的（is that explicit word and word sequence in formation is not fully exploited, which can be potentially useful）

​			本文为解决上述问题，通过使用 lattic LSTM 来表示句子中的词典词，将潜在词信息整合到基于字符的 LSTM-CRF 中（integrate latent word information into character based LSTM-CRF by representing lexicon words from the sentence using a lattice structure LSTM）

​	如图1：**通过将句子与自动获取的大型词典进行匹配来构造一个词字符格**	（ we construct a word character lattice by matching a sentence with a large automatically-obtained lexicon）

诸如“长江大桥”、“长江（长江）”和“大桥（桥）”等词序列可**用于消除上下文中潜在的相关命名实体的歧义**（ can be used to disambiguate potential relevant named entities in a context），例如人名 “江大桥（Daqiao Jiang）”。	

<img src="./pictures/image-20210721213027656.png" alt="image-20210721213027656" style="zoom:50%;" />

### 3、第四段

​		由于一个 lattic 中单词字符路径的数量是 **指数级** 的（ an exponential number of word character paths in a lattice），**利用一个 lattic LSTM 结构自动控制从句子开始到句子结束的信息流**。（ leverage a lattice LSTM structure for automatically controlling information flflow from the beginning of the sentence to the end），如图2：

<img src="./pictures/image-20210722095027303.png" alt="image-20210722095027303" style="zoom: 50%;" />

**门控单元被用来将不同路径的信息动态地传递给每个字符**。（gated cells are used to dynamically route information from different paths to each character）

通过对NER数据的训练，**lattic  LSTM 可以学习从上下文中自动找到更有用的词，以获得更好的NER性能**（ lattice LSTM can learn to fifind more useful words from context automatically for better NER performance）。与基于字符和基于单词的NER方法相比，**模型利用了明确的单词信息而不是字符序列标**记，并且不会出现切分错误。（ leveraging explicit word information over character sequence labeling without suffering from segmentation erro）

### 3、第五段

​		该模型**显著优于基于LSTM CRF的字符序列标注模型和单词序列标注模型**，在不同领域的多种中文NER数据集上取得了最好的结果。代码和数据发布在    https://github.com/jiesutd/LatticeLSTM.

## 四、Related Word

### 1、第一段

​			本文与现有的使用神经网络进行NER的方法是一致。

​			Hammerton (2003)   尝试使用单向 LSTM 解决该问题，这是 NER 的第一个神经模型之一

​			Collobert et al. (2011)     使用 CNN-CRF  ，获得最好的竞赛结果obtaining competitive results to the best statistical models

​			dos Santos et al. (2015)  使用 character CNN  增强  CNN-CRF

​			最近多使用  LSTM-CRF 

​			Huang et al. (2015)  使用手工制作的拼写特征     （uses hand-crafted spelling features）

​			 Ma and Hovy (2016)    Chiu and Nichols (2016)    使用字符CNN来表示拼写特征  （use a character LSTM instead）

​			**本文基于单词的 baseline 系统采用与此工作线类似的结构**（ Our baseline word-based system takes a similar structure to this line of work）

### 2、第二段

​			字符序列标注一直是中文NER的主流方法  (Chen et al.,2006b; Lu et al., 2016; Dong et al., 2016)

​			基于字符的方法   优于  基于统计学的词  (He and Wang, 2008; Liu et al.,2010; Li et al., 2014)

​			lattice LSTM 优于 word LSTM   、 character LSTM

### 3、第三段

​			如何更好地利用词的信息来处理中文NER得到了持续的研究关注（Gao等人，2005）

​			其中分割信息被用作NER的软特征 （Zhao and Kit, 2008; Peng and Dredze, 2015; He and Sun, 2017a)

​			joint segmentation and NER has been investigated using dual decomposition  (Xu et al., 2014)

​			多任务学习 (Peng and Dredze, 2016)

​			我们的工作是一致的，专注于神经表征学习。 虽然上述方法会受到分段训练数据和分段错误的影响，但我们的方法**不需要分词**。 由于不考虑多任务设置，该模型在概念上更简单。

### 4、第四段

​			外部来源的信息被用于 NER 

​			特别，词库信息(Collobert et al., 2011; Passos et al., 2014; Huang et al., 2015; Luo et al., 2015)

​			Rei (2017)  使用单词级别的语言建模目标来增强NER的训练，在大型原始文本上进行多任务学习

​			Peters et al. (2017)   预训练一个字符语言模型来增强单词表征

​			Yang等人(2017b)   利用跨领域和跨语言的知识通过多任务学习

​			**我们通过在大型自动分割文本上预训练词嵌入词典来利用外部数据，而语言建模等半监督技术与我们的 lattic LSTM 模型正交，也可用于我们的 LSTM 模型。**

### 5、第五段

​		Lattice structured RNNs 树状结构的RNN（Tai et al, 2015）对DAGs的自然扩展

​		用在 model motion dynamics (Sun et al., 2017)

​				dependencydiscourse DAGs (Peng et al., 2017)

​				speech tokenization lattice (Sperber et al., 2017) 

​				multi-granularity segmentation outputs (Su et al., 2017) for NMT encoders

​		本文的不同，**lattice LSTM 动机与结构都不同 **

​					由于是为以字符为中心的 lattic  LSTM  CRF   序列标签而设计的，它有   recurrent cells ，但没有单词的隐藏向量。就我们所知。我们是第一个设计了一个新的格子LSTM 表示，也是第一个使用 lattic LSTM来表示混合 字符和词库 的新型 lattic LSTM  表示，也是第一个将   word-character lattice  用于 segmentation-free的中文NER。

## 五、Model

​			使用  **LSTM-CRF **作为主要的网络框架     best English NER model    (Huang  et al., 2015; Ma and Hovy, 2016; Lample et al., 2016)

​			输入：		s = c1,c2,...,cm          cj表示第j个字符

​								s = w1, w2, .... , wn，其中wi表示句子中的第i个词

​								t(i, k) 表示句子中第 i 个单词中第 k 个字符的索引 j

​			例如 ：s = “南京市 长江大桥”    中， t（2，1）=4（长），t(1, 3) = 3 (市)

​			本文使用 BIOES 标签（Ratinov and Roth, 2009） 进行基于词和基于字符的NER标注

### 3.1**Character-Based Model**

**（1） 单向：     Char**

​					<img src="./pictures/image-20210723213024067.png" alt="image-20210723213024067" style="zoom:33%;" />**e代表的是embedding**，c代表字符

​			 **character-based LSTM-CRF model** ，如下图3（a）

<img src="./pictures/image-20210723094608058.png" alt="image-20210723094608058" style="zoom: 33%;" /> 

<img src="./pictures/image-20210723213149280.png" alt="image-20210723213149280" style="zoom:33%;" />

**（2）双向 ：char   +  反向的char   （拼接）**

**（3）单向： char   +  segmentation label embeddings (字符对应的标签embedding)  （拼接）**

​						字符对应的标签：	BMES

<img src="./pictures/image-20210723214935622.png" alt="image-20210723214935622" style="zoom:33%;" />

### 3.2**Word-Based Model**

<img src="/Users/funplus/PycharmProjects/git_file/researches/paper_notes/pictures/image-20210725113015986.png" alt="image-20210725113015986" style="zoom: 50%;" />

**（1）单向： word + char**（拼接）

<img src="/Users/funplus/PycharmProjects/git_file/researches/paper_notes/pictures/image-20210724183729814.png" alt="image-20210724183729814" style="zoom:33%;" />

**（2）双向：word + char LSTM**             len(i)  为词wi的字符数

**（3）单向：word + char LSTM'**

<img src="/Users/funplus/PycharmProjects/git_file/researches/paper_notes/pictures/image-20210724183814871.png" alt="image-20210724183814871" style="zoom:33%;" />

**（4）word +char CNN** 

​			对每个词的字符做cnn，再经过max pool 得到其表征

<img src="/Users/funplus/PycharmProjects/git_file/researches/paper_notes/pictures/image-20210725113229806.png" alt="image-20210725113229806" style="zoom:33%;" />

### 3.3 **Lattice Model**

​		**word-character lattice model **的整体架构如上图：figure2<img src="./pictures/image-20210722095027303.png" alt="image-20210722095027303" style="zoom: 50%;" />

​		上图**extension ** 的  character-based model，integrating word-based cells and additional gates fo
controlling information flow。

<img src="/Users/funplus/PycharmProjects/git_file/researches/paper_notes/pictures/image-20210726211800226.png" alt="image-20210726211800226" style="zoom: 50%;" />

​	如图 Figure3，<img src="/Users/funplus/PycharmProjects/git_file/researches/paper_notes/pictures/image-20210726213102266.png" alt="image-20210726213102266" style="zoom:33%;" />

​		1、模型输入的是 **字符序列**（ character sequence ）c1,...,cm ， 以及与**词库D中的单词相匹配的所有字符子序列**。（a character sequence *c*1*, c*2, . . . , cm, together with all character subsequences that match words in a lexicon D）

​		2、在**四、Related Word第4段**使用**自动分割的大型原始文本来建立D**  （we use automatically segmented large raw text for buinding D.）

​		**3、模型有四种类型的向量**：

​		   *input vectors*,      *output hidden vectors*,      *cell vectors*     and      *gate vectors*

<img src="/Users/funplus/PycharmProjects/git_file/researches/paper_notes/pictures/image-20210726214455413.png" alt="image-20210726214455413" style="zoom:33%;" />

​		（1） *input vectors*，与 character-based model 不同的是，计算c 考虑了词库子序列w。

<img src="/Users/funplus/PycharmProjects/git_file/researches/paper_notes/pictures/image-20210726220729839.png" alt="image-20210726220729839" style="zoom:33%;" />

​		（2）*cell vectors*，有更多的循环路径（ more recurrent paths）到字符c。

<img src="/Users/funplus/PycharmProjects/git_file/researches/paper_notes/pictures/image-20210726221337851.png" alt="image-20210726221337851" style="zoom:33%;" />

​		（3）*gate vectors*，---------》nomarlised 到和为1。

<img src="/Users/funplus/PycharmProjects/git_file/researches/paper_notes/pictures/image-20210726221422220.png" alt="image-20210726221422220" style="zoom:33%;" />

​		（4）*output hidden vectors*，通过等式11计算，在 NER 训练期间，损失值反向传播到参数 <img src="/Users/funplus/PycharmProjects/git_file/researches/paper_notes/pictures/image-20210726221816322.png" alt="image-20210726221816322" style="zoom: 50%;" />，允许模型在 NER  labelling期间**动态关注更相关的词**。

### 3.4 **Lattice Model**

​		在 h1, h2, . . . , hτ 之上使用标准的 CRF 层，其中 τ 对于 character-based 和 lattice-based 的模型是 n，对于 word-based 的模型是 m。

​		使用**一阶维特比算法**在基于单词或基于字符的输入序列上找到得分最高的标签序列。 给定一组手动标记的训练数据 {(si, yi)}|Ni=1，使用 L2 正则化的句子级对数似然损失来训练模型：

<img src="/Users/funplus/PycharmProjects/git_file/researches/paper_notes/pictures/image-20210726223548518.png" alt="image-20210726223548518" style="zoom: 33%;" />

## 六、**Experiments**

​				进行大量实验，研究跨不同领域的 word-character lattice LSTM 的有效性。此外，目标是在不同的设置下，对基于词和基于字符的神经中文NER进行经验性比较（word-based and character-based neural Chinese NER under different settings），精确率（P）、召回率（R）和F1分数（F1）作为评价指标。

###  **1、Experimental Settings**

**（1）数据集**：（四个）

<img src="/Users/funplus/PycharmProjects/git_file/researches/paper_notes/pictures/image-20210726224935854.png" alt="image-20210726224935854" style="zoom:50%;" />

​	**OntoNotes** 和**MSRA** 数据集在新闻领域，**微博 NER** 数据集来自社交媒体网站新浪微博，以及自己标注的**中文简历**（resume）

