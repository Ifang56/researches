# **FLAT: Chinese NER Using Flat-Lattice Transformer**

# 一、Abstract

**1、背景：**

 					最近，character-word lattice structure 在中文NER 有效，但其结构复杂、动态（complex and dynamic），难以充分利用GPU，推理速度较慢。

**2、提出**

​					提出 **FLAT: Flat-LAttice Transformer **对中文 NER。

**3、做法**

​					lattice 结构转换为**由跨度组成的平面结构（a flat structure consisting of spans**），每个跨度**对应一个字符或潜在词及其在原始点阵中的位置**

**3、模型优势**

​				借助 Transformer 的强大功能和精心设计的位置编码，**FLAT 可以充分利用点阵信息并具有出色的并行化能力**

**4、实验表明**

​				四个数据集上，**FLAT 优于其他基于词典的模型**

##  二、**Conclusion and Future Work**

​		**flflat-lattice Transformer** 核心是：**lattice 结构转换 跨度集合 和 特殊位置编码**（converting

lattice structure into a set of spans and introducing the specifific position encoding.）

​		未来：

​			**调整不同的lattice 和 graph** （We leave adjusting our model to different kinds of lattice or graph as our future work）

​				

