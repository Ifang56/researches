# Long Short-Term Memory with Dynamic Skip Connections

## 一、Abstract

**1、背景：**

 				近年来，长短时记忆(LSTM)被成功地用于建模变长序列数据。然而，LSTM在捕获长期依赖项时仍然会遇到困难。（In recent years, long short-term memory (LSTM) has been successfully used to model sequential data of variable length.However, LSTM can still experience difficulty in capturing long-term dependencies. ）

**2、提出**

​					缓和 LSTM 上述问题，提出 **动态跳跃连接** （ a dynamic skip connection），它可以学习直接连接两个依赖的单词（ learn to directly connect two dependent words）

**3、做法**

​					由于训练数据中没有依赖信息，我们提出了一种新的  **基于强化学习**  的**依赖关系建模和连接依赖词**的方法。（ propose a novel reinforcement learning-based method to model the dependency relationship and connect dependent words.）

​				该模型**基于跳跃连接计算循环传递函数**( computes the recurrent transition functions based on the skip connections)

​				相对于总是顺序处理整个句子的rnn具有动态跳跃的优势。		

**3、模型优势**

​				相对于总是连续地处理句子的rnn，模型有 **动态跳跃的优势**（ provides a dynamic skipping advantage over RNNs that always tackle entire sentences sequentially）

**4、实验表明**

​				在三种自然语言处理任务上的实验结果表明，该方法能够取得比现有方法更好的性能。在数字预测实验中，该模型的准确率比LSTM高出近20%。（In the number prediction experiment, the proposed model outperformed LSTM with respect to accuracy by nearly 20%.）

##  二、Conclusion

​				提出一种基于强化学习的LSTM模型，该模型使用**动态跳跃连接**扩展现有的LSTM模型。( extends the existing LSTM model with dynamic skip connections)

​				该模型可以动态地从过去的几个状态中选择一组最优的隐藏状态和细胞状态。（The proposed model can dynamically choose one optimal set of hidden and cell states from the past few states）

​		**优点：**				

​		1、通过动态跳跃连接（dynamic skip connections）

​			该模型比固定跳跃的句子具有更强的建模能力，并能解决语言中**可变长度的依赖问题**（the model has a stronger ability to model sentences than those with fifixed skip, and can tackle the dependency problem with variable lengths in the language）

​		2、梯度反向传播路径较短（shorter gradient backpropagation path）

​			该模型可以缓解梯度消失带来的挑战（the model can alleviate the challenges of vanishing gradient）

