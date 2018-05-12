# Playing Atari with Deep Reinforcement Learning

- > Requirements(submit a reading report)
  >
  > 1. What does the paper solve?
  > 2. Methods and details.
  > 3. Advantages and disadvantages.
  > 4. Compare with similar papers.



## 1.What does the paper solve?

### problem:

直接从高维的输入（比如视觉或听觉）来学习一个控制策略 是 RL增强学习的长期挑战。

### previous solutions:

- 人工提取特征（比如物体的位置）
- 使用线性的value function或者policy策略来表征

而且，性能的好坏主要取决于Feature engineering的好坏即特征提取的好坏

### DL的发展带来了转机，但也有很大困难：

深度学习已经在视觉，语音等领域取得突破性进展，根本的方法就是通过神经网络自动提取复杂特征。所以，很自然的我们会考虑一个问题： 

#### RL能否也借助DL进一步发展 ?           

#### Of course yes!But there are some difficulties.

- DL的成功依赖于大量labelled的sample，从而进行Supervised learning.但RL只有一个reward返回值，并且通常还带有噪声，延迟并且是sparse(稀少)的，也就是说不可能每个state给一个reward.常常是在Delay几千毫秒后才返回

- DL的sample都是independent(独立)的，但RL中却常是相关的，前后的state(状态)是有影响的

- DL的target(目标)的分布是固定的。比如一张图片上面是我老婆远坂凛，那她就是我老婆，是不会变的。但RL的分布却是一直变化的，就好比你玩游戏超级玛丽，前后的场景是不同的，可能前面训练好了后面用不了，也可能后面训练好了前面又用不了。

  ​

这抛出了几个问题，也正是这篇paper解决的问题：

1. 没有标签怎么办？
2. 样本相关性太高怎么办？
3. 目标分布不固定怎么办？

## 2.Solution:CNN + Q-Learning = Deep Q Network

1. 通过Q-Learning使用reward来构造标签
2. 通过experience replay的方法来解决相关性及非静态分布问题

#### 第一个将深度学习模型与增强学习结合在一起从而成功地直接从高维的输入学习控制策略：

主要是将卷积神经网络和Q Learning结合在一起。卷积神经网络的输入是原始图像数据（作为状态）输出则为每个动作对应的价值Value Function来估计未来的反馈Reward

### 实验环境：

使用Arcade Learning Environment 来训练Atari 2600 游戏。 

- 目标：使用一个基于神经网络的agent来学习玩各种游戏，玩的越多越好。 

- 输入：要求只输入图像数据和得分，和人类基本一样 

- 输出： 控制动作 

- 要求：对于不同游戏，网络的architecture(结构)及顶层parameter(参数)设定一样

  ​

#### DRL目标：

目前DL的方式核心在于采用大量的数据集，然后使用SGD进行权值的更新。因此，这里的目标就是将增强学习的算法连接到深度神经网络中，然后能直接输入RGB的原始图像，并使用SGD进行处理。

这里必须提到第一个将神经网络用于RL的工作：

### TD-gammon

TD-gammon使用了MLP(Multi-layer perceptron)也就是一般的神经网络，一个隐藏层（hidden layer）来训练。并且将其应用到了玩backgammon游戏上取得了人类水平。但相当可惜的是，当时人们把算法用到其他游戏象棋围棋并不成功，导致人们认为TD-gammon算法只适用于backgammon这个特殊的例子，不具备通用性。

本质上，使用神经网络是为了模拟一个非线性的函数（value或者policy都行，比如flappy bird，设定它上升到一个高度下降这就是一个分段函数）。人们发现，将model-free的算法比如Q-learning与非线性函数拟合的方法（神经网络是一种）很容易导致Q-network发散。因此，大部分的工作就使用线性的函数拟合（linear function approximation），收敛性好。

这里本文的改进就体现出来了：

相比于TD-gammon的在线学习方式，Deep mind使用了**experience replay**的技巧。简单的说就是建立一个经验池，把每次的经验都存起来，要训练的时候就 **随机** 的拿出一个样本来训练。这样就可以解决状态state相关的问题。以此同时，动作的选择采用常规的ϵϵ-greedy policy。 就是小概率选择随机动作，大概率选择最优动作。

并且输入的历史数据不可能是随机长度，这里就采用固定长度的历史数据，比如deep mind使用的4帧图像作为一个状态输入。

整个算法就叫做**Deep-Q-Learning**。



##### 说实话算法我并没有看懂，基础知识差的较多，只是尽力在理解其中的名词，算法目前无法进行分析

#### 文中提到的对比standard online Q-learning的优点：

- 每一步的经验都能带来很多权值的更新，拥有更高的数据效率

- 就是experience replay的优势，打破数据的相关性，降低数据更新的不确定性variance。

- experience replay的另一个优点就是不容易陷入局部最优解或者更糟糕的不收敛。 如果是on-policy learning，也就是来一个新的经验就学一个。那么下一个动作就会受当前的影响，如果最大的动作是向左，那么就会一直向左。使用experience replay 获取的行为的分布就比较平均，就能防止大的波动和发散。也因此，这是一个off-policy的学习。

  ​

  实际应用中，只存储n个经验在经验池里（毕竟空间有限嘛）这个方法的局限性就是这个经验池并没有区分重要的转移transition，总是覆盖最新的transition。 

  所以，采用有优先级的使用memory是一个更好的方式。

##### 再之后是具体的步骤了：

#### 与处理和网络模型架构：

因为输入是RGB，像素也高，因此，对图像进行初步的图像处理，变成灰度矩形84*84的图像作为输入，有利于卷积。 
接下来就是模型的构建问题，毕竟Q(s,a)包含s和a。一种方法就是输入s和a，输出q值，这样并不方便，每个a都需要forward一遍网络。

Deep mind的做法是神经网络只输入s，输出则是每个a对应的q。这种做法的优点就是只要输入s，forward前向传播一遍就可以获取所有a的q值，毕竟a的数量有限。

#### 实验：

- 测试7个游戏
- 统一不同游戏的reward，正的为1，负的为-1，其他为0。这样做a,R的好处是限制误差的比例并且可以使用统一的训练速度来训练不同的游戏
- 使用RMSProp算法，就是minibatch gradient descent方法中的一种。Divide the gradient by a running average of its recent magnitude. 梯度下降有很多种方法包括（SGD,Momenturn,NAG,Adagrad,Adadelta,Rmsprop) 相关问题以后再分析。
- ϵϵ-greedy 前1百万次从1 下降到0.1，然后保持不变。这样一开始的时候就更多的是随机搜索，之后慢慢使用最优的方法。
- 使用**frame-skipping technique**,意思就是每k frame才执行一次动作，而不是每帧都执行。在实际的研究中，如果每帧都输出一个动作，那么频率就太高，基本上会导致失败。在这里，中间跳过的帧使用的动作为之前最后的动作。这和人类的行为是一致的，人类的反应时间只有0.1，也是采用同样的做法。并且这样做可以提速明显的。那么这里Deepmind大部分是选择k=4，也就是每4帧输出一个动作。



#### 训练：

如何在训练的过程中估计训练的效果在RL上是个Challenge。毕竟不像监督学习，可以有training 和validation set。那么只能使用reward，或者说平均的reward来判定。也就是玩的好就是训练的好。

但是存在问题就是reward的噪声很大，因为很小的权值改变都将导致策略输出的巨大变化。

但平均Q值的变化却是稳定的，这是必然的，因为每次的Target计算都是使用Q的最大值。

关键的是所有的实验都收敛了！

**虽然没有理论支持为什么保证收敛**，但是就是实现了，Deep mind的方法可以在一个稳定的状态下使用大规模的深度神经网络结合增强学习。



#### Value Function:

在敌人出现时，Q值上升，快消灭敌人时，Q值到顶峰，敌人消失，Q值回到正常水平。这说明Q值确实代表了整个复杂的状态。实际上到后面发现，整个神经网络可以同时跟踪多个图上的目标，就是战机的那款游戏

## 3.Adventages and Disadvantages

#### Advantages:

- 算法有通用性，同样的网络可以学习不同的游戏（当然，游戏要求具有相似性）

- 采用End-to-End的训练方式，无需人工提取Feature（比如游戏中敌人的position等等）

- 通过不断的测试训练，可以实时生成无尽的样本用于有监督训练(Supervised Learning)

  ​

#### Disadvantages:

- 由于输入的状态是短时的，所以只适用于处理只需短时记忆的问题，无法处理需要长时间经验的问题。（比如玩塞尔达传说）
- 使用CNN（卷积神经网络）来训练不一定能够收敛，需要对网络的参数进行精良的设置才行





## 4.Conclusion

这篇paper采用了一个全新的方法结合深度学习和增强学习，可以说是deep reinforcement learning的开山之作。采用stochastic mini batch updates以及experience replay的技巧。 效果很强，具有通用性。  



#### 下一篇论文很想看

> Mastering the Game of Go with Deep Neural Networks and Tree Search



#### 但是这一篇应该是最简单的都是看的懵懵懂懂，有太多概念，目前只是较为机械的学习。。。得恶补基础才能吃透