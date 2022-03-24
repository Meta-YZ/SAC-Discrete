zDDPG

## OU

OU是一种序贯相关的随机过程，在DDPG中实现RL的动作探索。

OU的公式：

![image-20211215100227297](README.assets/image-20211215100227297.png)

核心代码在utils文件中：

![image-20211215100322530](README.assets/image-20211215100322530.png)

OU过程有两点优势：

- 增加训练的step，使训练的更快
- 探索效率高

### 参考OU

[知乎大佬讲解OU非常清晰](https://zhuanlan.zhihu.com/p/54670989)

