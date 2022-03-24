#  SAC-Discrete

离散的SAC相比连续的有以下5处改变：

- **Q network 结构：由输入S，A输出Q-value→输入S，输出所有A的Q-value**

- **policy 结构：由高斯分布→n个action的categorical distribution**

- **value loss function：其拟合由gaussian→MC采样估计**

- **policy loss function：由于policy不再是gaussian，无需利用重参数化trick，改为采用MC采样估计**

- **temperature loss function：同理改为MC采样**

  

### 1. Q网络结构：

```python
class Critic(nn.Module):
    """Critic网络（Q值）"""
    def __init__(self, state_size, action_size, hidden_size):
        super(Critic, self).__init__()
        self.net = nn.Sequential(nn.Linear(state_size, hidden_size), nn.ReLU(),
                                 nn.Linear(hidden_size, hidden_size), nn.ReLU(),
                                 nn.Linear(hidden_size, hidden_size), nn.Hardswish(),
                                 nn.Linear(hidden_size, action_size))

    def forward(self, state):
        return self.net(state)
```

### 2. Policy网络结构:

```python
class Actor(nn.Module):
    """Actor网络or动作策略网络"""

    def __init__(self, state_size, action_size, hidden_size, config):
        super(Actor, self).__init__()
        self.config = config
        self.net_state = nn.Sequential(nn.Linear(state_size, hidden_size), nn.ReLU(),
                                       nn.Linear(hidden_size, hidden_size), nn.ReLU(),
                                       nn.Softmax(hidden_size, action_size))

    def forward(self, state):
        policy = self.net_state(state)
        return policy

    def act(self, state):
        action_probs = self.forward(state)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        log_action_prob = torch.log(action_probs + self.config.epsilon)
        return action.detch().cpu(), action_probs, log_action_prob
```

### 3. 值函数估计：

```python
# 为当前Q值计算价值
min_q = torch.min(q_target1_next, q_target2_next)
q_targets = rewards + (1 - dones) * gamma * action_probs * (min_q - self.alpha * log_probs)
q_targets = q_targets.detach()
```

### 4. 策略网络函数：

```python
actor_loss = -action_probs * (torch.min(sample_q1, sample_q2) - self.alpha * sample_log_prob)
```

### 5 . 温度损失函数：

```python
-action_probs * self.alpha * (log_probs + self.target_entropy).detach().cpu().mean()
```

