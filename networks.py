import torch
import torch.nn as nn
from torch.distributions import Normal


class Actor(nn.Module):
    """Actor网络or动作策略网络"""

    def __init__(self, state_size, action_size, hidden_size, config):
        super(Actor, self).__init__()
        self.config = config
        self.net_state = nn.Sequential(nn.Linear(state_size, hidden_size), nn.ReLU(),
                                       nn.Linear(hidden_size, hidden_size), nn.ReLU())

        self.net_mu = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU(),
                                    nn.Linear(hidden_size, action_size))

        self.net_sigma = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU(),
                                       nn.Linear(hidden_size, action_size))

    def forward(self, state):
        x = self.net_state(state)
        mu = self.net_mu(x)
        log_sigma = self.net_sigma(x)
        log_sigma = torch.clamp(log_sigma, self.config.min_log_sigma, self.config.max_log_sigma)
        return mu, log_sigma

    def act(self, state):
        mu, log_sigma = self.forward(state)
        sigma = torch.exp(log_sigma)
        dist = Normal(mu, sigma)
        # 重参数化
        action = dist.rsample()
        # 对动作进行了裁剪，使我们得到了评估的动作，然后计算策略的
        tanh_action = torch.tanh(action)

        # 动作本来是无界的高斯分布，但是为了裁剪限定了边界，所以要在原始的log计算公式后添加一个修正系数
        log_prob = dist.log_prob(action)
        log_prob = log_prob - torch.log(1 - tanh_action.pow(2) + self.config.epsilon).sum(1, keepdim=True)
        return tanh_action, log_prob


class Critic(nn.Module):
    """Critic网络（Q值）"""
    def __init__(self, state_size, action_size, hidden_size):
        super(Critic, self).__init__()
        self.net = nn.Sequential(nn.Linear(state_size + action_size, hidden_size), nn.ReLU(),
                                 nn.Linear(hidden_size, hidden_size), nn.ReLU(),
                                 nn.Linear(hidden_size, hidden_size), nn.Hardswish(),
                                 nn.Linear(hidden_size, 1))

    def forward(self, state, action):
        return self.net(torch.cat((state, action), dim=1))