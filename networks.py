import torch
import torch.nn as nn
from torch.distributions import Normal


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