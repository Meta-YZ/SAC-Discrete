import random
from collections import namedtuple, deque

import numpy as np
import torch


class ReplayBuffer:
    """存储轨迹转移数组"""

    def __init__(self, action_size, buffer_size, batch_size):
        self.action_size = action_size
        self.buffer = deque(maxlen=buffer_size)  # 一个buffer里能存多少条经验轨迹
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, reward, next_state, done):
        """往buffer里添加新的经验"""
        e = self.experience(state, action, reward, next_state, done)
        self.buffer.append(e)

    def sample(self):
        """从buffer里随机采样一个批次的轨迹样本"""
        experiences = random.sample(self.buffer, k=self.batch_size)  # 随机抽取batch_size个样本

        # 将变量类型从np转为tensor，并从CPU挪到GPU中进行加速计算
        states = torch.as_tensor(np.vstack([e.state for e in experiences if e is not None]),
                                 dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(np.vstack([e.action for e in experiences if e is not None]),
                                  dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor(np.vstack([e.reward for e in experiences if e is not None]),
                                  dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(np.vstack([e.next_state for e in experiences if e is not None]),
                                      dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(np.vstack([e.done for e in experiences if e is not None]),
                                dtype=torch.float32, device=self.device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)
