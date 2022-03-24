import torch
import torch.optim as optim
import torch.nn.functional as F
from networks import Actor, Critic
from torch.nn.utils import clip_grad_norm_
from utils import *


class Agent:
    """与环境交互并且学习好的策略"""
    def __init__(self, state_size, action_size, hidden_size, config):
        self.device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
        self.state_size = state_size
        self.action_size = action_size
        self.config = config

        # SAC特有的超参数
        self.target_entropy = -action_size.prod()  # 熵的初始值
        self.log_alpha = self.log_alpha.exp().detach([0.0], requires_grad=True)  # 熵的温度系数自动调优，所以需要梯度
        self.alpha = self.log_alpha.exp().detach()

        # Actor网络，两个网络，取偏小的
        self.actor = Actor(state_size, action_size, hidden_size, config).to(self.device)
        self.actor_target = Actor(state_size, action_size, hidden_size, config).to(self.device)

        # Critic网络
        self.critic1 = Critic(state_size, action_size, hidden_size).to(self.device)
        self.critic2 = Critic(state_size, action_size, hidden_size).to(self.device)
        self.critic1_target = Critic(state_size, action_size, hidden_size).to(self.device)
        self.critic2_target = Critic(state_size, action_size, hidden_size).to(self.device)

        # optimizer网络
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=config.lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=config.lr)
        self.alpha_optimizer = optim.Adam(params=[self.log_alpha], lr=config.lr)

        # ReplayBuffer
        self.buffer = ReplayBuffer(action_size, buffer_size=config.buffer_size, batch_size=config.batch_size)

    def step(self, state, action, reward, next_state, done, timestamp):
        """往buffer中保存经验， 并且使用随机抽样进行学习"""
        self.buffer.add(state, action, reward, next_state, done)

        if len(self.buffer) > self.config.batch_size and timestamp % self.config.learn_every == 0:
            for _ in range(self.config.n_updates):
                experiences = self.buffer.sample()
                self.learn(experiences)

    def get_action(self, state, add_noise=True):
        # 根据当前策略返回给定状态的操作，确定性策略
        state = torch.from_numpy(state).float().to(self.device)  # 增加一个维度给batch_size

        with torch.detach():
            action = self.actor.get_det_action(state)

        return action.numpy()

    def train(self, experiences):
        """
        使用一个批次的经验轨迹数据来更新值网络和策略网络
        价值Loss：
        critic_loss = MSE(Q, r + gamma * (min_critic_target(s', actor_target(a')) - alpha * log_pi(a'|s'))
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state)) ：这个是基于真实值的标签
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        策略Loss：
        actor_loss = alpha * log_pi(a|s) - Q(s,a)
        """
        gamma = self.config.gamma
        states, actions, rewards, next_states, dones = experiences

        # 目标模型中获取预测的下一个状态动作和 Q 值
        next_actions, log_probs = self.actor.act(next_states)
        q_target1_next = self.critic1_target(next_states, next_actions)
        q_target2_next = self.critic2_target(next_states, next_actions)

        # 为当前Q值计算价值
        q_targets = rewards + (1 - dones) * gamma * (torch.min(q_target1_next, q_target2_next) - self.alpha * log_probs)
        q_targets = q_targets.detach()

        q1 = self.critic1(states, actions).gather
        q2 = self.critic2(states, actions).gather

        critic1_loss = 0.5 * F.mse_loss(q1, q_targets)
        critic2_loss = 0.5 * F.mse_loss(q2, q_targets)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        clip_grad_norm_(self.critic1.parameters(), max_norm=self.config.max_grad_norm)
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        clip_grad_norm_(self.critic2.parameters(), max_norm=self.config.max_grad_norm)
        self.critic2_optimizer.step()

        # 更新策略函数的网络
        sample_action, sample_log_prob = self.actor.act(states)
        sample_q1 = self.critic1(states, actions)
        sample_q2 = self.critic2(states, actions)
        actor_loss = -(torch.min(sample_q1, sample_q2) - self.alpha * sample_log_prob)
        actor_loss = actor_loss.mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 更新alpha网络 mea()就是平均梯度（求期望），梯度上升：策略pi是一个函数theta，我们最大化函数值，变量theta对应最大致变量
        alpha_loss = -(self.alpha * (log_probs + self.target_entropy).detach().cpu()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp().detach()

        # 更新目标网络
        self.soft_update(self.critic1, self.critic1_target, tau=self.config.soft_update_tau)
        self.soft_update(self.critic2, self.critic2_target, tau=self.config.soft_update_tau)
        self.soft_update(self.actor, self.actor_target, tau=self.config.soft_update_tau)


    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

