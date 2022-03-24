import os
import gym
import torch
import numpy as np

from agent import Agent
from pathlib import Path


class Init:
    def __init__(self, config):
        self.config = config
        self.state_size = 0
        self.action_size = 0
        torch.set_num_threads(self.config.num_threads)
        torch.set_default_dtype(torch.float32)

    def init_seed(self):
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        if torch.backends.cudnn.enabled:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

    def init_env(self):
        env = gym.make(self.config.env_name)
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.shape[0]
        return env #, self.state_size, self.action_size

    def init_agent(self):
        agent = Agent(self.state_size, self.action_size, hidden_size=self.config.hidden_size, config=self.config)
        return agent

    def init_results_dir(self):
        model_dir = Path('./models') / self.config.env_id / self.config.algorithm
        if not model_dir.exists():
            curr_run = 'run1'
        else:
            exist_run_nums = [int(str(folder.name).split('run')[1]) for folder in model_dir.iterdir()
                              if str(folder.name).startswith('run')]
            if len(exist_run_nums) == 0:
                curr_run = 'run1'
            else:
                curr_run = f'run{max(exist_run_nums)+1}'
        run_dir = model_dir / curr_run
        log_dir = run_dir / 'logs'
        os.makedirs(log_dir)
        return run_dir, log_dir


