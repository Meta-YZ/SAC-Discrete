import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

from config import get_config
from collections import deque
from initialization import Init


def run(config):
    initialization = Init(config)
    initialization.init_seed()
    env = initialization.init_env()
    agent = initialization.init_agent()
    scores_deque = deque(maxlen=100)
    scores = []
    average_100_scores = []
    time_stamp = 0
    for ep_i in range(1, config.n_episodes + 1):
        state = env.reset()
        agent.reset()
        score = 0
        for t in range(config.episode_length):
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action*2)
            agent.step(state, action, reward, next_state, done, time_stamp)
            state = next_state
            score += reward
            time_stamp += 1

            if done:
                break

        scores_deque.append(score)
        scores.append(score)
        average_100_scores.append(np.mean(scores_deque))
        print(f'Episode {ep_i} Reward {score}  Average100 Score: {np.mean(scores_deque)}')

    fig = plt.figure(figsize=(15, 7))
    ax = fig.add_subplot(111)
    plt.plot(np.arange(1, len(scores) + 1), scores, label=" mean scores")
    plt.plot(np.arange(1, len(average_100_scores) + 1), average_100_scores, label="average_100")
    plt.legend()
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()


if __name__ == '__main__':
    parser = get_config()
    config = parser.parse_args()
    run(config)