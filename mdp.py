import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gymnasium as gym
import json
import pickle
import os

from algorithms.rl import RL
from algorithms.planner import Planner
from examples.plots import Plots
from collections import defaultdict

np.random.seed(8588)

"""
MDP 1: Blackjack
"""
# Taken from: https://github.com/jlm429/bettermdptools/blob/master/examples/blackjack.py
class Blackjack:
    def __init__(self, natural=False, sab=True):
        self._env = gym.make('Blackjack-v1', natural=natural, sab=sab, render_mode=None)
        self._convert_state_obs = lambda state, done: (
            -1 if done else int(f"{state[0] + 6}{(state[1] - 2) % 10}") if state[2] else int(
                f"{state[0] - 4}{(state[1] - 2) % 10}"))
        # Transitions and rewards matrix from: https://github.com/rhalbersma/gym-blackjack-v1
        current_dir = os.path.dirname(__file__)
        file_name = 'blackjack-envP'
        f = os.path.join(current_dir, file_name)
        try:
            self._P = pickle.load(open(f, "rb"))
        except IOError:
            print("Pickle load failed.  Check path", f)
        self._n_actions = self.env.action_space.n
        self._n_states = len(self._P)

    @property
    def n_actions(self):
        return self._n_actions

    @n_actions.setter
    def n_actions(self, n_actions):
        self._n_actions = n_actions

    @property
    def n_states(self):
        return self._n_states

    @n_states.setter
    def n_states(self, n_states):
        self._n_states = n_states

    @property
    def P(self):
        return self._P

    @P.setter
    def P(self, P):
        self._P = P

    @property
    def env(self):
        return self._env

    @env.setter
    def env(self, env):
        self._env = env

    @property
    def convert_state_obs(self):
        return self._convert_state_obs

    @convert_state_obs.setter
    def convert_state_obs(self, convert_state_obs):
        self._convert_state_obs = convert_state_obs

def bj_statistics(rewards, num_episodes):
    rewards = np.asarray(rewards)
    print(f'Average reward over {num_episodes} games: {np.sum(rewards) / num_episodes}')
    uniques, counts = np.unique(rewards, return_counts=True)
    percentages = dict(zip(uniques, counts * 100 / len(rewards)))
    for reward, percentage in percentages.items():
        if reward == -1.0:
            print(f'Lost {percentage}% of games')
        elif reward == 0.0:
            print(f'Tied {percentage}% of games')
        elif reward == 1.0:
            print(f'Won {percentage}% of games')

# Plays blackjack by randomly hitting or sticking
def random_blackjack(num_episodes=10_000):
    print('Blackjack with randomized actions')
    env = gym.make('Blackjack-v1')
    rewards = []
    for _ in range(num_episodes):
        # Reset the game state
        env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            _, reward, done, _, _ = env.step(action)
        rewards.append(reward)
    env.close()
    rewards = np.asarray(rewards)
    bj_statistics(rewards, num_episodes)
    return

def blackjack_value_iteration(num_episodes=1_000_000, convert_state_obs=Blackjack().convert_state_obs):
    print("================= Value Iteration =================")
    blackjack = Blackjack()
    V, V_track, pi = Planner(blackjack.P).value_iteration()
    play_with_policy(blackjack.env, pi=pi)
    return
    
def blackjack_policy_iteration():
    print("================= Policy Iteration =================")
    blackjack = Blackjack()
    V, V_track, pi = Planner(blackjack.P).policy_iteration()
    play_with_policy(blackjack.env, pi=pi)
    return
    
def play_with_policy(env, pi, num_episodes=100_000, convert_state_obs=Blackjack().convert_state_obs):
    rewards = []
    for _ in range(num_episodes):
        observation, _= env.reset()
        done = False
        state = convert_state_obs(observation, done)
        while not done:
            action = pi(state)
            next_obs, reward, terminated, truncated, _= env.step(action)
            done = terminated or truncated
            next_state = convert_state_obs(next_obs, done)
            state = next_state
            if done:
                rewards.append(reward)
    env.close()
    rewards = np.asarray(rewards)
    bj_statistics(rewards, num_episodes)
if __name__ == "__main__":
    #random_blackjack()
    blackjack_value_iteration()
    #blackjack_policy_iteration()