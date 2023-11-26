import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gymnasium as gym
import itertools
#import json
import pickle
import os
import time

from algorithms.rl import RL
from algorithms.planner import Planner
from examples.plots import Plots
from examples.test_env import TestEnv
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

def bj_statistics(rewards, num_episodes, verbose=True):
    rewards = np.asarray(rewards)
    average_reward = np.sum(rewards) / num_episodes
    uniques, counts = np.unique(rewards, return_counts=True)
    percentages = dict(zip(uniques, counts * 100 / len(rewards)))
    tie = 0.
    win = 0.
    lose = 0.
    for reward, percentage in percentages.items():
        if reward == -1.0:
            lose = percentage
        elif reward == 0.0:
            tie = percentage
        elif reward == 1.0:
            win = percentage
    if verbose:
        print(f'Average reward over {num_episodes} games: {average_reward}')
        print(f'Lost {lose}% of games')
        print(f'Tied {tie}% of games')
        print(f'Won {win}% of games')
    return average_reward, win, tie, lose

# Plays blackjack by randomly hitting or sticking
def random_blackjack(num_episodes=1_000_000):
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

def blackjack_value_iteration():
    print("================= Blackjack: Value Iteration =================")
    blackjack = Blackjack()
    V, V_track, pi = Planner(blackjack.P).value_iteration()
    play_with_policy(blackjack.env, pi=pi)
    return
    
def blackjack_policy_iteration():
    print("================= Blackjack: Policy Iteration =================")
    blackjack = Blackjack()
    V, V_track, pi = Planner(blackjack.P).policy_iteration()
    play_with_policy(blackjack.env, pi=pi)
    return

def blackjack_q_learning( verbose=False, **kwargs):
    blackjack = Blackjack()
    t0 = time.time()
    Q, V, pi, Q_track, pi_track = RL(blackjack.env).q_learning(blackjack.n_states, blackjack.n_actions, blackjack.convert_state_obs, **kwargs)
    t1 = time.time()
    avg_reward, win_p, tie_p, lost_p = play_with_policy(blackjack.env, pi=pi, verbose=verbose)
    return (t1-t0), avg_reward, win_p, tie_p, lost_p

def blackjack_q_learning_iters(n_trials=3):
    episodes = [100, 1_000,] #5_000, 10_000, 50_000, 100_000]
    stats = {
        "n_episodes" : episodes,
        "times": [],
        "average_reward": [],
        "win_percentage": [],
        "tie_percentage": [],
        "loss_percentage": []
    }
    for _ in range(n_trials):
        times = []
        average_rewards = []
        wins = []
        ties = []
        losses = []
        for n in episodes:
            time, reward, win_p, tie_p, lose_p = blackjack_q_learning(n_episodes=n)
            times.append(time)
            average_rewards.append(reward)
            wins.append(win_p)
            ties.append(tie_p)
            losses.append(lose_p)
        stats['times'].append(times)
        stats['average_reward'].append(average_rewards)
        stats['win_percentage'].append(wins)
        stats['tie_percentage'].append(ties)
        stats['loss_percentage'].append(losses)
    for stat, values in stats.items():
        if stat != 'n_episodes':
            stats[stat] = np.mean(np.asarray(values), axis=0)
    
    stats = pd.DataFrame(stats)
    for stat in ['times', 'average_reward', 'win_percentage']:
        fig, ax = plt.subplots()
        ax.plot(stats['n_episodes'], stats[stat], marker='o')
        plt.xlabel('n_episodes')
        plt.ylabel(f'{stat}')
        plt.title(f'Q-Learning {stat} with Increasing n_episodes')
        plt.savefig(fname=f'BJ_Q_Learning_n_episodes_{stat}_Results')
        plt.clf()
    return

def play_with_policy(env, pi, num_episodes=1000, convert_state_obs=Blackjack().convert_state_obs, verbose=True):
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
    return bj_statistics(rewards, num_episodes, verbose=verbose)

"""
MDP 2: Taxi
"""
class Taxi:
    def __init__(self):
        self.env = gym.make('Taxi-v3', render_mode=None)

def taxi_with_policy(env, pi, num_episodes=5):
    rewards = []
    for _ in range(num_episodes):
        state, _= env.reset()
        done = False
        while not done:
            action = pi(state)
            next_state, reward, terminated, truncated, _= env.step(action)
            done = terminated or truncated
            state = next_state
            if done:
                rewards.append(reward)
    env.close()
    rewards = np.asarray(rewards)
    return np.mean(rewards)

def taxi_value_iteration(**kwargs):
    taxi = Taxi()
    V, V_track, pi = Planner(taxi.env.P).value_iteration(**kwargs)
    #test_scores = TestEnv.test_env(env=taxi.env, render=True, user_input=False, pi=pi)
    taxi_with_policy(taxi.env, pi)

def taxi_policy_iteration(**kwargs):
    taxi = Taxi()
    V, V_track, pi = Planner(taxi.env.P).policy_iteration(theta=1e-5, **kwargs)
    #test_scores = TestEnv.test_env(env=taxi.env, render=True, user_input=False, pi=pi)
    taxi_with_policy(taxi.env, pi)

def taxi_q_learning(**kwargs):
    taxi = Taxi()
    t0 = time.time()
    Q, V, pi, Q_track, pi_track = RL(taxi.env).q_learning(**kwargs)
    t1 = time.time()
    return (t1- t0)

if __name__ == "__main__":
    bj = Blackjack()
    print(bj.n_states)
    print(bj.n_actions)
    #random_blackjack()
    #blackjack_value_iteration()
    #blackjack_policy_iteration()
    #blackjack_q_learning_iters()
    #taxi_value_iteration()
    #taxi_policy_iteration()
