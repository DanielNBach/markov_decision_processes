import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gymnasium as gym
import math
import warnings
import pickle
import os
import time
from matplotlib.colors import LinearSegmentedColormap
from algorithms.rl import RL
from algorithms.planner import Planner
from examples.plots import Plots

np.random.seed(8588)

# Used to find the number of iterations it took for algorithms to converge
# https://stackoverflow.com/a/47269413/22862849
def first_nonzero_row(arr, axis=1):
    def nonzero(row):
        return not np.any(row)

    idx = np.argmax((np.apply_along_axis(nonzero, axis=axis, arr=arr))[1:, ])
    return idx

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

def blackjack_value_iteration(**kwargs):
    print("================= Blackjack: Value Iteration =================")
    blackjack = Blackjack()
    t0 = time.time()
    V, V_track, pi = Planner(blackjack.P).value_iteration(**kwargs)
    t1 = time.time()
    average_reward, win, _, _ = play_with_policy(blackjack.env, pi=pi)
    return (t1-t0), V_track, average_reward, win

def blackjack_gamma_value_exp():
    gammas = [0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    stats = { "gamma" : gammas,
              "num_iter" : [],
              "reward": [],
              "win_percentage": [],

            }
    for gamma in gammas:
        _, V_track, average_reward, win = blackjack_value_iteration(gamma=gamma)
        iterations = first_nonzero_row(V_track, 1)
        stats['num_iter'].append(iterations)
        stats['reward'].append(average_reward)
        stats['win_percentage'].append(win)
    stats = pd.DataFrame(stats)
    print(stats)

def blackjack_policy_iteration(**kwargs):
    print("================= Blackjack: Policy Iteration =================")
    blackjack = Blackjack()
    t0 = time.time()
    V, V_track, pi = Planner(blackjack.P).policy_iteration(**kwargs)
    t1 = time.time()
    average_reward, win, _, _ = play_with_policy(blackjack.env, pi=pi)
    return (t1 - t0), V_track, average_reward, win

def blackjack_policy_vs_value(n_trials=1):
    policy = {
        'times': [],
        'iterations': [],
        'average_reward': [],
        'win_percentage': []
    }
    value = {
        'times': [],
        'iterations': [],
        'average_reward': [],
        'win_percentage': []
    }
    for _ in range(n_trials):
        time, V_track, average_reward, win = blackjack_value_iteration()
        value['times'].append(time)
        value['iterations'].append(first_nonzero_row(V_track, 1))
        value['average_reward'].append(average_reward)
        value['win_percentage'].append(win)

        time, V_track, average_reward, win = blackjack_policy_iteration()
        policy['times'].append(time)
        policy['iterations'].append(first_nonzero_row(V_track, 1))
        policy['average_reward'].append(average_reward)
        policy['win_percentage'].append(win)
    policy = pd.DataFrame(policy)
    value = pd.DataFrame(value)
    print("Policy iteration stats")
    print(policy.mean(axis=0))
    print("Value iteration stats")
    print(value.mean(axis=0))

def blackjack_q_learning( verbose=False, **kwargs):
    blackjack = Blackjack()
    t0 = time.time()
    Q, V, pi, Q_track, pi_track = RL(blackjack.env).q_learning(blackjack.n_states, blackjack.n_actions, blackjack.convert_state_obs, **kwargs)
    t1 = time.time()
    avg_reward, win_p, tie_p, lost_p = play_with_policy(blackjack.env, pi=pi, verbose=verbose)
    return (t1-t0), avg_reward, win_p, tie_p, lost_p

def blackjack_q_learning_iters(n_trials=3):
    episodes = [100, 1_000, 5_000, 10_000, 50_000, 100_000]
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

def play_with_policy(env, pi, num_episodes=100_000, convert_state_obs=Blackjack().convert_state_obs, verbose=True):
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

def taxi_with_policy(env, pi, num_episodes=10):
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
    t0 = time.time()
    V, V_track, pi = Planner(taxi.env.P).value_iteration(**kwargs)
    t1 = time.time()
    iterations = first_nonzero_row(V_track)
    runtime = t1 - t0
    average_reward = taxi_with_policy(taxi.env, pi)
    return runtime, iterations, average_reward

def taxi_gamma_exp(algorithm):
    gammas = [0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
    stats = { "gamma" : gammas,
              "iterations" : [],
              "reward": [],
              "runtime": [],
            }
    for gamma in gammas:
        runtime, iterations, avg_reward = algorithm(gamma=gamma)
        stats['iterations'].append(iterations)
        stats['reward'].append(avg_reward)
        stats['runtime'].append(runtime)
    stats = pd.DataFrame(stats)
    return stats

def taxi_value_vs_policy():
    value_stats = taxi_gamma_exp(taxi_value_iteration)
    policy_stats = taxi_gamma_exp(taxi_policy_iteration)
    for stat in ['iterations', 'reward', 'runtime']:
        fig, ax = plt.subplots()
        ax.plot(policy_stats['gamma'], policy_stats[stat], marker='o')
        ax.plot(value_stats['gamma'], value_stats[stat], marker='o')
        ax.legend(['Policy Iteration', 'Value Iteration'])
        plt.xlabel('gamma')
        plt.ylabel(f'{stat}')
        plt.title(f'Taxi {stat}: Policy vs. Value Iteration')
        plt.savefig(fname=f'Taxi_policy_value_{stat}_Results')
        plt.clf()

def taxi_policy_iteration(**kwargs):
    taxi = Taxi()
    t0 = time.time()
    V, V_track, pi = Planner(taxi.env.P).policy_iteration(**kwargs)
    t1 = time.time()
    iterations = first_nonzero_row(V_track)
    runtime = t1 - t0
    average_reward = taxi_with_policy(taxi.env, pi)
    return runtime, iterations, average_reward

def taxi_q_learning(**kwargs):
    taxi = Taxi()
    t0 = time.time()
    Q, V, pi, Q_track, pi_track = RL(taxi.env).q_learning(**kwargs)
    t1 = time.time()
    runtime = t1 -t0
    return runtime, pi, Q_track

def init_epsilon():
    epsilons = [0.3, 0.5, 0.7, 0.9, 1.0]
    data = {'epsilon' : epsilons,
            'q_track': []
           }
    for epsilon in epsilons:
        runtime, pi, Q_track = taxi_q_learning(init_epsilon=epsilon, n_episodes=500)
        data['q_track'].append(Q_track)
    data = pd.DataFrame(data)
    fig, ax = plt.subplots()
    for index, row in data.iterrows():
        Q_track_max = np.amax(np.amax(row['q_track'], axis=2), axis=1)
        ax.plot(pd.RangeIndex(len(Q_track)), Q_track_max)
    plt.xlabel('Iterations')
    plt.ylabel('Max Q-Value')
    plt.title("Max Q-Values for Initial Epsilons")
    ax.legend(['Init Epsilon: ' + str(ep) for ep in epsilons])
    plt.savefig('Init_Episilon_Q_Max_Iter')

def min_epsilon():
    epsilons = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
    data = {'epsilon' : epsilons,
            'q_track': []
           }
    for epsilon in epsilons:
        runtime, pi, Q_track = taxi_q_learning(min_epsilon=epsilon, n_episodes=500)
        data['q_track'].append(Q_track)
    data = pd.DataFrame(data)
    fig, ax = plt.subplots()
    for index, row in data.iterrows():
        Q_track_max = np.amax(np.amax(row['q_track'], axis=2), axis=1)
        ax.plot(pd.RangeIndex(len(Q_track)), Q_track_max)
    plt.xlabel('Iterations')
    plt.ylabel('Max Q-Value')
    plt.title("Max Q-Values for Minimum Epsilons")
    ax.legend(['Min Epsilon: ' + str(ep) for ep in epsilons])
    plt.savefig('Min_Epsilon_Q_Max_Iter')

def alpha():
    alphas = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    data = {'alpha' : alphas,
            'q_track': []
           }
    for alpha in alphas:
        runtime, pi, Q_track = taxi_q_learning(init_alpha=alpha, n_episodes=500)
        data['q_track'].append(Q_track)
    data = pd.DataFrame(data)
    fig, ax = plt.subplots()
    for index, row in data.iterrows():
        Q_track_max = np.amax(np.amax(row['q_track'], axis=2), axis=1)
        ax.plot(pd.RangeIndex(len(Q_track)), Q_track_max)
    plt.xlabel('Iterations')
    plt.ylabel('Max Q-Value')
    plt.title("Max Q-Values for Initial alphas")
    ax.legend(['Init alpha: ' + str(al) for al in alphas])
    plt.savefig('Init_Alpha_Q_Max_Iter')

def gamma():
    gammas = [0.1, 0.3, 0.5, 0.7, 0.8, 0.99]
    data = {'gamma' : gammas,
            'q_track': []
           }
    for gamma in gammas:
        runtime, pi, Q_track = taxi_q_learning(gamma=gamma, n_episodes=500)
        data['q_track'].append(Q_track)
    data = pd.DataFrame(data)
    fig, ax = plt.subplots()
    for index, row in data.iterrows():
        Q_track_max = np.amax(np.amax(row['q_track'], axis=2), axis=1)
        ax.plot(pd.RangeIndex(len(Q_track)), Q_track_max)
    plt.xlabel('Iterations')
    plt.ylabel('Max Q-Value')
    plt.title("Max Q-Values for gamma")
    ax.legend(['gamma: ' + str(gam) for gam in gammas])
    plt.savefig('Gamma_Q_Max_Iter')

def taxi_q_value_plot(**kwargs):
    _, _, Q_track = taxi_q_learning(**kwargs)
    max_q_value_per_iter = np.amax(np.amax(Q_track, axis=2), axis=1)
    v_iters_plot(max_q_value_per_iter, "Taxi Max Q-Values")

def v_iters_plot(data, label):
        df = pd.DataFrame(data=data)
        df.columns = [label]
        sns.set_theme(style="whitegrid")
        title = label + " v Iterations"
        sns.lineplot(x=df.index, y=label, data=df).set_title(title)
        plt.xlabel("Iterations")
        plt.savefig(f'{label}')
        plt.clf()
if __name__ == "__main__":
    #random_blackjack()
    blackjack_gamma_value_exp()
    blackjack_policy_vs_value()
    blackjack_q_learning_iters()
    taxi_value_vs_policy()
    taxi_q_value_plot()
    init_epsilon()
    min_epsilon()
    alpha()
    gamma()
