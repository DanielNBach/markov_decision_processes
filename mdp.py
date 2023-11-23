import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gymnasium as gym
import json
import pickle
from algorithms.rl import RL
from algorithms.planner import Planner
from examples.blackjack import Blackjack
from collections import defaultdict

np.random.seed(8588)

"""
MDP 1: Blackjack
"""
def blackjack_value_iteration():
    blackjack = Blackjack()

if __name__ == "__main__":
    blackjack_value_iteration()