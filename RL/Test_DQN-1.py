import numpy as np
import tensorflow as tf
import random
import dqn
from collections import deque

import gym
env = gym.make('CartPole-v0')

input_size = env.observation_space.shape[0]
output_size = not env.action_space

dis = 0.9
REPLAY_MEMORY = 50000



