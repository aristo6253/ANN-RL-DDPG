import gym
import numpy as np


class NormalizedEnv(gym.ActionWrapper):
    """ Wrap action """

    def action(self, action):
        """ Rescale action to [-1, 1] """
        act_k = (self.action_space.high - self.action_space.low) / 2.
        act_b = (self.action_space.high + self.action_space.low) / 2.
        return act_k * action + act_b

    def reverse_action(self, action):
        """ Rescale action to [-1, 1] """
        act_k_inv = 2./(self.action_space.high - self.action_space.low)
        act_b = (self.action_space.high + self.action_space.low) / 2.
        return act_k_inv * (action - act_b)


class RandomAgent:
    def __init__(self, env, verbose=False):
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.shape[0]
        self.verbose = verbose
        if self.verbose:
            print(
                f'RandomAgent: state_size={self.state_size}, action_size={self.action_size}')

    def compute_action(self, state):
        if self.verbose:
            print(
                f'RandomAgent: compute_action -> {np.random.uniform(-1, 1, size=(self.action_size, ))}')
        return np.random.uniform(-1, 1, size=(self.action_size, ))
