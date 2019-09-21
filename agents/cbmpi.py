import numpy as np
from stew import StewMultinomialLogit, ChoiceSetData
from tetris import tetromino
from numba import njit
from sklearn.linear_model import LinearRegression
import gc
import cma
import time





class Cbmpi:
    def __init__(self,
                 policy_approximator,
                 value_function_approximator,
                 generative_model,
                 rollout_handler,
                 verbose):
        self.name = "cbmpi"
        self.policy_approximator = policy_approximator
        self.value_function_approximator = value_function_approximator  # TODO
        self.generative_model = generative_model
        self.rollout_handler = rollout_handler

        self.num_features = self.policy_approximator.num_features
        self.num_value_features = self.value_function_approximator.num_value_features
        self.policy_weights = policy_approximator.policy_weights
        self.value_weights = value_function_approximator.value_weights

        self.verbose = verbose
        self.is_learning = True
        self.step = 0

    def copy_current_policy_weights(self):
        return self.policy_weights.copy()

    def update_steps(self):
        """ Generic function called from certain RL run procedures. Pretty boring and useless here!"""
        self.step += 1

    def learn(self, *args, **kwargs):
        start_time = time.time()
        state_features, state_values, state_action_features, state_action_values, did_rollout, num_available_actions = \
            self.rollout_handler.perform_rollouts(self.policy_weights, self.value_weights, self.generative_model)
        if self.verbose:
            print("Rollouts took " + str((time.time() - start_time) / 60) + " minutes.")

        start_time = time.time()
        self.value_weights = self.value_function_approximator.fit(state_features, state_values)
        self.policy_weights = self.policy_approximator.fit(state_action_features, state_action_values, did_rollout, num_available_actions)

        if self.verbose:
            print("Function approximation took " + str((time.time() - start_time) / 60) + " minutes.")
            print("New value_weights: ", self.value_weights)
            print("New policy_weights: ", self.policy_weights)


