import numpy as np
import time


class Cbapi:
    """
    This should become a general classification-based RL algorithm in the sense of
    Lagoudakis & Parr (2003), that is, using rollouts and directly approximating
    the policy using a policy_approximator state ---> action
    """
    def __init__(self,
                 policy_approximator,
                 value_function_approximator,
                 generative_model,
                 rollout_handler,
                 verbose):
        self.name = f"cbapi_{policy_approximator.name}_{value_function_approximator.name}_{rollout_handler.name}"
        self.policy_approximator = policy_approximator
        self.value_function_approximator = value_function_approximator
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
        # state_features, state_values, state_action_features, state_action_values, did_rollout, num_available_actions = \
        #     self.rollout_handler.perform_rollouts(self.policy_weights, self.value_weights, self.generative_model)
        # print("Rollouts")
        rollout = self.rollout_handler.perform_rollouts(self.policy_weights, self.value_weights, self.generative_model)
        # print("Value approximation")
        if self.value_function_approximator.is_approximator:
            self.value_weights = self.value_function_approximator.fit(**rollout)
        # self.policy_weights = self.policy_approximator.fit(state_action_features, state_action_values, did_rollout, num_available_actions)
        # print("Policy approximation")
        self.policy_weights = self.policy_approximator.fit(**rollout)

