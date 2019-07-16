import numpy as np

import stew
import domtools
import cma

from tetris import mcts, tetromino, game
from tetris.utils import plot_analysis, plot_individual_agent
from numba import njit
from scipy.stats import binom_test
from sklearn.linear_model import LinearRegression
from statsmodels.stats.proportion import proportions_ztest


class RandomAgent:
    def __init__(self):
        self.name = "Random"
        self.policy_weights = None
        self.num_features = 0

    def choose_action(self, available_actions, env):
        return np.random.choice(available_actions), ""

    def choose_action_test(self, available_actions, env):
        return self.choose_action(available_actions, env)[0]

    def learn(self, old_state, reward, new_state, action, new_state_is_terminal):
        pass


class OptimalToyAgent:
    def __init__(self):
        self.name = "Optimal"
        self.policy_weights = None
        self.num_features = 0

    def choose_action(self, available_actions, env):
        if env.current_state.state_id in [0, 2]:
            action = 1
        else:
            action = 0
        return action, ""

    def choose_action_test(self, available_actions, env):
        return self.choose_action(available_actions, env)[0]

    def learn(self, old_state, reward, new_state, action, new_state_is_terminal):
        pass


class AgentQ():
    def __init__(self, policy="epsilon_greedy", epsilon=0.1, alpha=0.1, gamma=1,
                 num_nonterminal_states=3, num_actions=2):
        self.name = "AgentQ"
        self.q_table = np.zeros(shape=(num_nonterminal_states, num_actions))
        # Alternatively, initialize Q-table randomly:
        # self.q_table = np.random.normal(loc=0, scale=0.01, size=(self.environment.num_fields, self.environment.num_actions))
        self.policy = policy
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.policy_weights = None
        self.num_features = 0

    def choose_action(self, available_actions, env):
        if np.random.uniform(0, 1) < self.epsilon:
            action = np.random.choice(available_actions)
        else:
            q_values_of_state = self.q_table[env.current_state.state_id, :]
            maximum_q_values = np.where(q_values_of_state == np.amax(q_values_of_state))[0]
            action = np.random.choice(maximum_q_values)
        return action, ""

    def choose_action_test(self, available_actions, env):
        # No exploration!
        q_values_of_state = self.q_table[env.current_state.state_id, :]
        maximum_q_values = np.where(q_values_of_state == np.amax(q_values_of_state))[0]
        action = np.random.choice(maximum_q_values)
        return action

    def learn(self, old_state, reward, new_state, action, new_state_is_terminal):
        if new_state_is_terminal:
            max_q_value_in_new_state = 0.
        else:
            max_q_value_in_new_state = np.max(self.q_table[new_state.state_id, :])
        current_q_value = self.q_table[old_state.state_id, action]
        self.q_table[old_state.state_id, action] = (1 - self.alpha) * current_q_value + self.alpha * (reward + self.gamma * max_q_value_in_new_state)


class REINFORCE:
    def __init__(self, alpha, gamma, num_nonterminal_states=3, verbose=False,
                 baseline=False, alpha_v=0):
        self.alpha = alpha
        self.gamma = gamma
        self.num_features = num_nonterminal_states
        self.num_actions = 2
        self.policy_weights = np.zeros(self.num_features)
        self.verbose = verbose
        self.baseline = baseline
        if self.baseline:
            self.value_weights = np.zeros(self.num_features)
            self.alpha_v = alpha_v
            self.name = "REINFORCE_Baseline"
        else:
            self.name = "REINFORCE"

    def choose_action(self, available_actions, env):
        action_features = np.zeros((self.num_actions, self.num_features))
        for action in available_actions:
            start_state, rollout_reward, rollout_game_over = env.act(action)
            # start_state = game.ToyState(new_state_id, mdp=env)
            action_features[action, :] = start_state.get_features()

        utilities = action_features.dot(self.policy_weights)
        utilities = utilities - np.max(utilities)
        exp_utilities = np.exp(utilities)
        choice_probabilities = exp_utilities / np.sum(exp_utilities)
        chosen_action = np.random.choice(np.arange(len(utilities)), size=1, p=choice_probabilities)
        return chosen_action, action_features

    def choose_action_test(self, available_actions, env):
        action_features = np.zeros((len(available_actions), self.num_features))
        for action in available_actions:
            after_state, reward, game_over = env.current_state.act(action)
            action_features[action] = after_state.get_features()
        utilities = action_features.dot(self.policy_weights)
        max_indices = np.where(utilities == np.max(utilities))[0]
        chosen_action = np.random.choice(max_indices)
        return chosen_action

    def learn(self, action_features, actions, states, ep_rewards):
        T = len(states)
        for step in np.arange(T-1):
            G = 0.
            for k in np.arange(step, T-1):
                G += (self.gamma ** (k-step)) * ep_rewards[k]
            if self.baseline:
                state_features = states[step].get_features()
                delta = G - state_features.dot(self.value_weights)
                self.policy_weights += self.alpha * (self.gamma ** step) * delta * \
                                       grad_choice_probs(action_features[step], actions[step], self.policy_weights)[0]
                self.value_weights += self.alpha * delta * state_features
            else:
                # print("Last reward is ", ep_rewards[k])
                self.policy_weights += self.alpha * (self.gamma ** step) * G * grad_choice_probs(action_features[step], actions[step], self.policy_weights)[0]


class ActorCritic:
    def __init__(self, alpha, gamma, num_nonterminal_states=3, verbose=False, alpha_v=0):
        self.name = "ActorCritic"
        self.alpha = alpha
        self.gamma = gamma
        self.num_features = num_nonterminal_states
        self.num_actions = 2
        self.policy_weights = np.zeros(self.num_features)
        self.verbose = verbose

        self.value_weights = np.zeros(self.num_features)
        self.alpha_v = alpha_v

        self.ind = 1  # for gamma-discounting

    def choose_action(self, available_actions, env):
        action_features = np.zeros((self.num_actions, self.num_features))
        for action in available_actions:
            start_state, rollout_reward, rollout_game_over = env.act(action)
            # start_state = game.ToyState(new_state_id, mdp=env)
            action_features[action, :] = start_state.get_features()

        utilities = action_features.dot(self.policy_weights)
        utilities = utilities - np.max(utilities)
        exp_utilities = np.exp(utilities)
        choice_probabilities = exp_utilities / np.sum(exp_utilities)
        chosen_action = np.random.choice(np.arange(len(utilities)), size=1, p=choice_probabilities)
        return chosen_action, action_features

    def choose_action_test(self, available_actions, env):
        action_features = np.zeros((len(available_actions), self.num_features))
        for action in available_actions:
            after_state, reward, game_over = env.current_state.act(action)
            action_features[action] = after_state.get_features()
        utilities = action_features.dot(self.policy_weights)
        max_indices = np.where(utilities == np.max(utilities))[0]
        chosen_action = np.random.choice(max_indices)
        return chosen_action

    def learn(self, old_state, reward, new_state, action, new_state_is_terminal, action_features):
        if new_state_is_terminal:
            value_new_state = 0.
        else:
            value_new_state = new_state.get_features().dot(self.value_weights)
        features_old_state = old_state.get_features()
        delta = reward + self.gamma * value_new_state - features_old_state.dot(self.value_weights)
        self.policy_weights += self.alpha * self.ind * delta * grad_choice_probs(action_features, action, self.policy_weights)[0]
        self.value_weights += self.alpha * delta * features_old_state
        self.ind = self.ind * self.gamma


def grad_choice_probs(action_features, action_index, beta):
    utilities = action_features.dot(beta)
    utilities = utilities - np.max(utilities)
    exp_utilities = np.exp(utilities)
    choice_probabilities = exp_utilities / np.sum(exp_utilities)
    grad = action_features[action_index] - np.sum(choice_probabilities.reshape((-1, 1)) * action_features, axis=0)
    return grad


class ChoiceAgent:
    def __init__(self, alpha, num_rollouts=3, rollout_length=5, num_nonterminal_states=3,
                 lambda_max=4, lambda_min=-8.0, num_lambdas=100, verbose_stew=False):
        self.name = "Choice"
        self.num_features = num_nonterminal_states
        self.num_actions = 2
        self.num_rollouts = num_rollouts
        self.rollout_length = rollout_length
        self.policy_weights = np.zeros(self.num_features)

        # Data and model
        self.verbose_stew = verbose_stew
        self.max_choice_set_size = self.num_actions
        self.model = stew.StewMultinomialLogit(alpha=alpha, num_features=self.num_features, lambda_min=lambda_min,
                                               lambda_max=lambda_max, num_lambdas=num_lambdas, verbose=self.verbose_stew)
        # self.mlogit_data = MlogitData(num_features=self.num_features, max_choice_set_size=self.max_choice_set_size)

    def choose_action(self, available_actions, env):
        rewards = np.zeros((self.num_actions, self.num_rollouts))
        action_features = np.zeros((self.num_actions, self.num_features))
        for action in available_actions:
            start_state, rollout_reward, rollout_game_over = env.act(action)
            # start_state = game.ToyState(new_state_id, mdp=env)
            action_features[action, :] = start_state.get_features()
            for rollout in range(self.num_rollouts):
                rollout_state = start_state
                rollout_step = 1
                while rollout_step < self.rollout_length and not rollout_game_over:
                    chosen_action = self.choose_rollout_action(available_actions, rollout_state)
                    rollout_state, reward, rollout_game_over = rollout_state.act(chosen_action)
                    rollout_reward += reward
                    rollout_step += 1
                rewards[action, rollout] = rollout_reward
        mean_rewards = rewards.mean(axis=1)
        max_indices = np.where(mean_rewards == np.max(mean_rewards))[0]
        chosen_action = np.random.choice(max_indices)
        return chosen_action, action_features

    def choose_rollout_action(self, available_actions, current_state):
        action_features = np.zeros((len(available_actions), self.num_features))
        for action in available_actions:
            after_state, reward, game_over = current_state.act(action)
            action_features[action] = after_state.get_features()
        utilities = action_features.dot(self.policy_weights)
        max_indices = np.where(utilities == np.max(utilities))[0]
        chosen_action = np.random.choice(max_indices)
        return chosen_action

    def choose_action_test(self, available_actions, env):
        return self.choose_rollout_action(available_actions, env.current_state)

    def learn(self, action_features, action_index):
        # delete_oldest = self.delete_every > 0 and self.step_in_new_phase % self.delete_every == 0 and self.step_in_new_phase >= self.learn_from_step + 1
        # self.mlogit_data.push(features=action_features, choice_index=action_index, delete_oldest=delete_oldest)
        choice_set_len = len(action_features)
        one_hot_choice = np.zeros((choice_set_len, 1))
        one_hot_choice[action_index] = 1.
        choice_set_index = np.full(shape=(choice_set_len, 1), fill_value=1)
        data = np.hstack((choice_set_index, one_hot_choice, action_features))
        self.policy_weights = self.model.sgd_update(weights=self.policy_weights, data=data)


class MlogitData(object):
    def __init__(self, num_features, max_choice_set_size):
        self.num_features = num_features
        self.data = np.zeros((0, self.num_features + 2))
        self.choice_set_counter = 0.
        self.max_choice_set_size = max_choice_set_size

    def push(self, features, choice_index, delete_oldest=False):
        choice_set_len = len(features)
        one_hot_choice = np.zeros((choice_set_len, 1))
        one_hot_choice[choice_index] = 1.
        choice_set_index = np.full(shape=(choice_set_len, 1), fill_value=self.choice_set_counter)
        self.data = np.vstack((self.data, np.hstack((choice_set_index, one_hot_choice, features))))
        self.choice_set_counter += 1.
        if delete_oldest:
            first_choice_set_index = self.data[0, 0]
            for ix in range(self.max_choice_set_size):
                if self.data[ix, 0] != first_choice_set_index:
                    break
            if ix > self.max_choice_set_size:
                raise ValueError("Choice set should not be higher than 34.")
            self.data = self.data[ix:]

    def sample(self):
        # TODO: add sample_size (in terms of choice sets!)
        return self.data

    def delete_data(self):
        self.data = np.zeros((0, self.num_features + 2))

