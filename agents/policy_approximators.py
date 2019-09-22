import numpy as np
from stew import StewMultinomialLogit, ChoiceSetData
from tetris import tetromino
from numba import njit
import gc
import cma
import time
from run import utils_run


class MultinomialLogisticRegression:
    def __init__(self, regularization, feature_type="bcts"):
        self.name = "cmaes"
        if feature_type == "bcts":
            self.num_features = 8
        else:
            raise ValueError("Only BCTS features are implemented!")
        self.policy_weights = np.zeros(self.num_features)
        self.regularization = regularization

        # TODO: include function arguments
        self.fixed_lambda = 0
        self.max_choice_set_size = 34

        self.model = StewMultinomialLogit(num_features=self.num_features)
        self.mlogit_data = ChoiceSetData(num_features=self.num_features, max_choice_set_size=self.max_choice_set_size)

    def fit(self, **rollout):
        self.mlogit_data.delete_data()
        for ix in range(len(rollout['state_action_values'])):
            if rollout['did_rollout'][ix]:
                action_features = rollout['state_action_features'][ix][:rollout['num_available_actions'][ix], :]
                action_values = rollout['state_action_values'][ix][:rollout['num_available_actions'][ix]]
                choice_index = np.random.choice(np.flatnonzero(action_values == np.max(action_values)))
                self.mlogit_data.push(features=action_features,
                                      choice_index=choice_index,
                                      delete_oldest=False)
        if self.regularization in ["no_regularization", "nonnegative"]:
            policy_weights = self.model.fit(data=self.mlogit_data.sample(), lam=0, standardize=False)
        elif self.regularization == "stew_fixed_lambda":
            policy_weights = self.model.fit(data=self.mlogit_data.sample(), lam=self.fixed_lambda, standardize=False)
        elif self.regularization in ["ridge", "stew"]:
            policy_weights, _ = self.model.cv_fit(data=self.mlogit_data.sample())
        policy_weights = np.ascontiguousarray(policy_weights)
        return policy_weights
    #
    # if self.discrete_choice:
    #     self.mlogit_data.delete_data()
    #     # stacked_features = np.concatenate([self.state_action_features, axis=0)
    #     # TODO: rewrite to use less copying / memory...
    #     for state_ix in range(self.N):
    #         state_act_feat_ix = self.state_action_features[state_ix]
    #         if state_act_feat_ix is not None:
    #             action_values = self.state_action_values[state_ix, :len(state_act_feat_ix)]
    #             self.mlogit_data.push(features=self.state_action_features[state_ix], choice_index=np.argmax(action_values), delete_oldest=False)
    #     if self.ols:
    #         self.policy_weights = self.model.fit(data=self.mlogit_data.sample(), lam=0, standardize=False)
    #     # else:
    #     #     self.policy_weights, _ = self.model.cv_fit(data=self.mlogit_data.sample(), standardize=self.standardize_features)


class CmaesClassifier:
    def __init__(self, feature_type, cmaes_var, min_iterations, seed=0):
        self.name = "cmaes"
        if feature_type == "bcts":
            self.num_features = 8
        else:
            raise ValueError("Only BCTS features are implemented!")
        self.policy_weights = np.random.normal(loc=0, scale=cmaes_var, size=self.num_features)
        self.n = int(self.num_features * 15)  # CMA-ES parameter ('popsize' in cma)
        self.min_iterations = min_iterations
        self.cmaes_var = cmaes_var
        self.seed = seed
        self.cmaes = cma.CMAEvolutionStrategy(
            np.random.normal(loc=0, scale=1, size=self.num_features),
            cmaes_var,
            inopts={'verb_disp': 0,
                    'verb_filenameprefix': "output/cmaesout" + str(seed),
                    'popsize': self.n})

    def fit(self, **rollout):
        # state_action_features, state_action_values, did_rollout, num_available_actions
        # self.policy_approximator = cma.CMAEvolutionStrategy(np.random.normal(loc=0, scale=1, size=self.num_features), self.cmaes_var,
        #                                            inopts={'verb_disp': 0,
        #                                                    'verb_filenameprefix': "output/cmaesout" + str(self.seed),
        #                                                    'popsize': self.n})
        self.cmaes = cma.CMAEvolutionStrategy(
            np.random.normal(loc=0, scale=1, size=self.num_features),
            self.cmaes_var,
            inopts={'verb_disp': 0,
                    'verb_filenameprefix': "output/cmaesout" + str(self.seed),
                    'popsize': self.n})
        N = len(rollout['state_action_features'])
        policy_weights = self.cmaes.optimize(lambda x: policy_loss_function(x,
                                                                            N,
                                                                            rollout['did_rollout'],
                                                                            rollout['state_action_features'],
                                                                            rollout['num_available_actions'],
                                                                            rollout['state_action_values']),
                                             min_iterations=self.min_iterations).result.xbest
        return policy_weights


@njit(cache=False)
def policy_loss_function(pol_weights, N, did_rollout, state_action_features, num_available_actions,
                         state_action_values):
    loss = 0.
    number_of_samples = 0
    for state_ix in range(N):
        if did_rollout[state_ix]:
            values = state_action_features[state_ix, :num_available_actions[state_ix]].dot(pol_weights)
            pol_value = state_action_values[state_ix, np.argmax(values)]
            max_value = np.max(state_action_values[state_ix, :len(values)])
            loss += max_value - pol_value
            number_of_samples += 1
    loss /= number_of_samples
    return loss


