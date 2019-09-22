import numpy as np
from sklearn.linear_model import LinearRegression


class DummyApproximator:
    def __init__(self):
        self.is_approximator = False
        self.value_weights = None
        self.num_value_features = 0

    def fit(self, *args, **kwargs):
        return None


class LinearFunction:
    def __init__(self, feature_type="bcts+rbf"):
        self.is_approximator = True
        self.name = "linear"
        if feature_type == "bcts+rbf":
            self.num_value_features = 13
        self.value_weights = np.zeros(self.num_value_features + 1)  # +1 is for intercept
        self.lin_reg = LinearRegression(n_jobs=1, fit_intercept=True)  # n_jobs must be 1... otherwise clashes with multiprocessing.Pool
        self.lin_reg.coef_ = np.zeros(self.num_value_features)
        self.lin_reg.intercept_ = 0.

    def fit(self, **rollout):
        self.lin_reg.fit(rollout['state_features'], rollout['state_values'])
        value_weights = np.hstack((self.lin_reg.intercept_, self.lin_reg.coef_))
        return value_weights

    # def fit(self, state_features, state_values):
    #     self.lin_reg.fit(state_features, state_values)
    #     value_weights = np.hstack((self.lin_reg.intercept_, self.lin_reg.coef_))
    #     return value_weights
