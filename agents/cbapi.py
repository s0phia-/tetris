import numpy as np


class Cbapi:
    """
    This should become a general classification-based RL algorithm in the sense of
    Lagoudakis & Parr (2003), that is, using rollouts and directly approximating
    the policy using a policy_approximator state ---> action
    """
    def __init__(self, classifier):
        self.classifier = classifier

    def choose_action(self):
        pass

    def learn(self):
        pass

    def append_data(self):
        pass