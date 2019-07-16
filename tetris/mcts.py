import numpy as np
import domtools


class Node:
    def __init__(self, state, tetromino, environment, parent=None, move_index=0, cp=1):
        self.state = state
        if tetromino is not None:
            # The root node has a fixed associated tetromino... children can have different tetrominos.
            self.tetromino = tetromino
            self.tetromino_name = type(self.tetromino).__name__
            self.child_priors = {self.tetromino_name: None}
            self.child_total_value = {self.tetromino_name: None}
            self.child_number_visits = {self.tetromino_name: None}
            self.child_features = {self.tetromino_name: None}
            self.children = {self.tetromino_name: np.array([])}
        self.environment = environment
        self.parent = parent
        self.move_index = move_index
        self.cp = cp

        # self.total_value = 0.0
        # self.number_visits = 0.0
        # self.prior = 1.0

        self.is_expanded = False
        # TODO: not IF terminal state
        # self.terminal_state = self.state.terminal_state

    @property
    def number_visits(self):
        return self.parent.child_number_visits[self.parent.tetromino_name][self.move_index]

    @number_visits.setter
    def number_visits(self, value):
        self.parent.child_number_visits[self.parent.tetromino_name][self.move_index] = value

    @property
    def total_value(self):
        return self.parent.child_total_value[self.parent.tetromino_name][self.move_index]

    @total_value.setter
    def total_value(self, value):
        self.parent.child_total_value[self.parent.tetromino_name][self.move_index] = value

    def child_Q(self):
        return self.child_total_value[self.tetromino_name] / self.child_number_visits[self.tetromino_name]

    def child_U(self):
        return np.sqrt(self.number_visits) * self.child_priors[self.tetromino_name] / self.child_number_visits[self.tetromino_name]

    def store_Q_estimate(self):

        self.state.value_estimate = self.parent.child_total_value[self.parent.tetromino_name][self.move_index] / \
                                    self.parent.child_number_visits[self.parent.tetromino_name][self.move_index]

    def best_child(self):
        print("self.child_number_visits")
        print(self.child_priors)
        print("self.child_priors")
        print(self.child_number_visits)
        compound = self.child_Q() + self.cp * self.child_U()
        return self.children[self.tetromino_name][np.random.choice(np.flatnonzero(compound == compound.max()))]

    def select_leaf(self):
        current = self
        # level = 0
        while current.is_expanded:  #and not current.state.terminal_state
            # level += 1
            current = current.best_child()
        # print(level)
        return current

    def expand(self):
        self.is_expanded = True
        children_states = self.tetromino.get_after_states(current_state=self.state)
        self.num_children = len(children_states)
        # TODO: implementing same tetromino for all
        # tetromino_tmp = self.environment.tetromino_sampler.next_tetromino()
        self.children[self.tetromino_name] = [Node(state=chil, tetromino=None,
                                                   environment=self.environment, parent=self,
                                                   move_index=chil_ix, cp=self.cp)
                                              for chil_ix, chil in enumerate(children_states)]

        # TODO: change priors from 1 to individual priors
        self.child_priors[self.tetromino_name] = np.ones(self.num_children, dtype=np.float32)
        self.child_total_value[self.tetromino_name] = np.zeros(self.num_children, dtype=np.float32)
        self.child_number_visits[self.tetromino_name] = np.zeros(self.num_children, dtype=np.float32)

    def backup(self, value_estimate):
        current = self
        while current.parent is not None:
            current.number_visits += 1
            current.total_value += value_estimate
            current = current.parent


class NodeRAC(Node):
    def __init__(self, state, tetromino, environment, dom_filter, cumu_dom_filter, feature_directors,
                 parent=None, move_index=0, cp=1):
        super().__init__(state, tetromino, environment, parent, move_index, cp)
        self.dom_filter = dom_filter
        self.cumu_dom_filter = cumu_dom_filter
        self.feature_directors = feature_directors

    def expand(self):
        self.is_expanded = True
        children_states = self.tetromino.get_after_states(current_state=self.state)
        # TODO: should have different num_children for different associated tetrominos
        self.num_children = len(children_states)
        # tetromino_tmp = self.environment.tetromino_sampler.next_tetromino()
        self.children[self.tetromino_name] = np.array([NodeRAC(state=chil, tetromino=None,
                                                               environment=self.environment,
                                                               dom_filter=self.dom_filter,
                                                               cumu_dom_filter=self.cumu_dom_filter,
                                                               feature_directors=self.feature_directors,
                                                               parent=self,
                                                               move_index=chil_ix, cp=self.cp)
                                                       for chil_ix, chil in enumerate(children_states)])
        self.child_features[self.tetromino_name] = np.zeros((self.num_children, self.state.num_features), dtype=np.float_)
        for ix in range(self.num_children):
            self.child_features[self.tetromino_name][ix] = self.children[self.tetromino_name][ix].state.get_features(direct_by=self.feature_directors)
        # if self.dom_filter or self.cumu_dom_filter:
        #     not_simply_dominated, not_cumu_dominated = domtools.dom_filter(self.child_features[self.tetromino_name],
        #                                                                    len_after_states=self.num_children)
        #     # for ix in range(self.num_children):
        #     #     print(ix)
        #     #     print(not_simply_dominated[ix])
        #     #     print(children_states[ix])
        #     # self.child_features[self.tetromino_name][np.array([16, 21, 25, 26, 29])]
        #     if self.cumu_dom_filter:
        #         self.children[self.tetromino_name] = self.children[self.tetromino_name][not_cumu_dominated]
        #         self.child_features[self.tetromino_name] = self.child_features[self.tetromino_name][not_cumu_dominated]
        #     else:  # Only simple dom
        #         self.children[self.tetromino_name] = self.children[self.tetromino_name][not_simply_dominated]
        #         self.child_features[self.tetromino_name] = self.child_features[self.tetromino_name][not_simply_dominated]
        #     self.num_children = len(self.child_features[self.tetromino_name])
        #     # TODO updating move_indexes could be made more efficient!???
        #     for ix in range(self.num_children):
        #         self.children[self.tetromino_name][ix].move_index = ix
        self.child_priors[self.tetromino_name] = np.ones(self.num_children, dtype=np.float32)
        self.child_total_value[self.tetromino_name] = np.zeros(self.num_children, dtype=np.float32)
        self.child_number_visits[self.tetromino_name] = np.zeros(self.num_children, dtype=np.float32)

    def filter(self):
        #TODO:  Filter terminal states.
        not_simply_dominated, not_cumu_dominated = domtools.dom_filter(self.child_features[self.tetromino_name],
                                                                       len_after_states=self.num_children)
        if self.cumu_dom_filter:
            self.children[self.tetromino_name] = self.children[self.tetromino_name][not_cumu_dominated]
            self.child_features[self.tetromino_name] = self.child_features[self.tetromino_name][not_cumu_dominated]
            self.child_priors[self.tetromino_name] = self.child_priors[self.tetromino_name][not_cumu_dominated]
            self.child_total_value[self.tetromino_name] = self.child_total_value[self.tetromino_name][not_cumu_dominated]
            self.child_number_visits[self.tetromino_name] = self.child_number_visits[self.tetromino_name][not_cumu_dominated]
            map_back_vector = np.nonzero(not_cumu_dominated)[0]
        else:  # Only simple dom
            self.children[self.tetromino_name] = self.children[self.tetromino_name][not_simply_dominated]
            self.child_features[self.tetromino_name] = self.child_features[self.tetromino_name][not_simply_dominated]
            self.child_priors[self.tetromino_name] = self.child_priors[self.tetromino_name][not_simply_dominated]
            self.child_total_value[self.tetromino_name] = self.child_total_value[self.tetromino_name][not_simply_dominated]
            self.child_number_visits[self.tetromino_name] = self.child_number_visits[self.tetromino_name][not_simply_dominated]
            map_back_vector = np.nonzero(not_simply_dominated)[0]
        self.num_children = len(self.child_features[self.tetromino_name])
        # TODO updating move_indexes could be made more efficient!???
        for ix in range(self.num_children):
            self.children[self.tetromino_name][ix].move_index = ix
        return map_back_vector

    def best_child(self):
        # print("self.child_number_visits")
        # print(self.child_priors)
        # print("self.child_priors")
        # print(self.child_number_visits)
        compound = self.child_Q() + self.cp * self.child_U()
        return self.children[self.tetromino_name][np.random.choice(np.flatnonzero(compound == compound.max()))]

class DummyNode(object):
    def __init__(self):
        self.parent = None
        self.tetromino_name = 0
        self.child_total_value = [[0.0]]
        self.child_number_visits = [[0.0]]


