import timeit

setup = """
from numba import njit
import numpy as np
def softmax(U):
    ps = np.exp(U - np.max(U))
    ps /= np.sum(ps)
    return ps
@njit
def softmax_n(U):
    ps = np.exp(U - np.max(U))
    ps /= np.sum(ps)
    return ps
U = np.array([3.5, 5, 8, 1, 2, 10, 2.5, 2.7, 3.8, 12])
softmax(U)
softmax_n(U)
"""
n = 100000
print(timeit.timeit('softmax(U)', setup=setup, number=n))
print(timeit.timeit('softmax_n(U)', setup=setup, number=n))


setup = """
import tetris
import agents
import numpy as np
import torch
import torch.nn as nn
from numba import njit
device = torch.device('cpu')
import random
seed = 13; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
n_episodes = 10
print_interval = 1
num_columns = 10
num_rows = 16
tetromino_size = 4
verbose = False
do_learn = False
add_reward = False
epsilon = 0.0
feature_names = ['rows_with_holes', 'column_transitions', 'holes', 'landing_height', 'cumulative_wells',
                 'row_transitions', 'eroded', 'hole_depth']
num_features = len(feature_names)
feature_type = 'bcts'
mlp = nn.Sequential(
    nn.Linear(inum_features=num_features, out_features=1)
).to(device)
target_mlp = nn.Sequential(
    nn.Linear(inum_features=num_features, out_features=1)
).to(device)
def init_weights(m):
    print(m)
    if type(m) == nn.Linear:
        a = torch.Tensor([-24.04, -19.77, -13.08, -12.63, -10.49, -9.22, 6.6, -1.61])
        m.weight.data = a.view(1, len(a))
        b = torch.Tensor([0])
        m.bias.data = b
mlp.apply(init_weights)
for param in mlp.parameters():
    print(param.data)

agent = agents.TorchAgent(mlp=mlp, num_features=num_features, feature_type=feature_type,
                           epsilon=epsilon, add_reward=add_reward)

environment = tetris.Tetris(num_columns=num_columns, num_rows=num_rows, tetromino_size=tetromino_size,
                            agent=agent, verbose=verbose)

current_tetromino = environment.tetromino_sampler.next_tetromino()
after_states = np.array(current_tetromino.get_after_states(environment.current_state))

def f_pyt(after_states, agent):
    values = np.zeros(len(after_states))
    for ix, after_state in enumerate(after_states):
        features = after_state.features
        with torch.no_grad():
            values[ix] = agent.mlp(features)


def f_pyt_all(after_states, agent):
    features = torch.zeros(len(after_states), 8)
    for ix, after_state in enumerate(after_states):
        features[ix] = after_state.features
    with torch.no_grad():
        values = agent.mlp(features)
    values = values.numpy().flatten()

def f_np(after_states, agent):
    features = np.zeros((len(after_states), 8), dtype=np.float_)
    for ix, after_state in enumerate(after_states):
        features[ix] = after_state.features
    values = features.dot(agent.mlp.state_dict()['0.weight'].numpy()[0])



# weights = agent.mlp.state_dict()['0.weight'].numpy()[0]

# @njit
# def f_nb(after_states, weights):
#     values = np.zeros(len(after_states), dtype=np.float_)
#     for ix, after_state in enumerate(after_states):
#         features = after_state.features[0]
#         v = 0.0
#         for jx in range(len(weights)):
#             v += weights[jx] * features[jx]
#         values[ix] = v

f_pyt(after_states, agent)
f_pyt_all(after_states, agent)  
f_np(after_states, agent)      
# f_nb(after_states, weights)
"""

n = 100000
print(timeit.timeit('f_pyt(after_states, agent)', setup=setup, number=n))
print(timeit.timeit('f_pyt_all(after_states, agent)', setup=setup, number=n))
print(timeit.timeit('f_np(after_states, agent)', setup=setup, number=n))
# print(timeit.timeit('f_nb(after_states, weights)', setup=setup, number=n))



setup ="""
import tetris
import agents
import numpy as np
import current_state
from numba import jit
np.random.seed(9)
num_columns = 4
num_rows = 8
tetromino_size = 3

environment = tetris.Tetris(num_columns=num_columns, num_rows=num_rows, tetromino_size=tetromino_size,
                            agent=agents.RandomAgent(), verbose=False)
representation = np.array([[1, 1, 0, 1],
                           [1, 1, 1, 1],
                           [1, 0, 0, 1],
                           [1, 0, 0, 1],
                           [1, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0]])
st = current_state.State(representation=representation)
environment.current_state = st
environment.current_state.calc_lowest_free_rows()

@jit(nopython=True)
def get_lowest_free_row_1d(arr):
    last_occurrence = np.nonzero(arr)[0]
    if len(last_occurrence) == 0:
        lowest_free_row = 0
    else:
        lowest_free_row = last_occurrence[-1] + 1
    return lowest_free_row


@jit(nopython=True)
def calc_lowest_free_rows2(rep):
    n_cols = rep.shape[1]
    lowest_free_rows = np.zeros(n_cols, dtype=np.int64)
    for col in range(n_cols):
        lowest_free_rows[col] = get_lowest_free_row_1d(rep[:, col])
    return lowest_free_rows


@jit(nopython=True)
def calc_lowest_free_rows3(rep):
    num_rows, n_cols = rep.shape
    lowest_free_rows = np.zeros(n_cols, dtype=np.int64)
    for col_ix in range(n_cols):
        lowest = 0
        for row_ix in range(num_rows):
            if rep[row_ix, col_ix] == 1:
                lowest = row_ix + 1
        lowest_free_rows[col_ix] = lowest
    return lowest_free_rows


calc_lowest_free_rows2(environment.current_state.representation)
calc_lowest_free_rows3(environment.current_state.representation)
"""


print(timeit.timeit('environment.current_state.calc_lowest_free_rows()', setup=setup, number=100000))
print(timeit.timeit('calc_lowest_free_rows2(environment.current_state.representation)', setup=setup, number=100000))
print(timeit.timeit('calc_lowest_free_rows3(environment.current_state.representation)', setup=setup, number=100000))

setup = """
import numpy as np;
a = np.array([0, 1, 1, 0, 0, 0, 1, 1])
b = np.array([False, True, True, False, False, False, True, True])
c = np.array([0, 1, 1, 0, 0, 0, 1, 1], dtype=np.int64)
d = np.array([False, True, True, False, False, False, True, True], dtype=np.bool_)
e = np.array([0, 1, 1, 0, 0, 0, 1, 1], dtype=np.int8)
"""


print(timeit.timeit('np.sum(a)', setup=setup, number=1000000))
print(timeit.timeit('np.sum(b)', setup=setup, number=1000000))
print(timeit.timeit('np.sum(c)', setup=setup, number=1000000))
print(timeit.timeit('np.sum(d)', setup=setup, number=1000000))
print(timeit.timeit('np.sum(e)', setup=setup, number=1000000))

print(timeit.timeit('if a[1]: pass', setup=setup, number=10000000))
print(timeit.timeit('if b[1]: pass', setup=setup, number=10000000))
print(timeit.timeit('if c[1]: pass', setup=setup, number=10000000))
print(timeit.timeit('if d[1]: pass', setup=setup, number=10000000))
print(timeit.timeit('if e[1]: pass', setup=setup, number=10000000))


timeit.timeit('if 1: pass', number=10000000)
timeit.timeit('if True: pass', number=10000000)

