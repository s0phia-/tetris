from game import Tetris
import numpy as np

env = Tetris(10, 10, False)
env.reset()
done = False
cleared_lines = 0
feature_directions = np.array([-1, -1, -1, -1, -1, -1, 1, -1])

for x in range(10):
    env.print_current_tetromino()  ##
    after_state_features = env.get_after_states()
    i = np.argmax([np.dot(y, feature_directions) for y in after_state_features])  # equal weights, directed
    observation, reward, done, _ = env.step(i)
    env.print_current_board()  ##
    cleared_lines += reward
    print(cleared_lines)  ##
    if done:
        env.reset()
        cleared_lines = 0

