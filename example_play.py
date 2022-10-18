from game import Tetris
import numpy as np

feature_directions = np.array([-1, -1, -1, -1, -1, -1, 1, -1])

env = Tetris(10, 10, False, feature_directions=feature_directions)
env.reset()
done = False
cleared_lines = 0

for x in range(100):
    env.print_current_tetromino()  ##
    after_state_features = env.get_after_states()
    i = np.argmax([np.sum(y) for y in after_state_features])  # equal weights, directed
    observation, reward, done, _ = env.step(i)
    env.print_current_board()  ##
    cleared_lines += reward
    print(cleared_lines)  ##
    if done:
        env.reset()
        # cleared_lines = 0
