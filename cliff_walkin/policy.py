import numpy as np


def greedy_policy(action_spaces, q_table, state, explore=0.0):
    action = np.argmax(q_table[state])

    if np.random.rand() < explore:
        action = action_spaces.sample()

    return action