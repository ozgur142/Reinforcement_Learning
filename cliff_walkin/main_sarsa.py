import gymnasium as gym
import numpy as np
import pickle as pkl


env = gym.make('CliffWalking-v0', render_mode='human')

q_table = pkl.load(open('sarsa.pkl', 'rb'))



done = False
total_reward = 0
state = env.reset()[0]

while not done:
    action = np.argmax(q_table[state])

    state, reward, done, info, _ = env.step(action)
    total_reward += reward

env.close()

print('Total Reward:', total_reward)