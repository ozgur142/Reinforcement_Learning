import gymnasium as gym
import numpy as np
import pickle as pkl

# 0: Move up
# 1: Move right
# 2: Move down
# 3: Move left

# Each time step incurs -1 reward, unless the player stepped into the cliff, which incurs -100 reward.

EPSILON = 0.1
ALPHA = 0.1
GAMMA = 0.9
NUM_EPISODES = 500

env = gym.make('CliffWalking-v0')

q_table = np.zeros([env.observation_space.n, env.action_space.n]) # 48x4

def policy(state, explore=0.0):
    action = np.argmax(q_table[state])

    if np.random.rand() < explore:
        action = env.action_space.sample()

    return action


for episode in range(NUM_EPISODES):
    
    done = False
    total_reward = 0
    episode_length = 0

    state = env.reset()[0]
    action = policy(state, EPSILON)

    while not done:  
        #print('Action:', action)
        next_state, reward, done, info, _ = env.step(action)
        #print(reward)
        next_action = policy(next_state, EPSILON)

        #update q_table
        q_table[state][action] += ALPHA * (reward + GAMMA * q_table[next_state][next_action] - q_table[state][action])

        state = next_state
        action = next_action

        total_reward += reward
        episode_length += 1


    print('Episode length:', episode_length, 'Total Reward:', total_reward)

env.close()


pkl.dump(q_table, open('sarsa.pkl', 'wb'))
print("training complete. Model saved as 'sarsa.pkl'")
