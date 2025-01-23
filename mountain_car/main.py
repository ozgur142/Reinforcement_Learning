# Q-Learning for Mountain Car
import gymnasium as gym
import numpy as np

# action_space: Discrete(3) -> 0: left, 1: stop, 2: right
env = gym.make("MountainCar-v0", render_mode="rgb_array")  # Use "human" for rendering during testing
state, _ = env.reset()

# Hyperparameters
learning_rate = 0.1
discount_factor = 0.99
epsilon = 1.0  # Initial exploration rate
epsilon_decay = 0.995  # Decay rate for exploration
epsilon_min = 0.01
num_episodes = 1000  # Number of episodes to train
max_steps_per_episode = 200

# Discretization settings
state_bins = [20, 20]  # Number of bins for each state dimension
state_bounds = list(zip(env.observation_space.low, env.observation_space.high))
action_space_size = env.action_space.n

# Create Q-table
q_table = np.random.uniform(low=-2, high=0, size=(state_bins[0], state_bins[1], action_space_size))

# Helper function to discretize a state
def discretize_state(state):
    discretized = []
    for i in range(len(state)):
        scale = (state[i] - state_bounds[i][0]) / (state_bounds[i][1] - state_bounds[i][0])
        discretized.append(int(np.floor(scale * state_bins[i])))
    return tuple(np.clip(discretized, 0, np.array(state_bins) - 1))

# Training Loop
for episode in range(num_episodes):
    state, _ = env.reset()
    state_discrete = discretize_state(state)
    done = False
    total_reward = 0

    for step in range(max_steps_per_episode):
        # Epsilon-greedy action selection
        if np.random.random() < epsilon:
            action = env.action_space.sample()  # Explore
        else:
            action = np.argmax(q_table[state_discrete])  # Exploit

        # Perform action
        new_state, reward, done, truncated, info = env.step(action)
        new_state_discrete = discretize_state(new_state)
        total_reward += reward

        # Q-learning update rule
        q_value = q_table[state_discrete + (action,)]
        max_future_q = np.max(q_table[new_state_discrete])
        new_q_value = q_value + learning_rate * (reward + discount_factor * max_future_q - q_value)
        q_table[state_discrete + (action,)] = new_q_value

        # Move to the next state
        state_discrete = new_state_discrete

        if done or truncated:
            break


    # Decay epsilon
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    if new_state[0] >= 0.5:
        print(f"Goal reached in episode {episode + 1} with total reward = {total_reward}")
    elif episode % 100 == 0:
        print(f"Episode {episode + 1}: Total Reward = {total_reward}")



# Testing the trained policy
env = gym.make("MountainCar-v0", render_mode="human")
state, _ = env.reset()
state_discrete = discretize_state(state)
done = False

while not done:
    action = np.argmax(q_table[state_discrete])  # Always take the best action
    new_state, reward, done, truncated, info = env.step(action)
    env.render()
    state_discrete = discretize_state(new_state)

env.close()
