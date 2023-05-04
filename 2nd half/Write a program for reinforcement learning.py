import numpy as np

# Define the Q-table and learning rate
q_table = np.zeros((4, 2))
alpha = 0.1
gamma = 0.99

# Define the environment
env = [(0, 0), (0, 1), (0, 2), (0, 3)]

# Define the policy function
def epsilon_greedy_policy(state, epsilon):
    if np.random.random() < epsilon:
        return np.random.choice([0, 1])
    else:
        return np.argmax(q_table[state, :])

# Train the Q-Learning algorithm
for episode in range(1000):
    state = 0
    done = False
    while not done:
        # Choose an action using epsilon-greedy policy
        action = epsilon_greedy_policy(state, epsilon=0.1)

        # Take the action and observe the new state and reward
        if action == 0:
            next_state = max(state - 1, 0)
            reward = -1 if state > 0 else -10
        else:
            next_state = min(state + 1, len(env)-1)
            reward = 10 if state == 2 else 0

        # Update the Q-table using the Q-Learning update rule
        q_table[state, action] = (1 - alpha) * q_table[state, action] + alpha * (reward + gamma * np.max(q_table[next_state, :]))

        # Set the current state to the next state
        state = next_state

        # End the episode if the goal state is reached or too many time steps are taken
        if state == 2 or state == 0:
            done = True

# Test the trained Q-Learning algorithm
state = 0
done = False
while not done:
    # Choose an action with the highest Q-value for the current state
    action = np.argmax(q_table[state, :])

    # Take the action and observe the new state and reward
    if action == 0:
        next_state = max(state - 1, 0)
        reward = -1 if state > 0 else -10
    else:
        next_state = min(state + 1, len(env)-1)
        reward = 10 if state == 2 else 0

    # Render the current state and action
    print("State: {}, Action: {}, Reward: {}".format(env[state], action, reward))

    # Set the current state to the next state
    state = next_state

    # End the episode if the goal state is reached or too many time steps are taken
    if state == 2 or state == 0:
        done=True