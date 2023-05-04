import numpy as np

# Define the reward matrix
R = np.array([
    [-1, -1, -1, -1, 0, -1],
    [-1, -1, -1, 0, -1, 100],
    [-1, -1, -1, 0, -1, -1],
    [-1, 0, 0, -1, 0, -1],
    [0, -1, -1, 0, -1, 100],
    [-1, 0, -1, -1, 0, 100]
])

# Define the Q matrix
Q = np.zeros_like(R)

# Define hyperparameters
gamma = 0.8
num_episodes = 1000

# Train the agent
for episode in range(num_episodes):
    # Choose a random starting state
    state = np.random.randint(0, R.shape[0])
    while True:
        # Choose an action with epsilon-greedy policy
        if np.random.rand() < 0.2:
            action = np.random.randint(0, R.shape[1])
        else:
            action = np.argmax(Q[state])
        # Get the reward and next state
        next_state = np.argmax(R[action])
        reward = R[state, action]
        # Update the Q matrix
        Q[state, action] = reward + gamma * np.max(Q[next_state])
        # Move to the next state
        state = next_state
        if state == 5:
            break

# Test the agent
state = 2
paths = [state]
while state != 5:
    action = np.argmax(Q[state])
    state = np.argmax(R[action])
    paths.append(state)
print("Path taken:")
print(paths)