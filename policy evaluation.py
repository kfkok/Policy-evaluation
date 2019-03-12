import numpy as np
import math

states = np.array([
    0, 1, 2, 3,
    4, 5, 6, 7,
    8, 9, 10, 11,
    12, 13, 14
])
actions = np.array([0, 1, 2, 3])  # up, down, right, left
N_STATES = len(states)
N_ACTIONS = len(actions)

# State transition probability
STATE_TRANSITION = np.zeros((N_STATES, N_ACTIONS, N_STATES))
for state in states:
    for action in actions:
        # move up
        if action == 0:
            up_state = state - 4
            if up_state < 0:
                up_state = state
            STATE_TRANSITION[state, action, up_state] = 1

        # move down
        if action == 1:
            down_state = state + 4
            if down_state > N_STATES:
                down_state = state
            elif down_state == N_STATES:
                down_state = 0
            STATE_TRANSITION[state, action, down_state] = 1

        # move right
        elif action == 2:
            right_state = state + 1
            if right_state % 4 == 0:
                right_state = state
            elif right_state == N_STATES:
                right_state = 0  # terminal state
            STATE_TRANSITION[state, action, right_state] = 1

        # move left
        elif action == 3:
            left_state = state - 1
            if left_state < 0 or state % 4 == 0:
                left_state = state
            STATE_TRANSITION[state, action, left_state] = 1

# set transition probability for terminal state(state 0) = 0
STATE_TRANSITION[0, :, :] = 0

# Reward function
REWARD = np.zeros(N_STATES) - 1

gamma = 1

# Value function
V = np.zeros(N_STATES)

# Policy
policy = np.empty((N_STATES, 4))

# Initialize probability for each action in each state
for state in states:
    action_probabilities = [0.25, 0.25, 0.25, 0.25]
    policy[state] = action_probabilities


policy[0] = np.zeros(4)

max_delta = 0
iteration = 0

while True:
    max_delta = 0
    old_V = V.copy()

    for state in states:
        v = old_V[state]
        V[state] = sum([policy[state][action] * sum([STATE_TRANSITION[state, action, next_state] * (REWARD[next_state] + gamma * old_V[next_state]) for next_state
            in states]) for action in actions])

        max_delta = max(max_delta, abs(V[state] - v))

    iteration += 1

    # reshape the value into 4 by 4 matrix for visualization
    print("iteration {}".format(iteration))
    temp = V.copy()
    temp = np.append(temp, temp[0])
    print(np.reshape(temp, (4, 4)))
    print("\n")

    if max_delta < 0.001:
        break



