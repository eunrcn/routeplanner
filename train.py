import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import random

SEED = 3
random.seed(SEED)

from pedestrian_path_env import env, STATE_SIZE, ACTION_SIZE
from agents import SARSAAgent, QLearningAgent  # Assuming agents.py is correct


# Training parameters
num_episodes = 5000
max_steps_per_episode = 100  # Prevent infinitely long episodes

# Initialize agents - Make sure they use the same env instance
sarsa_agent = SARSAAgent(env, state_size=STATE_SIZE, action_size=ACTION_SIZE, alpha=0.1, gamma=0.99, epsilon=0.1)
q_learning_agent = QLearningAgent(
    env, state_size=STATE_SIZE, action_size=ACTION_SIZE, alpha=0.1, gamma=0.99, epsilon=0.1
)

# Store Q-table updates over episodes (optional, for analysis)
sarsa_q_tables: list[npt.NDArray[np.float64]] = []
q_learning_q_tables: list[npt.NDArray[np.float64]] = []

print("Starting Training with revised environment...")
print(f"Goal reward: {env.GOAL_REWARD}")
print(f"Weights: Weather={env.WEATHER_WEIGHT}, Safety={env.SAFETY_WEIGHT}, Time={env.TRAVEL_TIME_WEIGHT}")


# Train both agents
for episode in range(num_episodes):
    # --- Run SARSA ---
    state = env.reset()
    action = sarsa_agent.choose_action(state)  # Choose first action
    done = False
    steps = 0

    while not done and steps < max_steps_per_episode:
        next_state, reward, done = env.step(action)
        next_action = sarsa_agent.choose_action(next_state)

        # Update SARSA Q-table using the reward returned by the environment
        sarsa_agent.update(state, action, reward, next_state, next_action)

        # Move to the next state and action
        state = next_state
        action = next_action
        steps += 1

    # Store SARSA Q-table after episode
    sarsa_q_tables.append(sarsa_agent.q_table.copy())

    # --- Run Q-learning ---
    state = env.reset()  # Reset env for Q-learning agent
    done = False
    steps = 0

    while not done and steps < max_steps_per_episode:
        action = q_learning_agent.choose_action(state)
        next_state, reward, done = env.step(action)

        # Update Q-Learning Q-table using the reward returned by the environment
        q_learning_agent.update(state, action, reward, next_state)

        # Move to the next state
        state = next_state
        steps += 1

    # Store Q-Learning Q-table after episode
    q_learning_q_tables.append(q_learning_agent.q_table.copy())

    # Optional: Print progress
    if (episode + 1) % (num_episodes // 10) == 0:
        print(f"Episode {episode + 1}/{num_episodes} completed.")

print("Training finished.")

# --- [Keep the policy extraction and printing part as before] ---

# Example: Extract and print learned policy for SARSA
print("\nLearned SARSA Policy (S1 to S14):")
state = env.start_state  # Use start_state attribute
path_sarsa = [state]
visited_sarsa = {state}
current_steps = 0
while state != env.goal and current_steps < env.num_states * 2:  # Add step limit for safety
    state_index = int(state[1:]) - 1
    # Choose best action according to learned Q-values (no exploration)
    action_index = np.argmax(sarsa_agent.q_table[state_index])
    next_state = "S" + str(action_index + 1)

    # Check if the chosen action is actually possible from the current state
    if next_state not in env.possible_actions.get(state, []):
        print(f"  -> Error: SARSA policy chose invalid action {next_state} from {state}. Stopping.")
        # Find the best *valid* action instead
        valid_actions = env.possible_actions.get(state, [])
        if not valid_actions:
            print(f"  -> Error: No valid actions from {state}. Stopping.")
            break
        valid_action_indices = [int(a[1:]) - 1 for a in valid_actions]
        q_values_for_valid_actions = sarsa_agent.q_table[state_index, valid_action_indices]
        best_valid_action_local_index = np.argmax(q_values_for_valid_actions)
        action_index = valid_action_indices[best_valid_action_local_index]
        next_state = "S" + str(action_index + 1)
        print(f"  -> Corrected to best valid action: {next_state}")

    # Basic check to prevent cycles in policy extraction
    if next_state in visited_sarsa:
        print(f"  -> Cycle detected ({state} -> {next_state}), stopping path extraction.")
        # You might want to investigate why a cycle is the optimal path
        break
    path_sarsa.append(next_state)
    visited_sarsa.add(next_state)
    state = next_state
    current_steps += 1
if state == env.goal:
    print(" -> ".join(path_sarsa))
else:
    print(" -> ".join(path_sarsa) + " (Goal not reached)")


# Example: Extract and print learned policy for Q-Learning
print("\nLearned Q-Learning Policy (S1 to S14):")
state = env.start_state  # Use start_state attribute
path_q = [state]
visited_q = {state}
current_steps = 0
while state != env.goal and current_steps < env.num_states * 2:  # Add step limit
    state_index = int(state[1:]) - 1
    # Choose best action according to learned Q-values (no exploration)
    action_index = np.argmax(q_learning_agent.q_table[state_index])
    next_state = "S" + str(action_index + 1)

    # Check if the chosen action is actually possible from the current state
    if next_state not in env.possible_actions.get(state, []):
        print(f"  -> Error: Q-Learning policy chose invalid action {next_state} from {state}. Stopping.")
        # Find the best *valid* action instead
        valid_actions = env.possible_actions.get(state, [])
        if not valid_actions:
            print(f"  -> Error: No valid actions from {state}. Stopping.")
            break
        valid_action_indices = [int(a[1:]) - 1 for a in valid_actions]
        q_values_for_valid_actions = q_learning_agent.q_table[state_index, valid_action_indices]
        best_valid_action_local_index = np.argmax(q_values_for_valid_actions)
        action_index = valid_action_indices[best_valid_action_local_index]
        next_state = "S" + str(action_index + 1)
        print(f"  -> Corrected to best valid action: {next_state}")

    # Basic check to prevent cycles
    if next_state in visited_q:
        print(f"  -> Cycle detected ({state} -> {next_state}), stopping path extraction.")
        break
    path_q.append(next_state)
    visited_q.add(next_state)
    state = next_state
    current_steps += 1
if state == env.goal:
    print(" -> ".join(path_q))
else:
    print(" -> ".join(path_q) + " (Goal not reached)")


# Visualization: Q-table evolution for SARSA vs Q-learning
fig, axes = plt.subplots(1, 4, figsize=(16, 8))
row = 1
# Q-value convergence plots
axes[0].plot([q[12, 7] for q in sarsa_q_tables], label="SARSA 'S13' to 'S8'", color="blue")
axes[0].plot([q[12, 7] for q in q_learning_q_tables], label="Q-Learning 'S13' to 'S8'", color="red")
axes[0].set_title(f"'S13' to 'S8'' Value Progression")
axes[0].legend()

axes[1].plot([q[12, 8] for q in sarsa_q_tables], label="SARSA 'S13' to 'S9'", color="blue")
axes[1].plot([q[12, 8] for q in q_learning_q_tables], label="Q-Learning 'S13' to 'S9'", color="red")
axes[1].set_title(f"'S13' to 'S9' Value Progression")
axes[1].legend()

axes[2].plot([q[12, 11] for q in sarsa_q_tables], label="SARSA 'S13' to 'S12'", color="blue")
axes[2].plot([q[12, 11] for q in q_learning_q_tables], label="Q-Learning 'S13' to 'S12'", color="red")
axes[2].set_title(f"'S13' to 'S12' Value Progression")
axes[2].legend()

axes[3].plot([q[12, 13] for q in sarsa_q_tables], label="SARSA 'S13' to 'S14'", color="blue")
axes[3].plot([q[12, 13] for q in q_learning_q_tables], label="Q-Learning 'S13' to 'S14'", color="red")
axes[3].set_title(f"'S13' to 'S14' Value Progression")
axes[3].legend()

plt.tight_layout()
plt.show()
