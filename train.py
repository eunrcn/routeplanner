import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

from pedestrian_path_env import env, STATE_SIZE, ACTION_SIZE
from agents import SARSAAgent, QLearningAgent


# Training parameters
num_episodes = 10000  # Increased episodes for better convergence
max_steps_per_episode = 100  # Prevent infinitely long episodes if something goes wrong
goal_reward = 100.0  # A large positive reward for reaching the goal

# Initialize agents
sarsa_agent = SARSAAgent(env, state_size=STATE_SIZE, action_size=ACTION_SIZE)
q_learning_agent = QLearningAgent(env, state_size=STATE_SIZE, action_size=ACTION_SIZE)

# Store Q-table updates over episodes (optional, for analysis)
sarsa_q_tables: list[npt.NDArray[np.float64]] = []
q_learning_q_tables: list[npt.NDArray[np.float64]] = []

print("Starting Training...")

# Train both agents
for episode in range(num_episodes):
    # --- Run SARSA ---
    state = env.reset()
    action = sarsa_agent.choose_action(state)  # Choose first action
    done = False
    steps = 0
    # print(f"--- SARSA Episode {episode + 1} ---") # Uncomment for detailed logs

    while not done and steps < max_steps_per_episode:
        # Environment determines the actual next state based on the chosen action and probabilities
        actual_next_state, _, done = env.step(action)  # Use env.step to get the actual transition

        # Calculate reward based on state-action pair, handle goal state
        if done and actual_next_state == env.goal:
            reward = goal_reward
        elif done:  # Reached a terminal state which is not the goal (not possible in this env, but good practice)
            reward = -goal_reward  # Penalize ending prematurely if goal not reached
        else:
            reward = env.calculate_reward(state, action)  # Use score-based reward for intermediate steps

        # Choose the next action based on the actual next state
        next_action = sarsa_agent.choose_action(actual_next_state)

        # Update SARSA Q-table (On-policy: uses the action actually chosen for the next state)
        sarsa_agent.update(state, action, reward, actual_next_state, next_action)

        # Move to the next state and action
        state = actual_next_state
        action = next_action
        steps += 1

    # Store SARSA Q-table after episode
    sarsa_q_tables.append(sarsa_agent.q_table.copy())

    # --- Run Q-learning ---
    state = env.reset()
    done = False
    steps = 0
    # print(f"--- Q-Learning Episode {episode + 1} ---") # Uncomment for detailed logs

    while not done and steps < max_steps_per_episode:
        # Choose action based on current state
        action = q_learning_agent.choose_action(state)

        # Environment determines the actual next state based on the chosen action and probabilities
        actual_next_state, _, done = env.step(action)  # Use env.step to get the actual transition

        # Calculate reward based on state-action pair, handle goal state
        if done and actual_next_state == env.goal:
            reward = goal_reward
        elif done:
            reward = -goal_reward
        else:
            reward = env.calculate_reward(state, action)  # Use score-based reward for intermediate steps

        # Update Q-Learning Q-table (Off-policy: uses the best possible action for the next state)
        q_learning_agent.update(state, action, reward, actual_next_state)

        # Move to the next state
        state = actual_next_state
        steps += 1

    # Store Q-Learning Q-table after episode
    q_learning_q_tables.append(q_learning_agent.q_table.copy())

    # Optional: Print progress
    if (episode + 1) % (num_episodes // 10) == 0:
        print(f"Episode {episode + 1}/{num_episodes} completed.")

print("Training finished.")

# Example: Print final Q-tables (optional)
# print("\nFinal SARSA Q-Table:")
# print(sarsa_agent.q_table)
# print("\nFinal Q-Learning Q-Table:")
# print(q_learning_agent.q_table)

# Example: Extract and print learned policy for SARSA
print("\nLearned SARSA Policy (S1 to S14):")
state = "S1"
path_sarsa = [state]
visited_sarsa = {state}
while state != env.goal and len(path_sarsa) <= env.num_states:
    state_index = int(state[1:]) - 1
    # Choose best action according to learned Q-values (no exploration)
    action_index = np.argmax(sarsa_agent.q_table[state_index])
    next_state = "S" + str(action_index + 1)
    # Basic check to prevent cycles in policy extraction if learning is poor
    if next_state in visited_sarsa:
        print("  -> Cycle detected or stuck, stopping path extraction.")
        break
    path_sarsa.append(next_state)
    visited_sarsa.add(next_state)
    state = next_state
print(" -> ".join(path_sarsa))


# Example: Extract and print learned policy for Q-Learning
print("\nLearned Q-Learning Policy (S1 to S14):")
state = "S1"
path_q = [state]
visited_q = {state}
while state != env.goal and len(path_q) <= env.num_states:
    state_index = int(state[1:]) - 1
    # Choose best action according to learned Q-values (no exploration)
    action_index = np.argmax(q_learning_agent.q_table[state_index])
    next_state = "S" + str(action_index + 1)
    # Basic check to prevent cycles
    if next_state in visited_q:
        print("  -> Cycle detected or stuck, stopping path extraction.")
        break
    path_q.append(next_state)
    visited_q.add(next_state)
    state = next_state
print(" -> ".join(path_q))


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
