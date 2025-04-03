import random
import numpy as np


class SARSAAgent:
    def __init__(self, env, state_size, action_size, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.state_size = state_size
        self.action_size = action_size
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.q_table = np.full((state_size, action_size), float('-inf'))  # Q-value table instantiate all as impossible actions first
        self.env = env

        #certain values in the q_table for select few states that can be reached from each state
        curr_state = 0
        for lst_next_state in env.possible_actions.values():
            for next_state in lst_next_state:
                state_index = int(next_state[1:])-1
                self.q_table[curr_state][state_index] = 0 #instantiate to 0 for all possible actions
            curr_state+=1
            if curr_state == self.state_size:
                break

    def choose_action(self, state): #str -> str
        """ Epsilon-greedy policy for action selection """
        if random.random() < self.epsilon:
            return random.choice(self.env.possible_actions[state]) # Explore
        else:
            state_index = int(state[1:])-1
            return "S" + str(np.argmax(self.q_table[state_index])+1)  # Exploit

    def update(self, state, action, reward, next_state, next_action):
        """ SARSA Q-value update rule. On-Policy """
        state_index = int(state[1:])-1
        action_index = int(action[1:])-1
        next_state_index = int(next_state[1:])-1
        next_action_index = int(next_action[1:])-1
        td_target = reward + self.gamma * self.q_table[next_state_index, next_action_index]
        td_error = td_target - self.q_table[state_index, action_index]
        self.q_table[state_index, action_index] += self.alpha * td_error  # Update rule

class QLearningAgent:
    def __init__(self, env, state_size, action_size, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.state_size = state_size
        self.action_size = action_size
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.q_table = np.full((state_size, action_size), float('-inf'))  # Q-value table instantiate all as impossible actions first
        self.env = env

        #certain values in the q_table for select few states that can be reached from each state
        curr_state = 0
        for lst_next_state in env.possible_actions.values():
            for next_state in lst_next_state:
                state_index = int(next_state[1:])-1
                self.q_table[curr_state][state_index] = 0 #instantiate to 0 for all possible actions
            curr_state+=1
            if curr_state == self.state_size:
                break

    def choose_action(self, state):
        """ Epsilon-greedy policy for action selection """
        if random.random() < self.epsilon:
            return random.choice(self.env.possible_actions[state]) # Explore
        else:
            state_index = int(state[1:])-1
            return "S" + str(np.argmax(self.q_table[state_index])+1)  # Exploit

    def update(self, state, action, reward, next_state):
        """ Q-Learning update rule (Off-policy TD control) """
        state_index = int(state[1:])-1
        action_index = int(action[1:])-1
        next_state_index = int(next_state[1:])-1
        best_next_action_index = np.argmax(self.q_table[next_state_index])  # Off-policy: Use max Q-value
        td_target = reward + self.gamma * self.q_table[next_state_index, best_next_action_index]
        td_error = td_target - self.q_table[state_index, action_index]
        self.q_table[state_index, action_index] += self.alpha * td_error  # Update rule
