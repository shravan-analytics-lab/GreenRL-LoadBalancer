import numpy as np
import random

class QLearningAgent:
    def __init__(self):
        self.q_table = np.zeros((3, 10, 3))  # traffic_level, server_count, actions
        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 0.2

    def get_state(self, requests, servers):
        traffic_level = min(int(requests // 50), 2)
        server_level = min(servers, 9)
        return traffic_level, server_level

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, 2)
        return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state):
        best_next = np.max(self.q_table[next_state])
        self.q_table[state][action] += self.alpha * (
            reward + self.gamma * best_next - self.q_table[state][action]
        )