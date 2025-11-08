# q_learning.py

import numpy as np
import config

class QLearningAgent:
    
    def __init__(self, num_relays):

        self.num_relays = num_relays

        self.learning_rate = config.LEARNING_RATE
        self.discount_factor = config.DISCOUNT_FACTOR
        self.epsilon = config.EPSILON
        self.epsilon_decay = config.EPSILON_DECAY
        self.min_epsilon = config.MIN_EPSILON

        self.Q = np.zeros((3, num_relays))

    def policy(self, state):
        
        circuit_pos = self._encode_state(state)

        if np.random.random() < self.epsilon:

            action = np.random.randint(self.num_relays)
        else:
            action = np.argmax(self.Q[circuit_pos])

        return action
    
    def update(self, state, action, reward, next_state, terminated):
        state_idx = self._encode_state(state)
        next_state_idx = self._encode_state(next_state)

        current_q = self.Q[state_idx, action]

        if terminated:
            max_next_q = 0
        else:
            max_next_q = np.max(self.Q[next_state_idx])

        target = reward + self.discount_factor * max_next_q
        new_q = current_q + self.learning_rate * (target - current_q)

        self.Q[state_idx, action] = new_q

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def _encode_state(self, state):
        return int(state[0] * 2)
        
