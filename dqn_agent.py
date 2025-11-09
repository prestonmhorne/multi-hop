# dqn_agent.py

import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import numpy as np
import config

class DeepQNetwork(nn.Module):
    
    def __init__(self, state_dim, relay_feature_dim, hidden_dims=[512, 256, 128]):
        super(DeepQNetwork, self).__init__()
        layers = []
        in_dim = state_dim + relay_feature_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, state, relay_features):
        combined = torch.cat([state, relay_features], dim=-1)
        return self.network(combined)

class ReplayBuffer:

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, relay_features, reward, next_state, next_relay_features, done):
        self.buffer.append((state, action, relay_features, reward, next_state, next_relay_features, done))

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[idx] for idx in indices]

        states, _, relay_features, rewards, next_states, next_relay_features, dones = zip(*batch)
        return (
            np.array(states),
            np.array(relay_features),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(next_relay_features),
            np.array(dones, dtype=np.float32)
        )
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:

    def __init__(self, action_dim, state_dim, relay_feature_dim=4, hidden_dims=[512, 256, 128]):
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.relay_feature_dim = relay_feature_dim

        self.learning_rate = config.DQN_LEARNING_RATE
        self.discount_factor = config.DISCOUNT_FACTOR
        self.epsilon = config.DQN_EPSILON
        self.epsilon_decay = config.DQN_EPSILON_DECAY
        self.min_epsilon = config.DQN_MIN_EPSILON
        self.batch_size = config.DQN_BATCH_SIZE
        self.target_update_freq = config.DQN_TARGET_UPDATE_FREQ

        self.q_network = DeepQNetwork(state_dim, relay_feature_dim, hidden_dims)
        self.target_network = DeepQNetwork(state_dim, relay_feature_dim, hidden_dims)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)

        self.memory = ReplayBuffer(config.DQN_MEMORY_SIZE)

        self.steps = 0

    def _extract_relay_features(self, relay):
        return np.array([
            relay['bandwidth'] / config.MAX_BANDWIDTH,
            relay['latency'] / config.MAX_LATENCY,
            float(relay['guard_flag']),
            float(relay['exit_flag'])
        ], dtype=np.float32)

    def policy(self, state, action_mask=None, relay_info=None):
        if np.random.random() < self.epsilon:
            if action_mask is not None:
                valid_actions = np.where(action_mask)[0]
                return np.random.choice(valid_actions)
            return np.random.randint(self.action_dim)
        else:
            valid_actions = np.where(action_mask)[0] if action_mask is not None else range(self.action_dim)
            best_action = None
            best_q_value = -float('inf')

            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)

                for action in valid_actions:
                    relay_features = self._extract_relay_features(relay_info[action])
                    relay_features_tensor = torch.FloatTensor(relay_features).unsqueeze(0)
                    q_value = self.q_network(state_tensor, relay_features_tensor).item()

                    if q_value > best_q_value:
                        best_q_value = q_value 
                        best_action = action               
            return best_action

    def update(self, state, action, reward, next_state, terminated, relay_info, next_relay_info, next_action_mask):
        relay_features = self._extract_relay_features(relay_info[action])
        
        if not terminated:
            next_valid_actions = np.where(next_action_mask)[0]
            best_next_q = -float('inf')
            best_next_relay_features = None

            with torch.no_grad():
                next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)

                for next_action in next_valid_actions:
                    next_rf = self._extract_relay_features(next_relay_info[next_action])
                    next_rf_tensor = torch.FloatTensor(next_rf).unsqueeze(0)

                    q_val = self.target_network(next_state_tensor, next_rf_tensor).item()

                    if q_val > best_next_q:
                        best_next_q = q_val
                        best_next_relay_features = next_rf

        else: 
            best_next_relay_features = np.zeros(self.relay_feature_dim, dtype=np.float32)


        self.memory.push(state, action, relay_features, reward, next_state, best_next_relay_features, terminated)

        if len(self.memory) < self.batch_size:
            return
        
        states, relay_features_batch, rewards, next_states, next_relay_features_batch, dones = self.memory.sample(self.batch_size)

        states = torch.FloatTensor(states)
        relay_features_batch = torch.FloatTensor(relay_features_batch)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        next_relay_features_batch = torch.FloatTensor(next_relay_features_batch)
        dones = torch.FloatTensor(dones)

        current_q_values = self.q_network(states, relay_features_batch).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_network(next_states, next_relay_features_batch).squeeze(1)
            target_q_values = rewards + (1 - dones) * self.discount_factor * next_q_values

        loss = nn.MSELoss()(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)