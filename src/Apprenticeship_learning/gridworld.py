import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist


class GridWorld:
    def __init__(self, size: int = 8, n_features: int = 3):
        self.size = size
        self.n_features = n_features
        self.state = np.array([size // 2, size // 2])  # Start in middle
        self.actions = {
            'N': np.array([0, 1]),
            'S': np.array([0, -1]),
            'E': np.array([1, 0]),
            'W': np.array([-1, 0]),
            'NOOP': np.array([0, 0])
        }
        # Generate random RBF centers as common knowledge
        self.feature_centers = np.random.rand(n_features, 2) * size
        
    def get_features(self, state: np.ndarray) -> np.ndarray:
        distances = cdist(state.reshape(1, -1), self.feature_centers)
        distances_squared = distances[0]**2
        sigma = 1.0
        return np.exp(-distances_squared / (2 * sigma**2))
    
    def get_reward(self, state: np.ndarray, theta: np.ndarray) -> float:
        """Calculate reward for state given parameters theta"""
        features = self.get_features(state)
        return np.dot(features, theta)
    
    def step(self, action: str) -> Tuple[np.ndarray, float]:
        """Take action and return new state and reward"""
        if action not in self.actions:
            raise ValueError(f"Invalid action: {action}")
            
        new_state = self.state + self.actions[action]
        # Clip to ensure we stay in bounds
        new_state = np.clip(new_state, 0, self.size - 1)
        self.state = new_state
        return new_state.copy()
    
    def reset(self, random_state: bool = False):
        """Reset environment, optionally to random state"""
        if random_state:
            self.state = np.random.randint(0, self.size, size=2)
        else:
            self.state = np.array([self.size // 2, self.size // 2])
        return self.state.copy()

    def get_p_transition(self):
        """Return transition probabilities for the environment"""
        n_states = self.size * self.size
        n_actions = len(self.actions)  # N, S, E, W, NOOP
        p_transition = np.zeros((n_states, n_states, n_actions))

        # Helper function to map (x, y) to a state index
        def state_to_index(state):
            return state[0] + state[1] * self.size

        # Iterate over all states
        for x in range(self.size):
            for y in range(self.size):
                current_state = np.array([x, y])
                current_index = state_to_index(current_state)

                # Iterate over all actions
                for action_index, (action, delta) in enumerate(self.actions.items()):
                    next_state = current_state + delta
                    # Clip next_state to ensure it's within bounds
                    next_state = np.clip(next_state, 0, self.size - 1)
                    next_index = state_to_index(next_state)

                    # Deterministic transition
                    p_transition[current_index, next_index, action_index] = 1.0

        return p_transition
    
    def get_feature_matrix(self) :
        # Setup features matrix
        n_states = self.size * self.size
        features = np.zeros((n_states, self.n_features))
        for s in range(n_states):
            row, col = s // self.size, s % self.size
            state = np.array([row, col])
            features[s] = self.get_features(state)
        return features