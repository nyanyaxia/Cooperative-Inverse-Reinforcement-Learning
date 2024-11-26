from gridworld import GridWorld
import numpy as np
from typing import List, Tuple

class RobotLearner:
    def __init__(self, env: GridWorld):
        self.env = env
        self.estimated_theta = None
        
    def learn_from_demonstration(self, trajectory: List[Tuple[np.ndarray, str]]):
        """Implement Maximum Entropy IRL to learn reward parameters"""
        # Simplified MaxEnt IRL implementation
        feature_expectations = np.zeros(self.env.n_features)
        for state, _ in trajectory:
            feature_expectations += self.env.get_features(state)
        feature_expectations /= len(trajectory)
        
        # Initialize theta randomly from prior
        self.estimated_theta = np.random.uniform(-1, 1, self.env.n_features)
        
        # Simple gradient descent to match feature expectations
        learning_rate = 0.01
        n_iterations = 100
        
        for _ in range(n_iterations):
            current_features = np.zeros(self.env.n_features)
            state = self.env.reset()
            
            for _ in range(len(trajectory)):
                action = self._policy(state)
                state = self.env.step(action)
                current_features += self.env.get_features(state)
            
            current_features /= len(trajectory)
            gradient = feature_expectations - current_features
            self.estimated_theta += learning_rate * gradient
    
    def _policy(self, state: np.ndarray) -> str:
        """Robot's policy based on learned rewards"""
        if self.estimated_theta is None:
            return np.random.choice(['N', 'S', 'E', 'W'])
            
        best_action = 'NOOP'
        best_value = float('-inf')
        
        for action in ['N', 'S', 'E', 'W']:
            next_state = state + self.env.actions[action]
            next_state = np.clip(next_state, 0, self.env.size - 1)
            value = self.env.get_reward(next_state, self.estimated_theta)
            
            if value > best_value:
                best_value = value
                best_action = action
                
        return best_action