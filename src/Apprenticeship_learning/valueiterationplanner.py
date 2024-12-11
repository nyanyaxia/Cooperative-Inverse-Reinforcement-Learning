from gridworld import GridWorld
import numpy as np
from collections import defaultdict
from typing import Dict

class ValueIterationPlanner:
    def __init__(self, env: GridWorld, gamma: float = 0.95, epsilon: float = 1e-6):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.actions = ['N', 'S', 'E', 'W']
        # Cache for value functions and policies under different thetas
        self.value_cache = {}
        self.policy_cache = {}
        
    def _state_to_tuple(self, state: np.ndarray) -> tuple:
        return tuple(state.flatten())
        
    def _get_cached_policy(self, theta_key: tuple):
        """Get cached policy or compute if not available"""
        if theta_key not in self.policy_cache:
            V = self.compute_values(theta_key)
            self.policy_cache[theta_key] = self.compute_policy(V, theta_key)
        return self.policy_cache[theta_key]
    
    def compute_values(self, theta_key: tuple) -> Dict[tuple, float]:
        """Compute optimal value function for given theta"""
        if theta_key in self.value_cache:
            return self.value_cache[theta_key]
            
        V = defaultdict(float)
        while True:
            delta = 0
            for i in range(self.env.size):
                for j in range(self.env.size):
                    state = np.array([i, j])
                    state_tuple = self._state_to_tuple(state)
                    v = V[state_tuple]
                    
                    values = []
                    for action in self.actions:
                        next_state = state + self.env.actions[action]
                        next_state = np.clip(next_state, 0, self.env.size - 1)
                        next_tuple = self._state_to_tuple(next_state)
                        reward = self.env.get_reward(next_state, np.array(theta_key))
                        values.append(reward + self.gamma * V[next_tuple])
                    
                    V[state_tuple] = max(values)
                    delta = max(delta, abs(v - V[state_tuple]))
            
            if delta < self.epsilon:
                break
                
        self.value_cache[theta_key] = V
        return V
    
    def compute_policy(self, V: Dict[tuple, float], theta_key: tuple) -> Dict[tuple, str]:
        """Compute optimal policy from value function"""
        policy = {}
        theta = np.array(theta_key)
        
        for i in range(self.env.size):
            for j in range(self.env.size):
                state = np.array([i, j])
                state_tuple = self._state_to_tuple(state)
                best_action = None
                best_value = float('-inf')
                
                for action in self.actions:
                    next_state = state + self.env.actions[action]
                    next_state = np.clip(next_state, 0, self.env.size - 1)
                    next_tuple = self._state_to_tuple(next_state)
                    reward = self.env.get_reward(next_state, theta)
                    value = reward + self.gamma * V[next_tuple]
                    
                    if value > best_value:
                        best_value = value
                        best_action = action
                        
                policy[state_tuple] = best_action
                
        return policy
    
    def get_action(self, state: np.ndarray, theta: np.ndarray) -> str:
        """Get optimal action for current state under given theta"""
        theta_key = tuple(theta.flatten())
        policy = self._get_cached_policy(theta_key)
        return policy[self._state_to_tuple(state)]
        
    def clear_cache(self):
        """Clear cached values and policies"""
        self.value_cache.clear()
        self.policy_cache.clear()