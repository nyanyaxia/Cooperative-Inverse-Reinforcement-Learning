from typing import List, Tuple, Dict 
import numpy as np 
from dataclasses import dataclass

@dataclass 
class Trajectory:
    """Wrapper class to make trajectories compatible with maxent package"""
    states_actions: List[Tuple[np.ndarray, str]]
    
    def states(self):
        """Return list of state indices"""
        state_indices = [self._state_to_idx(s) for s, _ in self.states_actions]
        return state_indices
    
    def transitions(self):
        """Return list of (state, next_state, action) indices"""
        transitions = []
        for i in range(len(self.states_actions) - 1):
            curr_state, action = self.states_actions[i]
            next_state, _ = self.states_actions[i + 1]
            transition = (
                self._state_to_idx(curr_state),
                self._state_to_idx(next_state),
                self._action_to_idx(action)
            )
            transitions.append(transition)
        return transitions
    
    def _state_to_idx(self, state: np.ndarray) -> int:
        """Convert 2D state to flat index"""
        idx = state[0] * self.grid_size + state[1]
        if not hasattr(self, '_printed_size'):
            self._printed_size = True
        return idx
    
    def _action_to_idx(self, action: str) -> int:
        """Convert action string to index"""
        action_map = {'N': 0, 'S': 1, 'E': 2, 'W': 3, 'NOOP': 4}
        return action_map[action]

class Optimizer:
    """Simple gradient descent optimizer for maxent package"""
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.parameters = None
        
    def reset(self, parameters):
        self.parameters = parameters.copy()  # Make copy to avoid reference issues
        
    def step(self, gradient):
        self.parameters += self.learning_rate * gradient

class Initializer:
    """Uniform random initializer for maxent package"""
    def __init__(self, low=-1, high=1):
        self.low = low
        self.high = high
    
    def __call__(self, n_features):
        params = np.random.uniform(self.low, self.high, n_features)
        return params