from gridworld import GridWorld
import numpy as np
from typing import List, Tuple

class HumanTeacher:
    def __init__(self, env: GridWorld, theta: np.ndarray, policy_type: str = 'expert'):
        self.env = env
        self.theta = theta
        self.policy_type = policy_type
        
    def demonstrate(self, horizon: int) -> List[Tuple[np.ndarray, str]]:
        """Generate demonstration trajectory"""
        trajectory = []
        state = self.env.reset()
        
        for _ in range(horizon):
            if self.policy_type == 'expert':
                action = self._expert_policy(state)
            else:  # best response
                action = self._best_response_policy(state)
            
            trajectory.append((state.copy(), action))
            state = self.env.step(action)
            
        return trajectory
    
    def _expert_policy(self, state: np.ndarray) -> str:
        """Expert policy that goes directly to highest reward"""
        best_action = 'NOOP'
        best_value = float('-inf')
        
        for action in ['N', 'S', 'E', 'W']:
            next_state = state + self.env.actions[action]
            next_state = np.clip(next_state, 0, self.env.size - 1)
            value = self.env.get_reward(next_state, self.theta)
            
            if value > best_value:
                best_value = value
                best_action = action
                
        return best_action
    
    def _best_response_policy(self, state: np.ndarray) -> str:
        """A Remplacer !!! Potentiellement dans le fichier best_response.py si c'est gros"""
        # Simple implementation: randomly choose between expert policy and exploration
        if np.random.random() < 0.7:  # 70% follow expert policy
            return self._expert_policy(state)
        else:  # 30% explore
            return np.random.choice(['N', 'S', 'E', 'W'])