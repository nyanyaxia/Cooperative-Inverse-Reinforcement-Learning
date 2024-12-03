from gridworld import GridWorld
import numpy as np
from typing import List, Tuple
from valueiterationplanner import ValueIterationPlanner

class HumanTeacher:
    def __init__(self, env: GridWorld, theta: np.ndarray, policy_type: str = 'expert'):
        self.env = env
        self.theta = theta
        self.policy_type = policy_type
        self.planner = ValueIterationPlanner(env)
        
    def demonstrate(self, horizon: int) -> List[Tuple[np.ndarray, str]]:
        """Generate demonstration trajectory"""
        trajectory = []
        state = self.env.reset()
        
        for _ in range(horizon):
            if self.policy_type == 'expert':
                action = self.planner.get_action(state, self.theta)
            else:  # best response
                action = self._best_response_policy(state)
            
            trajectory.append((state.copy(), action))
            state = self.env.step(action)
            
        return trajectory

    
    def _best_response_policy(self, state: np.ndarray) -> str:
        """A Remplacer !!! Potentiellement dans le fichier best_response.py si c'est gros"""
        # Simple implementation: randomly choose between expert policy and exploration
        if np.random.random() < 0.7:  # 70% follow expert policy
            return self.planner.get_action(state, self.theta)
        else:  # 30% explore
            return np.random.choice(['N', 'S', 'E', 'W'])