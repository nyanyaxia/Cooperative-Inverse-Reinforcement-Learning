from gridworld import GridWorld
from maxent_wrapper import Trajectory, Optimizer, Initializer
from maxent_utils.maxent import irl
import numpy as np
from typing import List, Tuple
from valueiterationplanner import ValueIterationPlanner

class RobotLearner:
    def __init__(self, env: GridWorld):
        self.env = env
        self.estimated_theta = None
        self.planner = ValueIterationPlanner(env)
        
    def learn_from_demonstration(self, trajectory_data: List[Tuple[np.ndarray, str]]):
        """Implement Maximum Entropy IRL using the maxent package"""
        # Convert trajectory to maxent package format
        trajectory = Trajectory(trajectory_data)
        print(f'Robot is learning from trajectory: {trajectory.states_actions}')
        trajectory.grid_size = self.env.size
        
        # Setup transition probabilities
        n_states = self.env.size * self.env.size
        n_actions = 5  # N, S, E, W, NOOP
        p_transition = np.zeros((n_states, n_states, n_actions))
        
        # Fill transition probabilities
        for s in range(n_states):
            row, col = s // self.env.size, s % self.env.size
            state = np.array([row, col])
            
            for action, delta in self.env.actions.items():
                next_state = state + delta
                next_state = np.clip(next_state, 0, self.env.size - 1)
                next_s = next_state[0] * self.env.size + next_state[1]
                p_transition[s, next_s, trajectory._action_to_idx(action)] = 1.0
        
        # Setup features matrix
        features = np.zeros((n_states, self.env.n_features))
        for s in range(n_states):
            row, col = s // self.env.size, s % self.env.size
            state = np.array([row, col])
            features[s] = self.env.get_features(state)
        
        # Define terminal states
        terminal_states = []
        for center in self.env.feature_centers:
            center_idx = int(center[0]) * self.env.size + int(center[1])
            terminal_states.append(center_idx)
        
        # Setup optimization
        optim = Optimizer(learning_rate=0.01)
        init = Initializer(low=-1, high=1)
        
        # Run MaxEnt IRL
        print(f'Running IRL')
        self.estimated_theta = irl(
            p_transition=p_transition,
            features=features,
            terminal=terminal_states,
            trajectories=[trajectory],
            optim=optim,
            init=init,
            eps=1e-4
        )
        print(f'Running finished. Estimated theta: {self.estimated_theta}')
    
    def _policy(self, state: np.ndarray) -> str:
        """Robot's policy based on learned rewards"""
        if self.estimated_theta is None:
            return np.random.choice(['N', 'S', 'E', 'W'])
            
        return self.planner.get_action(state, self.estimated_theta)