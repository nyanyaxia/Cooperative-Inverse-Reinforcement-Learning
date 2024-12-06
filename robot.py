from gridworld import GridWorld
from maxent_wrapper import Trajectory
from maxent_utils.maxent import irl
from maxent_wrapper import Optimizer, Initializer
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
        
        p_transition = self.env.get_p_transition()
        # Setup features matrix
        features = self.env.get_feature_matrix()
        
        # Define terminal states (none in this case)
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
            eps=1e-4,
            eps_esvf=1e-4,
        )
        print(f'Running finished. Estimated theta: {self.estimated_theta}')
    
    def _policy(self, state: np.ndarray) -> str:
        """Robot's policy based on learned rewards"""
        if self.estimated_theta is None:
            return np.random.choice(['N', 'S', 'E', 'W'])
            
        return self.planner.get_action(state, self.estimated_theta)