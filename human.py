from gridworld import GridWorld
import numpy as np
from typing import List, Tuple
from valueiterationplanner import ValueIterationPlanner
import time

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
        
        
        if self.policy_type == 'expert':
            for _ in range(horizon):
                
                action = self.planner.get_action(state, self.theta)
                
                
                trajectory.append((state.copy(), action))
                state = self.env.step(action)
        else :
            trajectory = self._best_response_policy(horizon, eta=0.5)
        return trajectory
    

    ##### Fonctions intermédiaires #####
    def policy_to_trajectory(
        self,
        env: GridWorld,
        planner: ValueIterationPlanner,
        theta: np.ndarray,
        start_state: np.ndarray,
        H: int
    ) -> List[Tuple[np.ndarray, str]]:
        """
        Transform a policy into a trajectory.

        env : an instance of GridWorld
        planner : an instance of ValueIterationPlanner
        theta : preference vector (parameters for rewards)
        start_state : initial state (grid position)
        H : horizon (maximum trajectory length)

        Returns:
            A list of tuples where each tuple contains:
            - The position reached (np.ndarray)
            - The action chosen to get there (str)
        """
        trajectory = []
        current_state = start_state.copy()

        for _ in range(H):
            action = planner.get_action(current_state, theta)  # Get the optimal action
            next_state = current_state + env.actions[action]  # Compute the next state
            next_state = np.clip(next_state, 0, env.size - 1)  # Keep within grid limits
            trajectory.append((current_state.copy(), action))  # Add the state and action to the trajectory
            current_state = next_state  # Update the current state

        return trajectory

    
    def feature_count_from_trajectory(self, trajectory: List[Tuple[np.ndarray, str]]) -> np.ndarray:
        """Compute feature count from trajectory"""
        phi_tau = np.zeros(self.env.n_features)
        gamma = 0.95  # Discount factor; set to a default value (could be a parameter)
        for i, (state, _) in enumerate(trajectory):
            phi_tau += self.env.get_features(state) * (gamma ** i)
        return phi_tau
    

    def generate_valid_trajectories(self , H: int) -> List[List[Tuple[np.ndarray, str]]]:
        """
        Génère toutes les trajectoires possibles de longueur H sur la grille d'un GridWorld.

        gridworld : une instance de la classe GridWorld
        H : longueur de la trajectoire
        """
        gridworld= self.env
        size = gridworld.size  # Taille de la grille
        start_pos = gridworld.state  # Position initiale
        actions = gridworld.actions  # Actions possibles (N, S, E, W, NOOP)
        valid_trajectories = []

        def backtrack(current_traj: List[Tuple[np.ndarray, str]], current_pos: np.ndarray):
            """Backtracking pour construire les trajectoires valides."""
            if len(current_traj) == H:  # Si la trajectoire atteint la longueur cible
                valid_trajectories.append(current_traj[:])
                return

            for action, move in actions.items():
                next_pos = current_pos + move
                # Vérifie que la position reste dans les limites
                if np.all(0 <= next_pos) and np.all(next_pos < size):
                    current_traj.append((current_pos.copy(), action))  # Ajoute le mouvement et l'action
                    backtrack(current_traj, next_pos)  # Continue la construction
                    current_traj.pop()  # Annule le dernier mouvement pour explorer d'autres options

        # Démarre à partir de la position initiale
        backtrack([], start_pos)

        return valid_trajectories
    
    def eval_BR(self, tau: List[Tuple[np.ndarray, str]], eta: float, phi_theta: np.ndarray) -> float:
        """fonction coût qui sert à trouver la meilleure réponse"""
        phi_tau = self.feature_count_from_trajectory(tau)
        return np.dot(self.theta, phi_tau) - eta * np.linalg.norm(phi_tau - phi_theta)
    
    def compute_phi_theta(self, H: int) -> np.ndarray:
        """Compute feature count of the best policy"""
        # Génère une clé unique à partir de theta
        theta_key = tuple(self.theta.flatten())
        
        # Calcule la politique optimale en passant la clé theta_key
        best_policy = self.planner.compute_policy(self.planner.compute_values(theta_key), theta_key)
        
        # Génère une trajectoire à partir de la politique optimale
        best_trajectory = self.policy_to_trajectory(self.env, self.planner, self.theta, self.env.state, H)
        
        # Calcule et retourne le feature count de la trajectoire optimale
        return self.feature_count_from_trajectory(best_trajectory)

    ### best response ####

    def _best_response_policy(self, H: int,eta: float = 0.5) -> List[Tuple[np.ndarray, str]]:
        """Calcul de la meilleure réponse"""
        time_start = time.time()
        phi_theta = self.compute_phi_theta(H)
        print(f'Computing phi_theta took {time.time() - time_start} seconds')
        
        time_start = time.time()
        valid_trajectories = self.generate_valid_trajectories(H)
        print(f'Generating valid trajectories took {time.time() - time_start} seconds')
        print(f'Number of valid trajectories: {len(valid_trajectories)}')
        
        best_response = None
        best_value = -np.inf
        
        count = 0
        time_start = time.time()
        for tau in valid_trajectories:
            
            value = self.eval_BR(tau, eta, phi_theta)
            
            count += 1
            if count % 100000 == 0:
                print('Eval BR count:', count*100/len(valid_trajectories), '%')

            if value > best_value:
                best_value = value
                best_response = tau
        
        print(f'Eval BR for {len(valid_trajectories)} trajectories took {time.time() - time_start} seconds')
        return best_response

    
