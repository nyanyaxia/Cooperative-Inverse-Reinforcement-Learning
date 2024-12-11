import yaml
import os
import importlib

import Solver_general
importlib.reload(Solver_general)
from Solver_general import *
from robot_functions import T, R


script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, "config.yaml")

with open(config_path, "r") as file:
    config = yaml.safe_load(file)

# Extract parameters
L = config["grid_size"]
N_phi = config["num_features"]
centroids = np.array(config["centroids"])
sigmas = np.array(config["sigmas"])
m = config["num_theta_values"]
gamma = config["gamma"]
Horizon = config["horizon"]

A_H = [tuple(a) for a in config["human_actions"]]
A_R = [tuple(a) for a in config["robot_actions"]]
X_space = [(i, j) for i in range(-1, L + 1) for j in range(-1, L + 1)]
theta_space_uni = [-1 + 2 * i / m for i in range(m + 1)]
Theta_space = [
    (theta1, theta2) for theta1 in theta_space_uni for theta2 in theta_space_uni
]

n_theta = len(Theta_space)
initial_belief = [1 / n_theta for _ in range(n_theta)]
         

P0={((L//2,L//2),Theta_space[5]):1.0}

######## Run du jeu ########
print(f'theta_space={Theta_space}')
print(f'initial_belief={initial_belief}, len(initial_belief)={len(initial_belief)}')
game=CIRLGame(X_space,Theta_space,A_H,A_R,T,R,gamma,P0,Horizon,initial_belief,verbose=True)
alpha_vectors=game.solve()
X_seq, A_H_seq, A_R_seq, rewards = game.forward_simulation(alpha_vectors)
game.print("\nForward Simulation Result:")
game.print("X_seq:",X_seq)
game.print("A_R_seq:",A_R_seq)
game.print("A_H_seq:",A_H_seq)
game.print("Rewards:", rewards)

# Plot the obtained rewards
plt.figure()
plt.plot(range(len(rewards)), rewards, marker='o')
plt.title("Reward per Step in Forward Simulation")
plt.xlabel("Step")
plt.ylabel("Reward")
plt.grid(True)