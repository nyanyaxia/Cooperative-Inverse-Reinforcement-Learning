import numpy as np
import matplotlib.pyplot as plt
import yaml
import os
import itertools
from robot_functions import R

script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, "config.yaml")

with open(config_path, "r") as file:
    config = yaml.safe_load(file)
    
centroids = np.array(config["centroids"])
L = config["grid_size"]
m = config["num_theta_values"]
N_phi = config["num_features"]
theta_space_uni = [-1 + 2 * i / m for i in range(m + 1)]
Theta_space=list(itertools.product(theta_space_uni, repeat=N_phi))
n_theta = len(Theta_space)
initial_belief = [1 / n_theta for _ in range(n_theta)]

def visualize(X_seq, a, b, true_theta, robot=True):
    # Create reward grid
    grid = np.zeros((L, L))
    
    # Fill grid with rewards based on either estimated or true theta
    if robot:
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                cell_pos = np.array([i, j])
                theta = b.dot(Theta_space)
                grid[i, j] = R(cell_pos, theta)
    else:
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                cell_pos = np.array([i, j])
                grid[i, j] = R(cell_pos, true_theta)
    
    # Create subplot layout
    plt.figure(figsize=(12, 5))
    
    # Plot reward grid
    plt.subplot(121)
    plt.imshow(grid, cmap='gray')
    plt.title('Estimated reward' if robot else 'Ground truth reward')
    plt.colorbar()
    
    # Plot position sequence and last action
    plt.subplot(122)
    plt.imshow(grid, cmap='gray')
    plt.title('Position sequence and last action')
    
    # Convert X_seq to arrays for plotting
    positions = np.array(X_seq)
    
    # Plot historical positions as connected X's in black
    plt.plot(positions[:-1, 1], positions[:-1, 0], 'k-')  # Connect with black lines
    plt.scatter(positions[:-1, 1], positions[:-1, 0], c='k', marker='x', s=100)  # X markers
    
    # Plot last position and next position
    last_pos = positions[-1]
    next_pos = last_pos + a
    
    # Connect last two positions with colored line
    color = 'red' if robot else 'blue'
    plt.plot([last_pos[1], next_pos[1]], [last_pos[0], next_pos[0]], 
             c=color, linestyle='-')
    
    # Plot last X and next X
    plt.scatter([last_pos[1], next_pos[1]], [last_pos[0], next_pos[0]], 
                c=color, marker='x', s=100)
    
    plt.tight_layout()
    plt.show()