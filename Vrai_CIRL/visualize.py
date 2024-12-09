import numpy as np
import matplotlib.pyplot as plt

####### Parameters of the problem #######
L=4# taille de la grille
N_phi=2 # nombre de features
centroids=np.array([[0,1],[1,4]]) # les cenroides de la feature function
sigmas=np.array([0.5,0.5,0.5,0.5]) # les sigmas de la feature function
m=2 # nombre de valeurs de theta possibles -1
gamma=0.9
Horizon=4

####### Définition des ensmbles######
A_H=[(0,0), (-1, 0), (1, 0), (0, -1), (0, 1)] # les actions de l'humain
A_R=[(0,0), (-1, 0), (1, 0), (0, -1), (0, 1)] # les actions du robot
X_space=[(i,j) for i in range(-1,L+1) for j in range(-1,L+1)] # les états, on crée des états fictifs pour les bords pour rendre les zones interidtes
theta_space_uni=[-1+2*i/m for i in range(m+1)] # les valeurs de theta
Theta_space=[(theta1, theta2) for theta1 in theta_space_uni for theta2 in theta_space_uni] # les valeurs de theta

n_theta=len(Theta_space)
initial_belief=[1/n_theta for i in range(n_theta)] # la distribution initiale de theta est a priori uniforme


def phi(x):
    """
    Fonction de features
    """
    return np.array([np.exp(-np.linalg.norm(np.array(x)-c)**2/sigma) for c,sigma in zip(centroids,sigmas)])


def T(x,a_H,a_R,xp):
    """
    Fonction de transition
    """
    new_x = tuple(np.array(x)+np.array(a_H)+np.array(a_R))
    return 1.0 if new_x==xp else 0.0

def R(x,theta):
    """
    Fonction de récompense
    """
    theta1=np.array(theta)
    if x[0]==-1 or x[0]==L or x[1]==-1 or x[1]==L:
        return -1000
    else:
        return np.dot(phi(x),theta1)
         

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