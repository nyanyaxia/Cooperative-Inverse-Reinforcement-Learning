import numpy as np

H = 8
n_feat = 4
size = 6
gamma = 0.9
eta = 0.1
theta = np.random.rand(n_feat)
phi_theta = np.ones(n_feat)

def generate_valid_trajectories(size, H):
    """
    Génère toutes les trajectoires possibles de longueur H sur une grille sans dépasser.
    size : taille de la grille (size x size)
    H : longueur de la trajectoire
    """
    start_pos = (size // 2, size // 2)  # Position initiale au centre
    valid_trajectories = []

    def backtrack(current_traj, current_pos):
        """Backtracking pour construire les trajectoires valides."""
        if len(current_traj) == H:  # Trajectoire complète
            valid_trajectories.append(current_traj[:])
            return

        # Mouvements possibles (haut, bas, gauche, droite)
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]

        for move in moves:
            next_pos = (current_pos[0] + move[0], current_pos[1] + move[1])
            # Vérifie que la position reste dans les limites
            if 0 <= next_pos[0] < size and 0 <= next_pos[1] < size:
                current_traj.append(next_pos)  # Ajoute le mouvement
                backtrack(current_traj, next_pos)  # Continue la construction
                current_traj.pop()  # Annule le dernier mouvement pour explorer d'autres options

    # Démarre à partir de la position initiale
    backtrack([start_pos], start_pos)

    return np.array(valid_trajectories)


def eval_BR(tau, theta, phi, eta, H, gamma, phi_theta, n_feat, sigmas, centers):
    """theta : vecteur de préférences (tailleN-feat)
    phi : fonction vecteur de features (tailleN-feat)
    eta : paramètre se trdeoff
    H : horizon
    tau : trajectoire à évaluer (suite d'états)
    phi_theta : feature count de la best policy
    """
    phi_tau = np.zeros(n_feat)
    for i in range(H):
        phi_tau += phi(tau[i], n_feat, sigmas, centers) * gamma**i
    return np.dot(theta, phi_tau) - eta * np.linalg.norm(phi_tau - phi_theta)


def BR(theta, phi, eta, H, gamma, n_feat):
    """theta : vecteur de préférences (tailleN-feat)
    phi : fonction vecteur de features (tailleN-feat)
    eta : paramètre se trdeoff
    H : horizon"""
    Poss_traj = generate_valid_trajectories(size, H)
    score_traj = np.zeros(len(Poss_traj))
    
    # Define sigmas and centers here for consistency
    sigmas = np.random.rand(n_feat)
    centers = np.random.rand(n_feat, 2)  # Adjust to match 2D state representation

    for i in range(len(Poss_traj)):
        score_traj[i] = eval_BR(Poss_traj[i], theta, phi, eta, H, gamma, phi_theta, n_feat, sigmas, centers)
    return Poss_traj[np.argmax(score_traj)]


def phi(s, n_feat, sigmas, centers): 
    """s : état
    sigmas : vecteur de taille n_feat
    centers : matrice de taille n_feat*2 (2D positions for features)
    """
    phi_s = np.zeros(n_feat)
    for i in range(n_feat):
        # Compute feature activation based on distance to 2D center
        phi_s[i] = np.exp(-np.linalg.norm(np.array(s) - centers[i])**2 / (2 * sigmas[i]**2))
    return phi_s


# Test
print(BR(theta, phi, eta, H, gamma, n_feat))
