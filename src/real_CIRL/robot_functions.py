import numpy as np
import yaml 
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, "config.yaml")

with open(config_path, "r") as file:
    config = yaml.safe_load(file)
    
centroids = np.array(config["centroids"])
sigmas = np.array(config["sigmas"])
L = config["grid_size"]

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