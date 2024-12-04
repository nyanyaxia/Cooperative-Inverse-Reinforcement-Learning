import numpy as np
import matplotlib.pyplot as plt
from gridworld import GridWorld
from human import HumanTeacher
from robot import RobotLearner


def run_experiment(n_trials: int = 2, grid_size: int = 10, horizon: int = 6):
    """Run the complete experiment"""
    results = {
        'expert_3': {'l2_norms': [], 'feature_counts': []},
        'best_response_3': {'l2_norms': [], 'feature_counts': []},
        'expert_10': {'l2_norms': [], 'feature_counts': []},
        'best_response_10': {'l2_norms': [], 'feature_counts': []}
    }
    
    for _ in range(n_trials):
        if (_ % 50 == 0):
            print(f"Trial: {_}")
        for n_features in [3, 10]:
            # Create environment and true reward parameters
            env = GridWorld(size=grid_size, n_features=n_features)
            
            #We use a uniform distribution in [-1, 1] for the prior on theta 
            true_theta = np.random.uniform(-1, 1, n_features)
            
            # Test both policies
            for policy_type in ['expert', 'best_response']:
                # Create agents
                human = HumanTeacher(env, true_theta, policy_type)
                robot = RobotLearner(env)
                
                # Learning phase
                trajectory = human.demonstrate(horizon // 2)
                robot.learn_from_demonstration(trajectory)
                
                # Store L2 norm
                l2 = np.linalg.norm(true_theta - robot.estimated_theta)
                
                # Compute and store feature counts
                feature_counts = human.feature_count_from_trajectory(trajectory)
                
                # Store results
                key = f"{policy_type}_{n_features}"
                results[key]['l2_norms'].append(l2)
                results[key]['feature_counts'].append(feature_counts)
                
                # Deployment phase
                state = env.reset(random_state=True)
                
                for _ in range(horizon // 2):
                    action = robot._policy(state)
                    state = env.step(action)
    
    # Compute averages for each policy and feature count
    averages = {}
    for key in results:
        averages[key] = {
            'avg_l2': np.mean(results[key]['l2_norms']),
            'std_l2': np.std(results[key]['l2_norms']),
            'avg_features': np.mean(results[key]['feature_counts'], axis=0),
            'std_features': np.std(results[key]['feature_counts'], axis=0)
        }
    
    return results, averages

# Example usage:
if __name__ == "__main__":
    results, averages = run_experiment()
    
    # Plot results
    plt.figure(figsize=(10, 6))
    for key in results:
        plt.hist(results[key]['l2_norms'], alpha=0.5, label=f"{key} (avg={averages[key]['avg_l2']:.3f})")
    plt.legend()
    plt.xlabel('L2 Norm')
    plt.ylabel('Frequency')
    plt.title('Distribution of L2 Norms')
    plt.show()