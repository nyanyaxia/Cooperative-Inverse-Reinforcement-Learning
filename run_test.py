import numpy as np
import matplotlib.pyplot as plt
from gridworld import GridWorld
from human import HumanTeacher
from robot import RobotLearner


def run_experiment(n_trials: int = 500, grid_size: int = 10, horizon: int = 20):
    """Run the complete experiment"""
    results = {
        'expert_3': [],
        'best_response_3': [],
        'expert_10': [],
        'best_response_10': []
    }
    
    for _ in range(n_trials):
        if (_ % 50 == 0 ):
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
                
                # Deployment phase
                total_reward = 0
                state = env.reset(random_state=True)
                
                for _ in range(horizon // 2):
                    action = robot._policy(state)
                    state = env.step(action)
                    total_reward += env.get_reward(state, true_theta)
                
                # Store results
                key = f"{policy_type}_{n_features}"
                results[key].append(total_reward)
    
    return results

# Example usage:
if __name__ == "__main__":
    results = run_experiment()
    
    # Plot results
    plt.figure(figsize=(10, 6))
    for key in results:
        plt.hist(results[key], alpha=0.5, label=key)
    plt.legend()
    plt.xlabel('Total Reward')
    plt.ylabel('Frequency')
    plt.title('Comparison of Expert vs Best Response Policies')
    plt.show()