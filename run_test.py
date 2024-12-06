import numpy as np
import matplotlib.pyplot as plt
from gridworld import GridWorld
from human import HumanTeacher
from robot import RobotLearner

def visualize_policies(env, true_theta, expert_trajectory, robot_expert_theta, br_trajectory, robot_br_theta):
   
   plt.figure(figsize=(12, 6))
   plt.suptitle(f'Comparison of Expert and Best Response Policies for {env.n_features} features')
   
   centers = np.floor(env.feature_centers).astype(int)
    # Calculate ground truth reward
   ground_truth_reward = np.zeros((env.size, env.size))
   for i in range(env.size):
        for j in range(env.size):
            cell_pos = np.array([i, j]) 
            ground_truth_reward[j, i] = env.get_reward(cell_pos, true_theta)
           
   robot_expert_reward = np.zeros((env.size, env.size))
   for i in range(env.size):
        for j in range(env.size):
            cell_pos = np.array([i, j]) 
            robot_expert_reward[j, i] = env.get_reward(cell_pos, robot_expert_theta)
            
   robot_br_reward = np.zeros((env.size, env.size))
   for i in range(env.size):
        for j in range(env.size):
            cell_pos = np.array([i, j]) 
            robot_br_reward[j, i] = env.get_reward(cell_pos, robot_br_theta)
           
   # Ground truth reward
   plt.subplot(131)
   plt.imshow(ground_truth_reward, cmap='gray')
   plt.scatter(centers[:, 1], centers[:, 0], c='r', marker='x', s=100)
   plt.title('Ground truth reward')
   
   
   # Best response trajectory  
   plt.subplot(132)
   br_states = np.array([s for s, _ in br_trajectory])
   plt.imshow(robot_br_reward, cmap='gray')
   plt.plot(br_states[:,0], br_states[:,1], 'b-x')
   plt.title('Best Response Policy')
   
   # Expert trajectory  
   plt.subplot(133)
   expert_states = np.array([s for s, _ in expert_trajectory])
   plt.imshow(robot_expert_reward, cmap='gray')
   plt.plot(expert_states[:,0], expert_states[:,1], 'b-x')
   plt.title('Expert Policy')
   
   plt.tight_layout()
   plt.show()

def run_experiment(n_trials: int = 4, grid_size: int = 10, horizon: int = 16):
    """Run the complete experiment"""
    results = {
        'expert_3': {'l2_norms': [], 'feature_counts': []},
        'best_response_3': {'l2_norms': [], 'feature_counts': []},
        'expert_10': {'l2_norms': [], 'feature_counts': []},
        'best_response_10': {'l2_norms': [], 'feature_counts': []}
    }
    
    for _ in range(n_trials):
        print(f"Trial: {_}")
        for n_features in [3, 10]:
            # Create environment and true reward parameters
            env = GridWorld(size=grid_size, n_features=n_features)
            print('feature centers:', env.feature_centers)
            
            #We use a uniform distribution in [-1, 1] for the prior on theta 
            true_theta = np.random.uniform(-1, 1, n_features)
            
            # Test both policies
            for policy_type in ['expert', 'best_response']:
                # Create agents
                human = HumanTeacher(env, true_theta, policy_type)
                robot = RobotLearner(env)
                
                trajectory = human.demonstrate(horizon // 2)
                
                # Learning phase
                robot.learn_from_demonstration(trajectory)
                
                if policy_type == 'expert':
                    expert_trajectory, robot_expert_theta = trajectory, robot.estimated_theta
                    print("Expert trajectory:", expert_trajectory)
                else:
                    br_trajectory, robot_br_theta = trajectory, robot.estimated_theta
                    print("BR trajectory:", br_trajectory)
                    visualize_policies(env, true_theta, expert_trajectory, robot_expert_theta, br_trajectory, robot_br_theta)
                
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
    
