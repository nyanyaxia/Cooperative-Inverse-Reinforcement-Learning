import numpy as np
import matplotlib.pyplot as plt
import time
import os 

from gridworld import GridWorld
from human import HumanTeacher
from robot import RobotLearner
from maxent_utils.maxent import compute_trajectory_kl_divergence

def create_timestamped_dir(base_dir="results"):
    """
    Create a timestamped directory to store results.
    """
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    directory = os.path.join(base_dir, timestamp)
    os.makedirs(directory, exist_ok=True)
    return directory

def visualize_policies(env, true_theta, expert_trajectory, robot_expert_theta, br_trajectory, robot_br_theta, save_path, iteration):
   
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
   plt.scatter(centers[:, 0], centers[:, 1], c='r', marker='x', s=100)
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
   
   # Save plot to file
   filename = os.path.join(save_path, f"policies_iteration_{iteration}_n_features{env.n_features}.png")
   plt.savefig(filename)
   plt.close()  # Close the figure to free memory
   
def plot_comparison_metrics(results):
    """
    Create side-by-side plots comparing metrics for different numbers of features
    
    Args:
        results: Dictionary containing experiment results
        averages: Dictionary containing computed averages and standard deviations
    """
    # Set style
    plt.style.use('seaborn')
    
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Define metrics to plot
    metrics = ['Regret', 'KL', '||θGT - θ||²']
    
    # Define width of bars
    bar_width = 0.35
    
    # Define positions for bars
    indices = np.arange(len(metrics))
    
    # Colors for best response and expert
    br_color = '#ff7f50'  # coral
    expert_color = '#4682b4'  # steelblue
    
    # Plot for num_features = 3
    ax1.set_title('num-features = 3')
    # Best response bars
    br_means_3 = [
        np.mean(results['best_response_3']['regret']),
        0,  # Placeholder for KL divergence
        np.mean(results['best_response_3']['l2_norms'])
    ]
    br_stds_3 = [
        np.std(results['best_response_3']['regret']),
        0,  # Placeholder for KL divergence
        np.std(results['best_response_3']['l2_norms'])
    ]
    ax1.bar(indices - bar_width/2, br_means_3, bar_width, 
            yerr=br_stds_3, label='br', color=br_color,
            capsize=5, alpha=0.7)
    
    # Expert bars
    expert_means_3 = [
        np.mean(results['expert_3']['regret']),
        0,  # Placeholder for KL divergence
        np.mean(results['expert_3']['l2_norms'])
    ]
    expert_stds_3 = [
        np.std(results['expert_3']['regret']),
        0,  # Placeholder for KL divergence
        np.std(results['expert_3']['l2_norms'])
    ]
    ax1.bar(indices + bar_width/2, expert_means_3, bar_width,
            yerr=expert_stds_3, label='πE', color=expert_color,
            capsize=5, alpha=0.7)
    
    # Plot for num_features = 10
    ax2.set_title('num-features = 10')
    # Best response bars
    br_means_10 = [
        np.mean(results['best_response_10']['regret']),
        0,  # Placeholder for KL divergence
        np.mean(results['best_response_10']['l2_norms'])
    ]
    br_stds_10 = [
        np.std(results['best_response_10']['regret']),
        0,  # Placeholder for KL divergence
        np.std(results['best_response_10']['l2_norms'])
    ]
    ax2.bar(indices - bar_width/2, br_means_10, bar_width,
            yerr=br_stds_10, label='br', color=br_color,
            capsize=5, alpha=0.7)
    
    # Expert bars
    expert_means_10 = [
        np.mean(results['expert_10']['regret']),
        0,  # Placeholder for KL divergence
        np.mean(results['expert_10']['l2_norms'])
    ]
    expert_stds_10 = [
        np.std(results['expert_10']['regret']),
        0,  # Placeholder for KL divergence
        np.std(results['expert_10']['l2_norms'])
    ]
    ax2.bar(indices + bar_width/2, expert_means_10, bar_width,
            yerr=expert_stds_10, label='πE', color=expert_color,
            capsize=5, alpha=0.7)
    
    # Customize both plots
    for ax in [ax1, ax2]:
        ax.set_xticks(indices)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Set y-axis limits based on data
        ymin = min(min(br_means_3), min(expert_means_3), 
                  min(br_means_10), min(expert_means_10))
        ymax = max(max(br_means_3), max(expert_means_3), 
                  max(br_means_10), max(expert_means_10))
        ax.set_ylim(ymin - abs(ymin)*0.2, ymax + abs(ymax)*0.2)
    
    # Adjust layout
    plt.tight_layout()
    return fig

def run_experiment(n_trials: int = 3, grid_size: int = 10, horizon: int = 16):
    """Run the complete experiment"""
    results = {
        'expert_3': {'l2_norms': [], 'regret': []},
        'best_response_3': {'l2_norms': [], 'regret': []},
        'expert_10': {'l2_norms': [], 'regret': []},
        'best_response_10': {'l2_norms': [], 'regret': []}
    }
    
    results_dir = create_timestamped_dir()
    print(f"Results will be saved in: {results_dir}")
    
    for trial in range(n_trials):
        print(f"Trial: {trial}")
        for n_features in [3, 10]:
            # Create environment and true reward parameters
            env = GridWorld(size=grid_size, n_features=n_features)
            print('feature centers:', env.feature_centers)
            
            #We use a uniform distribution in [-1, 1] for the prior on theta 
            true_theta = np.random.uniform(-1, 1, n_features)
            if n_features == 3:
                true_theta = np.array([1, -1, 1])
            
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
                    print(f"Current trial iteration: {trial}")
                    visualize_policies(env, true_theta, expert_trajectory, robot_expert_theta, br_trajectory, robot_br_theta, results_dir, trial)
                
                # Theta sign correction 
                
                # Deployment phase
                initial_state = env.reset(random_state=True)
                reward_estimated_theta = 0
                reward_true_theta = 0
                
                state = initial_state.copy()
                for _ in range(horizon // 2):
                    action = robot._policy(state, robot.estimated_theta)
                    reward_estimated_theta += env.get_reward(state, robot.estimated_theta)
                    state = env.step(action)
                
                state = initial_state.copy()
                for _ in range(horizon // 2):
                    action = robot._policy(state, true_theta)
                    reward_true_theta += env.get_reward(state, true_theta)
                    state = env.step(action)
                    
                # Calculate measurements
                l2 = np.linalg.norm(true_theta - robot.estimated_theta)
                regret = reward_true_theta - reward_estimated_theta
                #kl = compute_trajectory_kl_divergence(p_transition, features, theta_hat, theta_gt, terminal, 
                                   #p_initial)
                
                # Store results
                key = f"{policy_type}_{n_features}"
                results[key]['l2_norms'].append(l2)
                results[key]['regret'].append(regret)
                
                
    
    # Compute averages for each policy and feature count
    averages = {}
    for key in results:
        averages[key] = {
            'avg_l2': np.mean(results[key]['l2_norms']),
            'std_l2': np.std(results[key]['l2_norms']),
            'avg_regret': np.mean(results[key]['regret'], axis=0),
            'std_regret': np.std(results[key]['regret'], axis=0)
        }
    
    return results, averages

# Example usage:
if __name__ == "__main__":
    results, averages = run_experiment()
    
    fig  = plot_comparison_metrics(results)

    # Plotting the distributions
    plt.figure(figsize=(10, 6))

    for n_features in [3, 10]:
        # Get the results for the given number of features
        br_l2 = results[f'best_response_{n_features}']['l2_norms']
        expert_l2 = results[f'expert_{n_features}']['l2_norms']
        br_regret = results[f'best_response_{n_features}']['regret']
        expert_regret = results[f'expert_{n_features}']['regret']

        
        # Plot L2 norms for the feature count
        plt.hist(br_l2, alpha=0.5, label=f"BR (L2) - {n_features} features (avg={averages[f'best_response_{n_features}']['avg_l2']:.3f})")
        plt.hist(expert_l2, alpha=0.5, label=f"Expert (L2) - {n_features} features (avg={averages[f'expert_{n_features}']['avg_l2']:.3f})")
        
        # Plot regret for the feature count
        plt.hist(br_regret, alpha=0.5, linestyle='dashed', label=f"BR (Regret) - {n_features} features (avg={averages[f'best_response_{n_features}']['avg_regret']:.3f})")
        plt.hist(expert_regret, alpha=0.5, linestyle='dashed', label=f"Expert (Regret) - {n_features} features (avg={averages[f'expert_{n_features}']['avg_regret']:.3f})")

    # Customize the plot
    plt.legend()
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Distribution of L2 Norms and Regret')
    plt.show()
    
