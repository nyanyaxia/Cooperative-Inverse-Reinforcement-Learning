import matplotlib.pyplot as plt
import numpy as np
import time
import os

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
   
def plot_comparison_metrics(results, save_path):
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
    
    # Helper function for SEM calculation
    def compute_sem(data):
        return np.std(data) / np.sqrt(len(data))
    
    # Plot for num_features = 3
    ax1.set_title('num-features = 3')
    # Best response bars
    br_means_3 = [
        np.mean(results['best_response_3']['regret']),
        np.mean(results['best_response_3']['kl_divergence']),
        np.mean(results['best_response_3']['l2_norms'])
    ]
    br_sems_3 = [
        compute_sem(results['best_response_3']['regret']),
        compute_sem(results['best_response_3']['kl_divergence']),
        compute_sem(results['best_response_3']['l2_norms'])
    ]
    ax1.bar(indices - bar_width/2, br_means_3, bar_width,
            yerr=br_sems_3, label='br', color=br_color,
            capsize=5, alpha=0.7)
    
    # Expert bars
    expert_means_3 = [
        np.mean(results['expert_3']['regret']),
        np.mean(results['expert_3']['kl_divergence']),
        np.mean(results['expert_3']['l2_norms'])
    ]
    expert_sems_3 = [
        compute_sem(results['expert_3']['regret']),
        compute_sem(results['expert_3']['kl_divergence']),
        compute_sem(results['expert_3']['l2_norms'])
    ]
    ax1.bar(indices + bar_width/2, expert_means_3, bar_width,
            yerr=expert_sems_3, label='πE', color=expert_color,
            capsize=5, alpha=0.7)
    
    # Plot for num_features = 10
    ax2.set_title('num-features = 10')
    # Best response bars
    br_means_10 = [
        np.mean(results['best_response_10']['regret']),
        np.mean(results['best_response_10']['kl_divergence']),
        np.mean(results['best_response_10']['l2_norms'])
    ]
    br_sems_10 = [
        compute_sem(results['best_response_10']['regret']),
        compute_sem(results['best_response_10']['kl_divergence']),
        compute_sem(results['best_response_10']['l2_norms'])
    ]
    ax2.bar(indices - bar_width/2, br_means_10, bar_width,
            yerr=br_sems_10, label='br', color=br_color,
            capsize=5, alpha=0.7)
    
    # Expert bars
    expert_means_10 = [
        np.mean(results['expert_10']['regret']),
        np.mean(results['expert_10']['kl_divergence']),
        np.mean(results['expert_10']['l2_norms'])
    ]
    expert_sems_10 = [
        compute_sem(results['expert_10']['regret']),
        compute_sem(results['expert_10']['kl_divergence']),
        compute_sem(results['expert_10']['l2_norms'])
    ]
    ax2.bar(indices + bar_width/2, expert_means_10, bar_width,
            yerr=expert_sems_10, label='πE', color=expert_color,
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
    
    filename = os.path.join(save_path, f'comparison_metrics.png')
    plt.savefig(filename)
    return fig