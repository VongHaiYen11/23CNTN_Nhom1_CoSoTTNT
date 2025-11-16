import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add project root directory to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.problem.discrete.knapsack import TEST_CASES


def plot_knapsack_items(test_case_name, test_case, save_path=None):
    """
    Create a scatter plot visualizing knapsack items with weight, value, and value/weight ratio.
    
    Args:
        test_case_name: Name of the test case
        test_case: Dictionary containing 'weights', 'values', 'max_weight', 'n_items'
        save_path: Path to save the figure (optional)
    
    Returns:
        matplotlib figure object
    """
    weights = test_case['weights']
    values = test_case['values']
    max_weight = test_case['max_weight']
    n_items = test_case['n_items']
    
    # Calculate value/weight ratio for each item
    ratios = values / weights
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create scatter plot with color based on value/weight ratio
    scatter = ax.scatter(
        weights,
        values,
        c=ratios,
        cmap='plasma',  # Color map from dark purple (low ratio) to bright yellow (high ratio)
        s=100,  # Size of points
        edgecolors='black',
        linewidths=1.5,
        alpha=0.8,
        zorder=3
    )
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Value/Weight Ratio', fontsize=12, fontweight='bold')
    cbar.ax.tick_params(labelsize=10)
    
    # Set labels and title
    ax.set_xlabel('Weight', fontsize=13, fontweight='bold')
    ax.set_ylabel('Value', fontsize=13, fontweight='bold')
    ax.set_title(f'Knapsack Visualization ({n_items} items, Max Weight: {max_weight})', 
                fontsize=14, fontweight='bold', pad=15)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--', zorder=0)
    
    # Set axis limits with some padding
    weight_padding = (weights.max() - weights.min()) * 0.1 if weights.max() > weights.min() else weights.max() * 0.1
    value_padding = (values.max() - values.min()) * 0.1 if values.max() > values.min() else values.max() * 0.1
    
    ax.set_xlim(max(0, weights.min() - weight_padding), weights.max() + weight_padding)
    ax.set_ylim(max(0, values.min() - value_padding), values.max() + value_padding)
    
    # Add max weight line (vertical line)
    ax.axvline(x=max_weight, color='red', linestyle='--', linewidth=2, 
              label=f'Max Weight = {max_weight}', zorder=2, alpha=0.7)
    
    # Add legend
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
    
    # Add item indices as annotations (optional, for smaller test cases)
    if n_items <= 20:
        for i in range(n_items):
            ax.annotate(f'{i+1}', (weights[i], values[i]), 
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, alpha=0.7, zorder=4)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_all_knapsack_test_cases(save_dir='src/visualization'):
    """
    Generate scatter plots for all knapsack test cases.
    
    Args:
        save_dir: Directory to save the plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    print("=" * 60)
    print("Generating Knapsack Visualization Plots")
    print("=" * 60)
    
    for test_case_name, test_case in TEST_CASES.items():
        print(f"\nCreating plot for {test_case_name}...")
        save_path = os.path.join(save_dir, f'{test_case_name}_visualization.png')
        plot_knapsack_items(test_case_name, test_case, save_path)
        plt.close()  # Close figure to free memory
    
    print("\n" + "=" * 60)
    print("All knapsack visualizations generated successfully!")
    print("=" * 60)


if __name__ == "__main__":
    plot_all_knapsack_test_cases()

