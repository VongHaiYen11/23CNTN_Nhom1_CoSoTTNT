import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import time
import importlib.util

# Add project root directory to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# Import experiment modules
spec_sphere = importlib.util.spec_from_file_location("run_sphere", 
    os.path.join(project_root, "src/experiment/run_sphere.py"))
run_sphere = importlib.util.module_from_spec(spec_sphere)
spec_sphere.loader.exec_module(run_sphere)

spec_knapsack = importlib.util.spec_from_file_location("run_knapsack", 
    os.path.join(project_root, "src/experiment/run_knapsack.py"))
run_knapsack = importlib.util.module_from_spec(spec_knapsack)
spec_knapsack.loader.exec_module(run_knapsack)

from src.problem.continuous.sphere import sphere

# Color palette for different algorithms
COLORS = {
    'FA': '#FF6B6B',
    'ABC': '#4ECDC4',
    'Cuckoo': '#45B7D1',
    'PSO': '#FFA07A',
    'HC': '#98D8C8',
    'GA': '#F7DC6F',
    'SA': '#BB8FCE',
    'ACO': '#85C1E2'
}

# Marker styles for different algorithms
MARKERS = {
    'FA': 'o',
    'ABC': 's',
    'Cuckoo': '^',
    'PSO': 'D',
    'HC': 'v',
    'GA': 'p',
    'SA': '*',
    'ACO': 'X'
}


def run_sphere_experiments():
    """Run sphere experiments using run_sphere.py functions"""
    algorithms = run_sphere.ALGOS
    results = {}
    
    print("Running Sphere Function Experiments...")
    print("=" * 60)
    
    for algo in algorithms:
        print(f"Running {algo}...")
        try:
            res = run_sphere.run_algorithm(
                algo, 
                run_sphere.DIMS[0], 
                run_sphere.POP_SIZES[0], 
                run_sphere.SEED
            )
            # Get history by running algorithm again with verbose=False
            algorithm = run_sphere.create_algorithm(
                algo,
                run_sphere.DIMS[0],
                run_sphere.POP_SIZES[0],
                run_sphere.SEED
            )
            algorithm.verbose = False
            best_sol, best_fit, history = algorithm.run()
            
            results[algo] = {
                'best_solution': best_sol,
                'best_fitness': best_fit,
                'history': history,
                'runtime': res['elapsed'],
                'memory': res['space']
            }
            print(f"  Best Fitness: {results[algo]['best_fitness']:.6e}")
            print(f"  Runtime: {results[algo]['runtime']:.4f}s")
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    print("=" * 60)
    return results


def run_knapsack_experiments():
    """Run knapsack experiments using run_knapsack.py functions"""
    algorithms = run_knapsack.ALGOS
    results = {}
    
    print("\nRunning Knapsack Problem Experiments...")
    print("=" * 60)
    
    for algo in algorithms:
        print(f"Running {algo}...")
        try:
            res = run_knapsack.run_algorithm(
                algo,
                run_knapsack.POP_SIZES[0],
                run_knapsack.SEED
            )
            # Get history by running algorithm again with verbose=False
            algorithm = run_knapsack.create_algorithm(
                algo,
                run_knapsack.POP_SIZES[0],
                run_knapsack.SEED
            )
            algorithm.verbose = False
            solution, fitness, history = algorithm.run()
            
            results[algo] = {
                'best_solution': solution,
                'best_fitness': fitness,
                'history': history,
                'runtime': res['elapsed'],
                'memory': res['space']
            }
            print(f"  Best Fitness: {results[algo]['best_fitness']:.2f}")
            print(f"  Runtime: {results[algo]['runtime']:.4f}s")
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    print("=" * 60)
    return results


def plot_convergence_individual(results_dict, problem_type='sphere', save_dir='src/visualization'):
    """Plot convergence history for each algorithm individually"""
    os.makedirs(save_dir, exist_ok=True)
    
    for algo_name, result in results_dict.items():
        fig, ax = plt.subplots(figsize=(10, 6))
        
        history = result['history']
        iterations = np.arange(len(history))
        color = COLORS.get(algo_name, '#95A5A6')
        
        ax.plot(iterations, history, color=color, linewidth=2.5, alpha=0.8, label=algo_name)
        ax.set_xlabel('Iteration', fontsize=12, fontweight='bold')
        ax.set_ylabel('Best Fitness', fontsize=12, fontweight='bold')
        ax.set_title(f'{algo_name} Convergence ({problem_type.capitalize()} Problem)', 
                     fontsize=14, fontweight='bold')
        
        # Use linear scale with proper tick marks
        max_fitness = max(history)
        min_fitness = min(history)
        y_padding = (max_fitness - min_fitness) * 0.1
        y_max = max_fitness + y_padding
        y_min = max(0, min_fitness - y_padding)
        
        # Set y-axis limits and ticks
        ax.set_ylim(y_min, y_max)
        
        if problem_type == 'knapsack':
            # Less detailed ticks for knapsack (larger steps)
            step = max(10, int((y_max - y_min) / 5))
            y_ticks = np.arange(0, int(y_max) + step, step)
        else:
            # More detailed ticks for sphere (smaller steps)
            step = max(1, int((y_max - y_min) / 10))
            y_ticks = np.arange(0, int(y_max) + step, step)
        
        ax.set_yticks(y_ticks)
        
        ax.legend(loc='upper right', fontsize=11)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(left=0)
        
        plt.tight_layout()
        filename = f'{algo_name.lower()}_{problem_type}_convergence.png'
        plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filename}")


def plot_convergence_combined(results_dict, problem_type='sphere', save_path=None):
    """Plot convergence history for all algorithms in one graph (linear scale like the image)"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    for algo_name, result in results_dict.items():
        history = result['history']
        iterations = np.arange(len(history))
        color = COLORS.get(algo_name, '#95A5A6')
        
        ax.plot(iterations, history, label=algo_name, color=color, 
                linewidth=2.5, alpha=0.8)
    
    ax.set_xlabel('Iteration', fontsize=13, fontweight='bold')
    ax.set_ylabel('Objective Function Value', fontsize=13, fontweight='bold')
    ax.set_title(f'Algorithm Convergence Comparison ({problem_type.capitalize()} Problem)', 
                 fontsize=15, fontweight='bold')
    
    # Use linear scale (not log) and set appropriate y-axis limits with integer ticks
    max_fitness = max([max(result['history']) for result in results_dict.values()])
    min_fitness = min([min(result['history']) for result in results_dict.values()])
    
    # Set y-axis limits with some padding
    y_padding = (max_fitness - min_fitness) * 0.1
    y_max = max_fitness + y_padding
    y_min = max(0, min_fitness - y_padding)
    
    ax.set_ylim(y_min, y_max)
    
    if problem_type == 'knapsack':
        # Less detailed ticks for knapsack (larger steps)
        step = max(10, int((y_max - y_min) / 5))
        y_ticks = np.arange(0, int(y_max) + step, step)
    else:
        # More detailed ticks for sphere (smaller steps)
        step = max(1, int((y_max - y_min) / 10))
        y_ticks = np.arange(0, int(y_max) + step, step)
    
    ax.set_yticks(y_ticks)
    
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9, ncol=2)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(left=0)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    return fig


def plot_runtime_bar(results_dict, problem_type='sphere', save_path=None):
    """Plot runtime comparison as bar chart with better broken axis format"""
    algorithms = list(results_dict.keys())
    runtimes = [results_dict[algo]['runtime'] for algo in algorithms]
    colors = [COLORS.get(algo, '#95A5A6') for algo in algorithms]
    
    # Check if we need broken axis
    min_runtime = min(runtimes)
    max_runtime = max(runtimes)
    ratio = max_runtime / min_runtime if min_runtime > 0 else 1
    
    if ratio > 10:  # Use broken axis if ratio > 10
        fig = plt.figure(figsize=(12, 8))
        gs = fig.add_gridspec(3, 1, height_ratios=[1, 0.15, 2], hspace=0.08)
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[2])
        
        # Determine break point
        break_point = min_runtime * 1.15
        
        # Top subplot (small values)
        bars1 = ax1.bar(algorithms, runtimes, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax1.set_ylabel('Runtime (seconds)', fontsize=12, fontweight='bold')
        ax1.set_title('Algorithm Runtime Comparison', fontsize=14, fontweight='bold', pad=15)
        ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax1.set_ylim(0, break_point)
        
        # Bottom subplot (large values)
        bars2 = ax2.bar(algorithms, runtimes, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax2.set_xlabel('Algorithm', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Runtime (seconds)', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax2.set_ylim(max_runtime * 0.85, max_runtime * 1.1)
        
        # Add value labels
        for i, bar in enumerate(bars1):
            height = bar.get_height()
            if height < break_point:
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        for i, bar in enumerate(bars2):
            height = bar.get_height()
            if height >= break_point:
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Hide the spines between the subplots
        ax1.spines['bottom'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax1.xaxis.tick_top()
        ax1.tick_params(labeltop=False, top=False, labelsize=10)
        ax2.xaxis.tick_bottom()
        ax2.tick_params(labelsize=10)
        
        # Add diagonal break lines
        d = 0.02  # Size of diagonal lines
        kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False, linewidth=1.5)
        ax1.plot((-d, +d), (-d, +d), **kwargs)
        ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)
        
        kwargs.update(transform=ax2.transAxes)
        ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)
        ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
    else:
        fig, ax = plt.subplots(figsize=(12, 7))
        bars = ax.bar(algorithms, runtimes, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax.set_xlabel('Algorithm', fontsize=12, fontweight='bold')
        ax.set_ylabel('Runtime (seconds)', fontsize=12, fontweight='bold')
        ax.set_title('Algorithm Runtime Comparison', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    return fig


def plot_memory_bar(results_dict, problem_type='sphere', save_path=None):
    """Plot memory usage comparison as bar chart with better broken axis format"""
    algorithms = list(results_dict.keys())
    memories = [results_dict[algo]['memory'] / 1024 for algo in algorithms]  # Convert to KB
    colors = [COLORS.get(algo, '#95A5A6') for algo in algorithms]
    
    # Check if we need broken axis
    min_memory = min(memories)
    max_memory = max(memories)
    ratio = max_memory / min_memory if min_memory > 0 else 1
    
    if ratio > 10:  # Use broken axis if ratio > 10
        fig = plt.figure(figsize=(12, 8))
        gs = fig.add_gridspec(3, 1, height_ratios=[1, 0.15, 2], hspace=0.08)
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[2])
        
        # Determine break point
        break_point = min_memory * 1.15
        
        # Top subplot (small values)
        bars1 = ax1.bar(algorithms, memories, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax1.set_ylabel('Memory Usage (KB)', fontsize=12, fontweight='bold')
        ax1.set_title('Algorithm Memory Usage Comparison', fontsize=14, fontweight='bold', pad=15)
        ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax1.set_ylim(0, break_point)
        
        # Bottom subplot (large values)
        bars2 = ax2.bar(algorithms, memories, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax2.set_xlabel('Algorithm', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Memory Usage (KB)', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax2.set_ylim(max_memory * 0.85, max_memory * 1.1)
        
        # Add value labels
        for i, bar in enumerate(bars1):
            height = bar.get_height()
            if height < break_point:
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        for i, bar in enumerate(bars2):
            height = bar.get_height()
            if height >= break_point:
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Hide the spines between the subplots
        ax1.spines['bottom'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax1.xaxis.tick_top()
        ax1.tick_params(labeltop=False, top=False, labelsize=10)
        ax2.xaxis.tick_bottom()
        ax2.tick_params(labelsize=10)
        
        # Add diagonal break lines
        d = 0.02  # Size of diagonal lines
        kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False, linewidth=1.5)
        ax1.plot((-d, +d), (-d, +d), **kwargs)
        ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)
        
        kwargs.update(transform=ax2.transAxes)
        ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)
        ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
    else:
        fig, ax = plt.subplots(figsize=(12, 7))
        bars = ax.bar(algorithms, memories, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax.set_xlabel('Algorithm', fontsize=12, fontweight='bold')
        ax.set_ylabel('Memory Usage (KB)', fontsize=12, fontweight='bold')
        ax.set_title('Algorithm Memory Usage Comparison', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    return fig


def plot_sphere_with_algorithms(results_dict, save_path=None):
    """Plot sphere function with algorithm best fitness points (beautiful version with fixed limits)"""
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Plot sphere function f(x) = x^2
    x = np.linspace(-5, 5, 1000)
    y = x ** 2
    ax.plot(x, y, 'b-', linewidth=3, label='Sphere Function f(x) = x²', alpha=0.7, zorder=1)
    
    # Plot optimal fitness line at x=0
    ax.axvline(x=0, color='r', linestyle='--', linewidth=2.5, label='Optimal Fitness = 0', alpha=0.7, zorder=2)
    
    # Collect all algorithm points
    algo_points = []
    sa_point = None
    for algo_name, result in results_dict.items():
        best_sol = result['best_solution']
        best_fit = result['best_fitness']
        
        # For multi-dimensional sphere, use the norm of the solution as x-coordinate
        # But preserve the sign if possible (use first dimension for sign)
        if len(best_sol.shape) > 0:
            if len(best_sol) > 1:
                # Use first dimension to determine sign, norm for magnitude
                sign = 1 if best_sol[0] >= 0 else -1
                x_coord = sign * np.linalg.norm(best_sol)
            else:
                x_coord = best_sol[0]
        else:
            x_coord = best_sol
        
        # Clip coordinates to fit within the plot limits
        x_coord = np.clip(x_coord, -5, 5)
        best_fit = np.clip(best_fit, 0, 15)
        
        if algo_name == 'SA':
            sa_point = (x_coord, best_fit, algo_name)
        else:
            algo_points.append((x_coord, best_fit, algo_name))
    
    # Plot algorithm best fitness points with large, visible markers
    for x_coord, best_fit, algo_name in algo_points:
        color = COLORS.get(algo_name, '#95A5A6')
        marker = MARKERS.get(algo_name, 'o')
        
        ax.scatter(x_coord, best_fit, color=color, marker=marker, s=400, 
                  label=algo_name, edgecolors='black', linewidths=2.5, zorder=5, alpha=0.9)
    
    # Creative visualization for SA (Simulated Annealing)
    # Show a cooling path/trajectory from high temperature to low
    if sa_point:
        x_coord, best_fit, algo_name = sa_point
        color = COLORS.get('SA', '#BB8FCE')
        
        # Create a cooling path visualization - show the annealing process
        # Start from a higher point (simulating high temperature exploration)
        start_x = np.clip(x_coord + 1.5 * (1 if x_coord >= 0 else -1), -4.5, 4.5)
        start_y = min(best_fit + 8, 14)
        
        # Create a curved path showing the cooling/annealing process
        path_x = np.linspace(start_x, x_coord, 50)
        # Exponential cooling curve
        cooling_factor = np.exp(-np.linspace(0, 3, 50))
        path_y = start_y - (start_y - best_fit) * (1 - cooling_factor)
        
        # Plot the cooling path with gradient colors
        for i in range(len(path_x) - 1):
            alpha_val = 0.3 + 0.4 * (1 - i / len(path_x))
            ax.plot(path_x[i:i+2], path_y[i:i+2], color=color, 
                   linewidth=2, alpha=alpha_val, zorder=3)
        
        # Add temperature indicator circles along the path
        temp_indices = [0, len(path_x)//3, 2*len(path_x)//3, len(path_x)-1]
        temp_labels = ['High T', 'Medium T', 'Low T', 'Final']
        for idx, label in zip(temp_indices, temp_labels):
            if idx < len(path_x):
                ax.scatter(path_x[idx], path_y[idx], color=color, s=150, 
                          edgecolors='black', linewidths=1.5, zorder=4, alpha=0.7)
                if label == 'Final':
                    ax.annotate(label, (path_x[idx], path_y[idx]), 
                              xytext=(5, 5), textcoords='offset points',
                              fontsize=9, fontweight='bold', color=color)
        
        # Plot the final SA point with special styling
        ax.scatter(x_coord, best_fit, color=color, marker='*', s=600, 
                  label='SA (Simulated Annealing)', edgecolors='black', 
                  linewidths=2.5, zorder=6, alpha=1.0)
        
        # Add annotation for SA
        ax.annotate('SA Final Solution', (x_coord, best_fit), 
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=10, fontweight='bold', color=color,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                            edgecolor=color, alpha=0.8),
                   arrowprops=dict(arrowstyle='->', color=color, lw=2))
    
    # Set fixed limits as requested
    ax.set_xlim(-5, 5)
    ax.set_ylim(0, 15)
    
    # Set nice tick marks
    ax.set_xticks(np.arange(-5, 6, 1))
    ax.set_yticks(np.arange(0, 16, 1))
    
    ax.set_xlabel('x', fontsize=14, fontweight='bold')
    ax.set_ylabel('f(x)', fontsize=14, fontweight='bold')
    ax.set_title('Sphere Function with Algorithm Best Fitness', fontsize=16, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11, framealpha=0.95, ncol=2)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    return fig


def main():
    """Main function to run experiments and generate all visualizations"""
    output_dir = 'src/visualization'
    os.makedirs(output_dir, exist_ok=True)
    
    # Run sphere experiments
    sphere_results = run_sphere_experiments()
    
    # Generate sphere visualizations
    print("\nGenerating Sphere Problem Visualizations...")
    plot_convergence_individual(sphere_results, 'sphere', output_dir)
    plot_convergence_combined(sphere_results, 'sphere', 
                              os.path.join(output_dir, 'sphere_convergence_combined.png'))
    plot_runtime_bar(sphere_results, 'sphere', 
                     os.path.join(output_dir, 'sphere_runtime_comparison.png'))
    plot_memory_bar(sphere_results, 'sphere', 
                    os.path.join(output_dir, 'sphere_memory_comparison.png'))
    plot_sphere_with_algorithms(sphere_results, 
                                os.path.join(output_dir, 'sphere_with_algorithms.png'))
    
    # Run knapsack experiments
    knapsack_results = run_knapsack_experiments()
    
    # Generate knapsack visualizations
    print("\nGenerating Knapsack Problem Visualizations...")
    plot_convergence_individual(knapsack_results, 'knapsack', output_dir)
    plot_convergence_combined(knapsack_results, 'knapsack', 
                              os.path.join(output_dir, 'knapsack_convergence_combined.png'))
    plot_runtime_bar(knapsack_results, 'knapsack', 
                     os.path.join(output_dir, 'knapsack_runtime_comparison.png'))
    plot_memory_bar(knapsack_results, 'knapsack', 
                    os.path.join(output_dir, 'knapsack_memory_comparison.png'))
    
    print("\n" + "=" * 60)
    print("All visualizations generated successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
