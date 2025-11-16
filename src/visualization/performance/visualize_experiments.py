import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import importlib.util

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

spec_sphere = importlib.util.spec_from_file_location("run_sphere", 
    os.path.join(project_root, "src/experiment/run_sphere.py"))
run_sphere = importlib.util.module_from_spec(spec_sphere)
spec_sphere.loader.exec_module(run_sphere)

spec_knapsack = importlib.util.spec_from_file_location("run_knapsack", 
    os.path.join(project_root, "src/experiment/run_knapsack.py"))
run_knapsack = importlib.util.module_from_spec(spec_knapsack)
spec_knapsack.loader.exec_module(run_knapsack)

from src.problem.continuous.sphere import sphere


def get_nice_tick_step(data_range, max_ticks=10):
    """Calculate nice tick step size for axis formatting.

    Parameters:
    data_range (float): Range of data values
    max_ticks (int): Maximum number of ticks desired

    Returns:
    float: Nice tick step size
    """
    if data_range == 0:
        return 1.0
    initial_step = data_range / max_ticks
    magnitude = 10 ** np.floor(np.log10(initial_step))
    normalized = initial_step / magnitude
    if normalized <= 1:
        nice_normalized = 1
    elif normalized <= 2:
        nice_normalized = 2
    elif normalized <= 5:
        nice_normalized = 5
    else:
        nice_normalized = 10
    return nice_normalized * magnitude


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
    """Run sphere function experiments for all algorithms and collect results.

    Parameters:
    None

    Returns:
    dict: Dictionary mapping algorithm names to their results (best_solution, best_fitness, history, runtime, memory)
    """
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
    """Run knapsack problem experiments for all algorithms and collect results.

    Parameters:
    None

    Returns:
    dict: Dictionary mapping algorithm names to their results (best_solution, best_fitness, history, runtime, memory)
    """
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


def plot_convergence_individual(results_dict, problem_type='sphere', save_dir='src/visualization/performance'):
    """Generate individual convergence plots for each algorithm.

    Parameters:
    results_dict (dict): Dictionary mapping algorithm names to their results
    problem_type (str): Type of problem ('sphere' or 'knapsack')
    save_dir (str): Directory path to save the plots

    Returns:
    None
    """
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
        max_fitness = max(history)
        min_fitness = min(history)
        y_padding = (max_fitness - min_fitness) * 0.1
        y_max = max_fitness + y_padding
        y_min = max(0, min_fitness - y_padding)
        ax.set_ylim(y_min, y_max)
        data_range = y_max - y_min
        tick_step = get_nice_tick_step(data_range, max_ticks=10)
        y_min_tick = np.floor(y_min / tick_step) * tick_step
        y_max_tick = np.ceil(y_max / tick_step) * tick_step
        y_ticks = np.arange(y_min_tick, y_max_tick + tick_step, tick_step)
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
    """Generate combined convergence plot comparing all algorithms.

    Parameters:
    results_dict (dict): Dictionary mapping algorithm names to their results
    problem_type (str): Type of problem ('sphere' or 'knapsack')
    save_path (str, optional): File path to save the plot

    Returns:
    matplotlib.figure.Figure: Figure object containing the plot
    """
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
    max_fitness = max([max(result['history']) for result in results_dict.values()])
    min_fitness = min([min(result['history']) for result in results_dict.values()])
    y_padding = (max_fitness - min_fitness) * 0.1
    y_max = max_fitness + y_padding
    y_min = max(0, min_fitness - y_padding)
    ax.set_ylim(y_min, y_max)
    data_range = y_max - y_min
    tick_step = get_nice_tick_step(data_range, max_ticks=10)
    y_min_tick = np.floor(y_min / tick_step) * tick_step
    y_max_tick = np.ceil(y_max / tick_step) * tick_step
    y_ticks = np.arange(y_min_tick, y_max_tick + tick_step, tick_step)
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
    """Generate bar chart comparing algorithm runtime performance.

    Parameters:
    results_dict (dict): Dictionary mapping algorithm names to their results
    problem_type (str): Type of problem ('sphere' or 'knapsack')
    save_path (str, optional): File path to save the plot

    Returns:
    matplotlib.figure.Figure: Figure object containing the plot
    """
    algorithms = list(results_dict.keys())
    runtimes = [results_dict[algo]['runtime'] for algo in algorithms]
    colors = [COLORS.get(algo, '#95A5A6') for algo in algorithms]
    min_runtime = min(runtimes)
    max_runtime = max(runtimes)
    data_range = max_runtime - min_runtime
    tick_step = get_nice_tick_step(data_range, max_ticks=10)
    num_ticks = int(np.ceil(data_range / tick_step)) + 1
    sorted_runtimes = sorted(runtimes)
    gaps = [sorted_runtimes[i+1] - sorted_runtimes[i] for i in range(len(sorted_runtimes)-1)]
    max_gap = max(gaps) if len(gaps) > 0 else 0
    gap_percentage = (max_gap / data_range * 100) if data_range > 0 else 0
    use_broken_axis = (num_ticks > 50) or (gap_percentage > 60 and data_range > 0.1)
    
    if use_broken_axis:
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 8),
            gridspec_kw={'height_ratios': [1, 3]})
        fig.subplots_adjust(hspace=0.05)
        x = np.arange(len(algorithms))
        gaps = [sorted_runtimes[i+1] - sorted_runtimes[i] for i in range(len(sorted_runtimes)-1)]
        if len(gaps) > 0:
            max_gap_idx = np.argmax(gaps)
            upper_min = sorted_runtimes[max_gap_idx] + gaps[max_gap_idx] * 0.5
        else:
            upper_min = min_runtime * 1.15
        upper_values = [v for v in runtimes if v >= upper_min]
        lower_values = [v for v in runtimes if v < upper_min]
        
        if len(upper_values) > 0 and len(lower_values) > 0:
            upper_max = max(upper_values) * 1.25
            lower_max = max(lower_values) * 1.2
            upper_indices = [i for i, v in enumerate(runtimes) if v >= upper_min]
            upper_x = [x[i] for i in upper_indices]
            upper_runtimes = [runtimes[i] for i in upper_indices]
            upper_colors = [colors[i] for i in upper_indices]
            upper_heights = [v - upper_min for v in upper_runtimes]
            ax1.bar(upper_x, upper_heights, bottom=upper_min, color=upper_colors, alpha=0.8, edgecolor='black', linewidth=1.5)
            ax1.set_ylabel('Runtime (seconds)', fontsize=12, fontweight='bold')
            ax1.set_title('Algorithm Runtime Comparison', fontsize=14, fontweight='bold', pad=15)
            ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
            ax1.set_ylim(upper_min, upper_max)
            high_lower_x = [x[i] for i in upper_indices]
            high_lower_heights = [upper_min] * len(upper_indices)
            high_lower_colors = [colors[i] for i in upper_indices]
            ax2.bar(high_lower_x, high_lower_heights, bottom=0, color=high_lower_colors, alpha=0.8, edgecolor='black', linewidth=1.5)
            lower_indices = [i for i, v in enumerate(runtimes) if v < upper_min]
            lower_x = [x[i] for i in lower_indices]
            lower_runtimes = [runtimes[i] for i in lower_indices]
            lower_colors = [colors[i] for i in lower_indices]
            ax2.bar(lower_x, lower_runtimes, bottom=0, color=lower_colors, alpha=0.8, edgecolor='black', linewidth=1.5)
            ax2.set_xlabel('Algorithm', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Runtime (seconds)', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
            ax2.set_ylim(0, lower_max)
            ax1.spines['bottom'].set_visible(False)
            ax2.spines['top'].set_visible(False)
            ax1.xaxis.tick_top()
            ax1.tick_params(labeltop=False, top=False, labelsize=10)
            ax2.xaxis.tick_bottom()
            ax2.tick_params(labelsize=10)
            d = 0.01
            kwargs = dict(color='k', clip_on=False, linewidth=1.5)
            ax1.plot((-d, +d), (-d, +d), transform=ax1.transAxes, **kwargs)
            ax1.plot((1 - d, 1 + d), (-d, +d), transform=ax1.transAxes, **kwargs)
            ax2.plot((-d, +d), (1 - d, 1 + d), transform=ax2.transAxes, **kwargs)
            ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), transform=ax2.transAxes, **kwargs)
            for i, v in enumerate(runtimes):
                if v >= upper_min:
                    label_y = v + (upper_max - upper_min) * 0.05
                    if label_y > upper_max:
                        label_y = upper_max - (upper_max - upper_min) * 0.02
                    ax1.text(x[i], label_y, f'{v:.4f}', 
                            ha='center', va='bottom', fontsize=9, fontweight='bold', clip_on=False)
                else:
                    ax2.text(x[i], v + lower_max * 0.02, f'{v:.4f}', 
                            ha='center', va='bottom', fontsize=9, fontweight='bold', clip_on=False)
            ax2.set_xticks(x)
            ax2.set_xticklabels(algorithms)
        else:
            fig.delaxes(ax1)
            fig.delaxes(ax2)
            ax = fig.add_subplot(111)
            ax.bar(x, runtimes, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
            ax.set_xlabel('Algorithm', fontsize=12, fontweight='bold')
            ax.set_ylabel('Runtime (seconds)', fontsize=12, fontweight='bold')
            ax.set_title('Algorithm Runtime Comparison', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y', linestyle='--')
            ax.set_xticks(x)
            ax.set_xticklabels(algorithms)
            for i, v in enumerate(runtimes):
                ax.text(i, v + max_runtime * 0.01, f'{v:.4f}', 
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
    else:
        fig, ax = plt.subplots(figsize=(12, 7))
        x = np.arange(len(algorithms))
        ax.bar(x, runtimes, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax.set_xlabel('Algorithm', fontsize=12, fontweight='bold')
        ax.set_ylabel('Runtime (seconds)', fontsize=12, fontweight='bold')
        ax.set_title('Algorithm Runtime Comparison', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax.set_xticks(x)
        ax.set_xticklabels(algorithms)
        tick_step = get_nice_tick_step(data_range, max_ticks=10)
        y_min = np.floor(min_runtime / tick_step) * tick_step
        y_max = np.ceil(max_runtime / tick_step) * tick_step
        padding_bottom = (y_max - y_min) * 0.05
        padding_top = (y_max - y_min) * 0.15
        y_min = max(0, y_min - padding_bottom)
        y_max = y_max + padding_top
        y_ticks = np.arange(np.floor(y_min / tick_step) * tick_step, y_max + tick_step, tick_step)
        ax.set_yticks(y_ticks)
        ax.set_ylim(y_min, y_max)
        for i, v in enumerate(runtimes):
            ax.text(i, v + max_runtime * 0.01, f'{v:.4f}', 
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    return fig


def plot_memory_bar(results_dict, problem_type='sphere', save_path=None):
    """Generate bar chart comparing algorithm memory usage.

    Parameters:
    results_dict (dict): Dictionary mapping algorithm names to their results
    problem_type (str): Type of problem ('sphere' or 'knapsack')
    save_path (str, optional): File path to save the plot

    Returns:
    matplotlib.figure.Figure: Figure object containing the plot
    """
    algorithms = list(results_dict.keys())
    memories = [results_dict[algo]['memory'] / 1024 for algo in algorithms]
    colors = [COLORS.get(algo, '#95A5A6') for algo in algorithms]
    min_memory = min(memories)
    max_memory = max(memories)
    data_range = max_memory - min_memory
    tick_step = get_nice_tick_step(data_range, max_ticks=10)
    num_ticks = int(np.ceil(data_range / tick_step)) + 1
    sorted_memories = sorted(memories)
    gaps = [sorted_memories[i+1] - sorted_memories[i] for i in range(len(sorted_memories)-1)]
    max_gap = max(gaps) if len(gaps) > 0 else 0
    gap_percentage = (max_gap / data_range * 100) if data_range > 0 else 0
    use_broken_axis = (num_ticks > 50) or (gap_percentage > 60 and data_range > 5)
    
    if use_broken_axis:
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 8),
            gridspec_kw={'height_ratios': [1, 3]})
        fig.subplots_adjust(hspace=0.05)
        x = np.arange(len(algorithms))
        gaps = [sorted_memories[i+1] - sorted_memories[i] for i in range(len(sorted_memories)-1)]
        if len(gaps) > 0:
            max_gap_idx = np.argmax(gaps)
            upper_min = sorted_memories[max_gap_idx] + gaps[max_gap_idx] * 0.5
        else:
            upper_min = min_memory * 1.15
        upper_values = [v for v in memories if v >= upper_min]
        lower_values = [v for v in memories if v < upper_min]
        
        if len(upper_values) > 0 and len(lower_values) > 0:
            upper_max = max(upper_values) * 1.15
            lower_max = max(lower_values) * 1.2
            upper_indices = [i for i, v in enumerate(memories) if v >= upper_min]
            upper_x = [x[i] for i in upper_indices]
            upper_memories = [memories[i] for i in upper_indices]
            upper_colors = [colors[i] for i in upper_indices]
            upper_heights = [v - upper_min for v in upper_memories]
            ax1.bar(upper_x, upper_heights, bottom=upper_min, color=upper_colors, alpha=0.8, edgecolor='black', linewidth=1.5)
            ax1.set_ylabel('Memory Usage (KB)', fontsize=12, fontweight='bold')
            ax1.set_title('Algorithm Memory Usage Comparison', fontsize=14, fontweight='bold', pad=15)
            ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
            ax1.set_ylim(upper_min, upper_max)
            high_lower_x = [x[i] for i in upper_indices]
            high_lower_heights = [upper_min] * len(upper_indices)
            high_lower_colors = [colors[i] for i in upper_indices]
            ax2.bar(high_lower_x, high_lower_heights, bottom=0, color=high_lower_colors, alpha=0.8, edgecolor='black', linewidth=1.5)
            lower_indices = [i for i, v in enumerate(memories) if v < upper_min]
            lower_x = [x[i] for i in lower_indices]
            lower_memories = [memories[i] for i in lower_indices]
            lower_colors = [colors[i] for i in lower_indices]
            ax2.bar(lower_x, lower_memories, bottom=0, color=lower_colors, alpha=0.8, edgecolor='black', linewidth=1.5)
            ax2.set_xlabel('Algorithm', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Memory Usage (KB)', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
            ax2.set_ylim(0, lower_max)
            ax1.spines['bottom'].set_visible(False)
            ax2.spines['top'].set_visible(False)
            ax1.xaxis.tick_top()
            ax1.tick_params(labeltop=False, top=False, labelsize=10)
            ax2.xaxis.tick_bottom()
            ax2.tick_params(labelsize=10)
            d = 0.01
            kwargs = dict(color='k', clip_on=False, linewidth=1.5)
            ax1.plot((-d, +d), (-d, +d), transform=ax1.transAxes, **kwargs)
            ax1.plot((1 - d, 1 + d), (-d, +d), transform=ax1.transAxes, **kwargs)
            ax2.plot((-d, +d), (1 - d, 1 + d), transform=ax2.transAxes, **kwargs)
            ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), transform=ax2.transAxes, **kwargs)
            for i, v in enumerate(memories):
                if v >= upper_min:
                    label_y = v + (upper_max - upper_min) * 0.05
                    if label_y > upper_max:
                        label_y = upper_max - (upper_max - upper_min) * 0.02
                    ax1.text(x[i], label_y, f'{v:.2f}', 
                            ha='center', va='bottom', fontsize=9, fontweight='bold', clip_on=False)
                else:
                    ax2.text(x[i], v + lower_max * 0.02, f'{v:.2f}', 
                            ha='center', va='bottom', fontsize=9, fontweight='bold', clip_on=False)
            ax2.set_xticks(x)
            ax2.set_xticklabels(algorithms)
        else:
            fig.delaxes(ax1)
            fig.delaxes(ax2)
            ax = fig.add_subplot(111)
            ax.bar(x, memories, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
            ax.set_xlabel('Algorithm', fontsize=12, fontweight='bold')
            ax.set_ylabel('Memory Usage (KB)', fontsize=12, fontweight='bold')
            ax.set_title('Algorithm Memory Usage Comparison', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y', linestyle='--')
            ax.set_xticks(x)
            ax.set_xticklabels(algorithms)
            for i, v in enumerate(memories):
                ax.text(i, v + max_memory * 0.01, f'{v:.2f}', 
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
    else:
        fig, ax = plt.subplots(figsize=(12, 7))
        x = np.arange(len(algorithms))
        ax.bar(x, memories, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax.set_xlabel('Algorithm', fontsize=12, fontweight='bold')
        ax.set_ylabel('Memory Usage (KB)', fontsize=12, fontweight='bold')
        ax.set_title('Algorithm Memory Usage Comparison', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax.set_xticks(x)
        ax.set_xticklabels(algorithms)
        tick_step = get_nice_tick_step(data_range, max_ticks=10)
        y_min = np.floor(min_memory / tick_step) * tick_step
        y_max = np.ceil(max_memory / tick_step) * tick_step
        padding_bottom = (y_max - y_min) * 0.05
        padding_top = (y_max - y_min) * 0.10
        y_min = max(0, y_min - padding_bottom)
        y_max = y_max + padding_top
        y_ticks = np.arange(np.floor(y_min / tick_step) * tick_step, y_max + tick_step, tick_step)
        ax.set_yticks(y_ticks)
        ax.set_ylim(y_min, y_max)
        for i, v in enumerate(memories):
            ax.text(i, v + max_memory * 0.01, f'{v:.2f}', 
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    return fig


def plot_sphere_with_algorithms(results_dict, save_path=None):
    """Generate visualization of sphere function with algorithm best fitness points.

    Parameters:
    results_dict (dict): Dictionary mapping algorithm names to their results
    save_path (str, optional): File path to save the plot

    Returns:
    matplotlib.figure.Figure: Figure object containing the plot
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.linspace(-5.12, 5.12, 400)
    y = x ** 2
    ax.plot(x, y, linewidth=2, label="Sphere Function f(x)=x²", zorder=1)
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Optimal Fitness = 0', zorder=2)
    algorithms = []
    best_values = []
    for algo_name, result in results_dict.items():
        best_fit = result['best_fitness']
        algorithms.append(algo_name)
        best_values.append(best_fit)
    offset = np.linspace(-0.5, 0.5, len(best_values))
    for i, (algo, val) in enumerate(zip(algorithms, best_values)):
        color = COLORS.get(algo, '#95A5A6')
        marker = MARKERS.get(algo, 'o')
        ax.scatter(offset[i], val, s=60, color=color, marker=marker, 
                  label=algo, edgecolors='black', linewidths=1.5, zorder=5, alpha=0.9)
    ax.set_xlabel("x", fontsize=12, fontweight='bold')
    ax.set_ylabel("f(x)", fontsize=12, fontweight='bold')
    ax.set_title("Sphere Function with Algorithm Best Fitness", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper right', fontsize=10, framealpha=0.95, ncol=2)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    return fig


def main():
    """Main function to run experiments and generate all visualizations.

    Parameters:
    None

    Returns:
    None
    """
    output_dir = 'src/visualization/performance'
    os.makedirs(output_dir, exist_ok=True)
    sphere_results = run_sphere_experiments()
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
    knapsack_results = run_knapsack_experiments()
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
