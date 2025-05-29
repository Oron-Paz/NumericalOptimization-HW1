"""
utils.py - Utility functions for plotting and visualization
"""

def plot_contours_with_path(objective_func, histories, method_names, title, x_limits=(-2, 2), y_limits=(-2, 2)):
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        if not isinstance(x_limits, (tuple, list)) or len(x_limits) != 2:
            x_limits = (-2, 2)
        if not isinstance(y_limits, (tuple, list)) or len(y_limits) != 2:
            y_limits = (-2, 2)
        
        x_min, x_max = x_limits
        y_min, y_max = y_limits
        x = np.linspace(x_min, x_max, 100)
        y = np.linspace(y_min, y_max, 100)
        X, Y = np.meshgrid(x, y)
        
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                point = [X[i, j], Y[i, j]]
                f_val, _, _ = objective_func(point, hessian_needed=False)
                Z[i, j] = f_val
        
        plt.figure(figsize=(10, 8))
        
        contour_levels = np.logspace(-3, 2, 20)  
        plt.contour(X, Y, Z, levels=contour_levels, colors='gray', alpha=0.6)
        plt.contourf(X, Y, Z, levels=contour_levels, alpha=0.3, cmap='viridis')
        
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        for i, (history, method_name) in enumerate(zip(histories, method_names)):
            if history:
                x_path = [point["location"][0] for point in history]
                y_path = [point["location"][1] for point in history]
                
                plt.plot(x_path, y_path, color=colors[i % len(colors)], 
                        marker='o', markersize=4, linewidth=2, label=method_name)
                
                plt.plot(x_path[0], y_path[0], color=colors[i % len(colors)], 
                        marker='s', markersize=8, label=f'{method_name} Start')
                plt.plot(x_path[-1], y_path[-1], color=colors[i % len(colors)], 
                        marker='*', markersize=10, label=f'{method_name} End')
        
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.xlim(x_limits)
        plt.ylim(y_limits)
        
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("Matplotlib not available. Cannot create contour plot.")
        print("Install with: pip install matplotlib")
        
        print(f"\n{title}")
        print("-" * len(title))
        for history, method_name in zip(histories, method_names):
            if history:
                print(f"\n{method_name} path:")
                print(f"Start: {history[0]['location']}")
                print(f"End: {history[-1]['location']}")
                print(f"Iterations: {len(history)}")

def plot_function_values(histories, method_names, title):
    """
    Plot function values vs iteration number for comparison
    
    Args:
        histories: List of optimization histories (one per method)
        method_names: List of method names for the legend
        title: Plot title
    """
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        for i, (history, method_name) in enumerate(zip(histories, method_names)):
            if history:
                iterations = [point["iteration"] for point in history]
                function_values = [point["objective_value"] for point in history]
                
                plt.semilogy(iterations, function_values, color=colors[i % len(colors)], 
                           marker='o', linewidth=2, label=method_name)
        
        plt.xlabel('Iteration Number')
        plt.ylabel('Function Value (log scale)')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("Matplotlib not available. Cannot create function value plot.")
        print("Install with: pip install matplotlib")
        
        print(f"\n{title}")
        print("-" * len(title))
        for history, method_name in zip(histories, method_names):
            if history:
                print(f"\n{method_name} convergence:")
                print(f"Initial value: {history[0]['objective_value']:.6e}")
                print(f"Final value: {history[-1]['objective_value']:.6e}")
                print(f"Reduction: {history[0]['objective_value'] / history[-1]['objective_value']:.2e}")
                print(f"Iterations: {len(history)}")

def print_algorithm_summary(history, method_name):
    """
    Print a summary of the algorithm's performance
    
    Args:
        history: Optimization history
        method_name: Name of the optimization method
    """
    if not history:
        print(f"No history available for {method_name}")
        return
    
    print(f"\n{method_name} Summary:")
    print("=" * 40)
    print(f"Total iterations: {len(history)}")
    print(f"Initial location: {history[0]['location']}")
    print(f"Final location: {history[-1]['location']}")
    print(f"Initial function value: {history[0]['objective_value']:.6e}")
    print(f"Final function value: {history[-1]['objective_value']:.6e}")
    
    if history[0]['objective_value'] != 0:
        reduction = abs(history[0]['objective_value']) / abs(history[-1]['objective_value'])
        print(f"Function value reduction: {reduction:.2e}")

def create_simple_plot_fallback(histories, method_names, title):
    """
    Simple text-based plotting fallback when matplotlib is not available
    """
    print(f"\n{title}")
    print("=" * len(title))
    
    for history, method_name in zip(histories, method_names):
        if history:
            print(f"\n{method_name}:")
            print(f"  Start: {history[0]['location']} -> f = {history[0]['objective_value']:.6e}")
            print(f"  End:   {history[-1]['location']} -> f = {history[-1]['objective_value']:.6e}")
            print(f"  Iterations: {len(history)}")
            
            if len(history) > 3:
                mid_idx = len(history) // 2
                print(f"  Mid:   {history[mid_idx]['location']} -> f = {history[mid_idx]['objective_value']:.6e}")

def test_utils():
    """Test function to verify utils work properly"""
    print("Testing utils.py...")
    
    dummy_history = [
        {"iteration": 0, "location": [1.0, 1.0], "objective_value": 2.0},
        {"iteration": 1, "location": [0.5, 0.5], "objective_value": 0.5},
        {"iteration": 2, "location": [0.1, 0.1], "objective_value": 0.02}
    ]
    
    print_algorithm_summary(dummy_history, "Test Method")
    create_simple_plot_fallback([dummy_history], ["Test Method"], "Test Plot")
    
    print("Utils test completed!")

if __name__ == '__main__':
    test_utils()