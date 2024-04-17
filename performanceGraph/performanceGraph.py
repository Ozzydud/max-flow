import matplotlib
# Use Agg backend for non-interactive mode
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot_nodes_edges_time(nodes_edges, cpu_times, gpu_times, output_file):
    # Assuming nodes_edges, cpu_times, and gpu_times are lists of corresponding values

    plt.figure(figsize=(10, 5))  # Set the figure size

    # Plot CPU times as a blue line
    plt.plot(nodes_edges, cpu_times, marker='o', color='blue', label='CPU Time')

    # Plot GPU times as a red line
    plt.plot(nodes_edges, gpu_times, marker='x', color='red', label='GPU Time')

    # Add labels and title
    plt.xlabel('Number of Nodes & Edges')
    plt.ylabel('Time (ms)')
    plt.title('CPU vs GPU Time')

    # Add legend
    plt.legend()

    # Show grid
    plt.grid(True)

    # Save the plot to a file
    plt.tight_layout()
    plt.savefig(output_file)

# Example usage
nodes_edges = [100, 200, 300, 400, 500]  # Example number of nodes & edges
cpu_times = [10, 20, 30, 40, 50]  # Example CPU times in ms
gpu_times = [5, 10, 15, 20, 25]    # Example GPU times in ms
output_file = "performance_plot.png"  # Output file name

plot_nodes_edges_time(nodes_edges, cpu_times, gpu_times, output_file)
