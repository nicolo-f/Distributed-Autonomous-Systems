import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray as MsgFloat
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np


class Task2Plotter(Node):
    def __init__(self):
        super().__init__(
            "task2_plotter",
            allow_undeclared_parameters=True,
            automatically_declare_parameters_from_overrides=True,
        )

        # Get parameters
        self.N = self.get_parameter("N").value
        self.dt = self.get_parameter("dt").value
        
        # Storage for cost data from all agents
        self.agent_costs = {i: [] for i in range(self.N)}  # cost_i per iteration
        self.agent_grad_norms = {i: [] for i in range(self.N)}  # ||grad_i|| per iteration
        self.iterations = {i: [] for i in range(self.N)}
        
        # Current iteration data (for aggregation)
        self.current_iter_costs = {}
        self.current_iter_grad_norms = {}
        self.current_iteration = -1
        
        # Aggregated data for plotting
        self.total_cost = []  # Sum over all agents
        self.total_grad_norm = []  # Norm of sum of gradients
        self.iteration_history = []

        # Subscribe to cost topics from all agents
        for i in range(self.N):
            self.create_subscription(
                MsgFloat,
                f"/cost_{i}",
                lambda msg, agent_id=i: self.cost_callback(msg, agent_id),
                10,
            )

        # Setup matplotlib figure
        plt.ion()  # Interactive mode
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot 1: Total cost
        self.ax1.set_xlabel('Iteration')
        self.ax1.set_ylabel('Total Cost')
        self.ax1.set_title('Total Cost Function (Sum over all agents)')
        self.ax1.grid(True)
        self.line1, = self.ax1.plot([], [], 'b-', linewidth=2)
        
        # Plot 2: Gradient norm (log scale)
        self.ax2.set_xlabel('Iteration')
        self.ax2.set_ylabel('Gradient Norm (log scale)')
        self.ax2.set_title('Norm of Total Gradient')
        self.ax2.set_yscale('log')
        self.ax2.grid(True, which='both', alpha=0.3)
        self.line2, = self.ax2.plot([], [], 'r-', linewidth=2)
        
        plt.tight_layout()
        
        # Timer for updating plots
        self.plot_timer = self.create_timer(self.dt, self.update_plot)

    def cost_callback(self, msg, agent_id):
        """
        Receive cost data from agents
        Message format: [id, iter, cost_i, grad_norm_i]
        """
        iteration = int(msg.data[1])
        cost_i = msg.data[2]
        grad_norm_i = msg.data[3]
        
        # Store data for this agent
        if iteration not in [it for it in self.iterations[agent_id]]:
            self.iterations[agent_id].append(iteration)
            self.agent_costs[agent_id].append(cost_i)
            self.agent_grad_norms[agent_id].append(grad_norm_i)
        
        # Accumulate data for current iteration
        if iteration > self.current_iteration:
            # New iteration started - process previous iteration
            if len(self.current_iter_costs) == self.N:
                self.aggregate_iteration_data()
            
            # Reset for new iteration
            self.current_iteration = iteration
            self.current_iter_costs = {}
            self.current_iter_grad_norms = {}
        
        # Store current iteration data
        self.current_iter_costs[agent_id] = cost_i
        self.current_iter_grad_norms[agent_id] = grad_norm_i
        
        # If all agents reported for this iteration, aggregate
        if len(self.current_iter_costs) == self.N:
            self.aggregate_iteration_data()

    def aggregate_iteration_data(self):
        """Compute total cost and total gradient norm"""
        # Total cost = sum of individual costs
        total_cost = sum(self.current_iter_costs.values())
        
        # Total gradient norm = sqrt(sum of squared norms)
        # Note: This is an approximation since we don't have individual gradient vectors
        total_grad_norm = np.sqrt(sum(gn**2 for gn in self.current_iter_grad_norms.values()))
        
        self.total_cost.append(total_cost)
        self.total_grad_norm.append(total_grad_norm)
        self.iteration_history.append(self.current_iteration)
        
        if self.current_iteration % 50 == 0:
            print(f"Iteration {self.current_iteration}: "
                  f"Total Cost = {total_cost:.4f}, "
                  f"Grad Norm = {total_grad_norm:.4e}")

    def update_plot(self):
        """Update the plots with latest data"""
        if len(self.iteration_history) == 0:
            return
        
        # Update total cost plot
        self.line1.set_data(self.iteration_history, self.total_cost)
        self.ax1.relim()
        self.ax1.autoscale_view()
        
        # Update gradient norm plot
        if len(self.total_grad_norm) > 0 and max(self.total_grad_norm) > 0:
            self.line2.set_data(self.iteration_history, self.total_grad_norm)
            self.ax2.relim()
            self.ax2.autoscale_view()
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


def main(args=None):
    rclpy.init(args=args)
    plotter = Task2Plotter()
    
    print("📊 Plotter ready! Waiting for cost data...\n")
    
    try:
        rclpy.spin(plotter)
    except KeyboardInterrupt:
        print("\n📊 Plotter shutting down...")
    finally:
        plt.ioff()
        plt.show()  # Keep final plot open
        plotter.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()