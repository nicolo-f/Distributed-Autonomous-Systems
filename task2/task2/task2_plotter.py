import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray as MsgFloat
import matplotlib.pyplot as plt
import numpy as np


class Task2Plotter(Node):
    def __init__(self):
        super().__init__(
            "task2_plotter",
            allow_undeclared_parameters=True,
            automatically_declare_parameters_from_overrides=True,
        )

        # Get parameters
        self.N = self.get_parameter("N").value  # Total number of agents
        dt = self.get_parameter("dt").value     # Time step
        
        # Storage for data to plot from all agents
        self.costs = {i: 0.0 for i in range(self.N)}
        self.grad_norms = {i: 0.0 for i in range(self.N)}
        self.data_received = {i: False for i in range(self.N)}
        
        # History for plotting
        self.total_cost_history = []
        self.total_grad_norm_history = []
        self.iteration_history = []
        
        self.k = 0 # Iteration counter

        # Define publishers and subscribers

        # Crreate listener to receive data from all agents
        for i in range(self.N):
            self.create_subscription(
                MsgFloat,
                f"/cost_{i}",
                lambda msg, agent_id=i: self.cost_callback(msg, agent_id),
                10,
            )

         # Create a publisher to early stop the algorithm
        self.stop_publisher = self.create_publisher(
            MsgFloat,
            "/stop_signal",
            10
        )

        self.initialize_plot()  # Initialize plotting
        
        self.timer = self.create_timer(dt, self.timer_callback)     # Main control loop timer
        self.plot_timer = self.create_timer(0.2, self.update_plot)  # Plot update timer

    def cost_callback(self, msg, agent_id):
        """
        Receive cost data from agents 
        Message format: [id, iter, cost_i, grad_norm_i]
        """
        iteration = int(msg.data[1])
        cost_i = msg.data[2]
        grad_norm_i = msg.data[3]
        
        # Store received data
        self.costs[agent_id] = cost_i
        self.grad_norms[agent_id] = grad_norm_i
        self.data_received[agent_id] = True

    def timer_callback(self):
        """Main control loop"""
        
        # Wait for data from all neighbors 
        if self.k > 0 and len(self.neighbors) > 0:
        # if True:
            if not all(self.data_received.values()):
                return
            
        # Compute total cost and gradient norm from all agents
        total_cost = sum(self.costs.values())
        total_grad_norm = np.sqrt(sum(gn**2 for gn in self.grad_norms.values()))
        
        # Store for plotting
        self.total_cost_history.append(total_cost)
        self.total_grad_norm_history.append(total_grad_norm)
        self.iteration_history.append(self.k)
        
        # Reset data received flags
        self.data_received = {i: False for i in range(self.N)}

        self.check_early_stop(self.total_cost_history)

        self.k += 1 # Increment iteration counter


    def check_early_stop(self, total_cost):
        # Check for convergence
        if len(total_cost) > 100:
            recent_costs = total_cost[-20:]
            cost_variance = np.var(recent_costs)
            
            if cost_variance < 1e-3 :
                self.get_logger().info("Cost has stabilized!")
                self.get_logger().info("Shutting down all nodes and stopping the monitor")
                
                # Send stop signal to the agents
                stop_msg = MsgFloat()
                stop_msg.data = [1.0]  # Stop signal
                self.stop_publisher.publish(stop_msg)
                
                # Stop timers and keep final plot open 
                self.timer.cancel()
                self.plot_timer.cancel()

                plt.ioff() 
                plt.show(block=True)  # Keep plots visible and block

    # Setup matplotlib figure
    def initialize_plot(self):
        plt.ion()  # Interactive mode
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot 1: Total cost
        self.ax1.set_xlabel('Iteration')
        self.ax1.set_ylabel('Total Cost')
        self.ax1.set_title('Total Cost Function (Sum over all agents)')
        self.ax1.grid(True)
        self.line1, = self.ax1.plot([], [], 'b-', linewidth=2, label='Total Cost')
        self.ax1.legend()
        
        # Plot 2: Gradient norm (log scale)
        self.ax2.set_xlabel('Iteration')
        self.ax2.set_ylabel('Gradient Norm (log scale)')
        self.ax2.set_title('Norm of Total Gradient')
        self.ax2.set_yscale('log')
        self.ax2.grid(True, which='both', alpha=0.3)
        self.line2, = self.ax2.plot([], [], 'r-', linewidth=2, label='Gradient Norm')
        self.ax2.legend()
        
        plt.tight_layout()

    def update_plot(self):
        """Update the plots with latest data"""

        # Only update if we have data
        if len(self.iteration_history) == 0:
            return
        
        # Update total cost plot
        self.line1.set_data(self.iteration_history, self.total_cost_history)
        self.ax1.relim()
        self.ax1.autoscale_view()
        
        # Update gradient norm plot
        if len(self.total_grad_norm_history) > 0 and max(self.total_grad_norm_history) > 0:
            self.line2.set_data(self.iteration_history, self.total_grad_norm_history)
            self.ax2.relim()
            self.ax2.autoscale_view()
        
        self.fig.canvas.draw()   
        self.fig.canvas.flush_events()


def main(args=None):
    rclpy.init(args=args)
    plotter = Task2Plotter()
    
    try:
        rclpy.spin(plotter)
    except KeyboardInterrupt:
        print("\nPlotter shutting down")
    finally:
        plt.ioff()
        plt.show()  # Keep final plot open
        plotter.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()