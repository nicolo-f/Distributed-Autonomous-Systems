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
        self.N = self.get_parameter("N").value
        dt = self.get_parameter("dt").value
        
        # Storage for neighbor cost data (same pattern as the_agent.py!)
        self.neighbor_costs = {i: 0.0 for i in range(self.N)}
        self.neighbor_grad_norms = {i: 0.0 for i in range(self.N)}
        self.data_received = {i: False for i in range(self.N)}
        
        # History for plotting
        self.total_cost_history = []
        self.total_grad_norm_history = []
        self.iteration_history = []
        
        self.k = 0

        # Subscribe to cost topics from all agents
        for i in range(self.N):
            self.create_subscription(
                MsgFloat,
                f"/cost_{i}",
                lambda msg, agent_id=i: self.cost_callback(msg, agent_id),
                10,
            )

        # ✅ ADD: Publisher for stop signal
        self.stop_publisher = self.create_publisher(
            MsgFloat,
            "/stop_signal",
            10
        )

        # Setup matplotlib figure
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
        
        self.timer = self.create_timer(dt, self.timer_callback)

        # Timer 2: Update plots (separate, like old version)
        self.plot_timer = self.create_timer(0.2, self.update_plot)

    def cost_callback(self, msg, agent_id):
        """
        Receive cost data from agents (same pattern as listener_callback!)
        Message format: [id, iter, cost_i, grad_norm_i]
        """
        iteration = int(msg.data[1])
        cost_i = msg.data[2]
        grad_norm_i = msg.data[3]
        
        # Only accept data for current iteration (synchronization!)
        self.neighbor_costs[agent_id] = cost_i
        self.neighbor_grad_norms[agent_id] = grad_norm_i
        self.data_received[agent_id] = True

    def timer_callback(self):
        """Main control loop (SAME STRUCTURE as the_agent.py!)"""
        
        # Wait for all agent data (same pattern as the_agent.py!)
        if self.k > 0:
        # if True:
            if not all(self.data_received.values()):
                return

        # ============================================================
        # COMPUTE TOTAL COST AND GRADIENT NORM
        # ============================================================
        
        # Total cost = sum of individual costs
        total_cost = sum(self.neighbor_costs.values())
        
        # Total gradient norm = sqrt(sum of squared norms)
        total_grad_norm = np.sqrt(sum(gn**2 for gn in self.neighbor_grad_norms.values()))
        
        # Store for plotting
        self.total_cost_history.append(total_cost)
        self.total_grad_norm_history.append(total_grad_norm)
        self.iteration_history.append(self.k)
        
        # Reset flags for next iteration (same as the_agent.py!)
        self.data_received = {i: False for i in range(self.N)}
        self.k += 1

        # Check for convergence
        if len(self.total_cost_history) > 100:
            recent_costs = self.total_cost_history[-20:]
            cost_variance = np.var(recent_costs)
            cost_trend = recent_costs[-1] - recent_costs[0]
            
            # self.get_logger().info(
            #     f"Cost trend (last 10): Δ={cost_trend:.6f}, Var={cost_variance:.6e}"
            # )
            
            if cost_variance < 1e-3 :
                self.get_logger().info("✅ Cost has stabilized!")
                self.get_logger().info("Shutting down all nodes and stopping the monitor.")
                
                # ✅ CHANGE: Send stop signal instead of shutdown
                stop_msg = MsgFloat()
                stop_msg.data = [1.0]  # Stop signal
                self.stop_publisher.publish(stop_msg)
                
                # ✅ CHANGE: Don't close plots, just stop processing
                self.timer.cancel()  # Stop this timer
                self.plot_timer.cancel()  # Stop plot updates
                plt.ioff()
                plt.show(block=True)  # Keep plots visible and block


    def update_plot(self):
        """Update the plots with latest data (SEPARATE TIMER, like old version!)"""
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
        print("\n📊 Plotter shutting down...")
    finally:
        plt.ioff()
        plt.show()  # Keep final plot open
        plotter.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()