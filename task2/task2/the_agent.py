import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray as MsgFloat
from time import sleep
import numpy as np

from task2.helper import CostFunction

class Agent(Node):
    def __init__(self):
        super().__init__(
            "agent",
            allow_undeclared_parameters=True,
            automatically_declare_parameters_from_overrides=True,
        )

        # Get parameters
        self.agent_id = self.get_parameter("id").value    # Agent ID
        self.N = self.get_parameter("N").value            # Total number of agents
        self.d = self.get_parameter("d").value            # Dimension of decision variable

        self.maxIters = self.get_parameter("maxIters").value    # Maximum iterations
        dt = self.get_parameter("dt").value                     # Time step

        self.target = np.array(self.get_parameter("target").value)  # Target position for agent i
        self.neighbors = self.get_parameter("neighbors").value      # List of neighbor of agent i
        self.z = np.array(self.get_parameter("z0").value)           # Agent i initial position
        self.r0 = self.get_parameter("r0").value                    # Desired barycenter contribution
        
        self.alpha = self.get_parameter("alpha").value  # Gradient tracking step-size
        self.gamma = self.get_parameter("gamma").value  # Target/barycenter trade-off parameter

        # Extract neighbors (non-zero elements on A matrix, excluding self-loop)
        self.A_row = np.array(self.get_parameter("A_row").value)
        self.neighbors = [j for j in range(self.N) if self.A_row[j] > 0 and j != self.agent_id]
        
        self.s = self.z.copy() # Initialize s[0, i] = z[0, i]

        # Initialize v[0, i]
        _, _, self.v = CostFunction.distributed_aggregative(
            self.z, self.s, self.gamma, self.r0, self.target, self.d)
        
        # Initialize variables for storing neighbor data
        self.neighbor_z = {j: np.zeros(self.d) for j in self.neighbors}
        self.neighbor_s = {j: np.zeros(self.d) for j in self.neighbors}
        self.neighbor_v = {j: np.zeros(self.d) for j in self.neighbors}
        self.data_received = {j: False for j in self.neighbors} 

        self.k = 0  # Iteration counter

        self.stopped = False # Flag to indicate if the agent should stop

        # Print initialization
        print("\n" + "="*60)
        print(f"AGENT {self.agent_id} INITIALIZED")
        print("="*60)
        print(f"Position z[0]: {self.z}")
        print(f"Barycenter s[0]: {self.s}")
        print(f"Gradient v[0]: {self.v}")
        print(f"Target: {self.target}")
        print(f"r0: {self.r0}")
        print(f"Neighbors: {self.neighbors}")
        print(f"Weight row A[{self.agent_id}]: {self.A_row}")
        print(f"Alpha: {self.alpha}, Gamma: {self.gamma}")
        print("="*60 + "\n")

        # Define publishers and subscribers

        # Create a listener to early stop the algorithm
        # The plotter node will publish a stop signal when convergence is detected 
        self.create_subscription(
            MsgFloat,
            "/stop_signal",
            self.stop_callback,
            10
        )

        # Create a listener to each neighbor to pass iterations data
        for j in self.neighbors:
            self.create_subscription(
                MsgFloat,
                f"/data_{j}",
                self.data_callback,
                10,
            )

        # Create a publisher to send own data to neighbors
        self.publisher = self.create_publisher(
            MsgFloat,
            f"/data_{self.agent_id}",
            10,
        )

        # Create a publisher for cost data plotting
        self.cost_publisher = self.create_publisher(
            MsgFloat,
            f"/cost_{self.agent_id}",
            10,
        )

        self.timer = self.create_timer(dt, self.timer_callback) # Main control loop timer 

    def stop_callback(self, msg):
        """Receive stop signal from plotter"""
        self.stopped = True # Set stop flag
        print(f"\n Agent {self.agent_id}: Received STOP signal - algorithm converged!")
        print(f"   Final position: {self.z.round(3)}")
        print(f"   Target: {self.target.round(3)}")
        print(f"   Final iteration: {self.k}")
        print(f"   Distance to target: {np.linalg.norm(self.z - self.target):.6f}\n")

    def data_callback(self, msg):
        """
        Store received neighbor data
        Message format: [id, iter, z, s, v]
        """
        n_id = int(msg.data[0])
        iteration = int(msg.data[1])

        # Store neighbor data in appropriate variables
        self.neighbor_z[n_id] = np.array(msg.data[2:2+self.d])
        self.neighbor_s[n_id] = np.array(msg.data[2+self.d:2+2*self.d])
        self.neighbor_v[n_id] = np.array(msg.data[2+2*self.d:2+3*self.d])
        self.data_received[n_id] = True # Mark data as correctly received

    def timer_callback(self):
        """Main control loop"""

        # Check if algorithm has already converged
        if self.stopped:
            return  # Stop processing but keep node alive
        
        msg = MsgFloat()
            
        # Wait for data from all neighbors 
        if self.k > 0 and len(self.neighbors) > 0:
            if not all(self.data_received.values()):   
                return
        
        # Store old z for innovation term
        z_old = self.z.copy()

        # ============================================================
        # Aggregative Tracking Distributed Optimization Algorithm

        # Compute cost and gradients at current state
        cost, grad_1, grad_2 = CostFunction.distributed_aggregative(
            self.z, self.s, self.gamma, self.r0, self.target, self.d)

        # Update position z[k+1] = z[k] - alpha*(grad_1 + v[k])
        self.z = self.z - self.alpha * (grad_1 + self.v)

        self.r0 = self.z.copy()  # Update local contribution to barycenter
        
        # Weighted consensus for s and v from neighbors
        s_consensus = self.A_row[self.agent_id] * self.s
        v_consensus = self.A_row[self.agent_id] * self.v
        
        for j in self.neighbors:
            s_consensus += self.A_row[j] * self.neighbor_s[j]
            v_consensus += self.A_row[j] * self.neighbor_v[j]
        
        # Update s with innovation: s[k+1] = consensus + (z[k+1] - z[k])
        self.s = s_consensus + (self.z - z_old)
        
        # Update v with innovation: v[k+1] = consensus + (grad_2_next - grad_2)
        _, _, grad_2_next = CostFunction.distributed_aggregative(
            self.z, self.s, self.gamma, self.r0, self.target, self.d)
        
        self.v = v_consensus + (grad_2_next - grad_2)
        # ============================================================
        
        # Publish updated state: [id, iter, z, s, v]
        msg.data = [float(self.agent_id), float(self.k),
                    *self.z.tolist(), *self.s.tolist(), *self.v.tolist()]
        self.publisher.publish(msg)

        # Publish cost data: [id, iter, cost_i, grad_norm_i]
        total_grad = grad_1 + self.v  # Total gradient used in update
        grad_norm = np.linalg.norm(total_grad)

        cost_msg = MsgFloat()
        cost_msg.data = [float(self.agent_id), float(self.k), float(cost), float(grad_norm)]
        self.cost_publisher.publish(cost_msg)
        
        # Compute distance to target (for logging)
        dist_to_target = np.linalg.norm(self.z - self.target)

        print(f"Agent {self.agent_id} (k={self.k}): "
                f"x={self.z.round(3)}, dist_target={dist_to_target:.3f}")
        
        # Reset data received flags
        self.data_received = {j: False for j in self.neighbors}

        if self.k >= self.maxIters:
            print(f"\n Agent {self.agent_id}: Done!")
            print(f"   Final pos: {self.z.round(3)}")
            print(f"   Target: {self.target.round(3)}")
            print(f"   Error: {dist_to_target:.3f}\n")
            sleep(2)
            raise SystemExit
        
        self.k += 1 # Increment iteration counter


def main(args=None):
    rclpy.init(args=args)

    agent = Agent()
    
    sleep(2) # Give time to initialize publishers/subscribers

    try:
        rclpy.spin(agent)
    except SystemExit:
        rclpy.logging.get_logger("Quitting").info(f"Agent {agent.agent_id}: Shutting down")
    finally:
        agent.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
    