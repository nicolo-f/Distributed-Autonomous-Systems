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
        self.agent_id = self.get_parameter("id").value
        self.N = self.get_parameter("N").value  # Total number of agents
        self.d = self.get_parameter("d").value

        self.maxIters = self.get_parameter("maxIters").value
        dt = self.get_parameter("dt").value

        self.target = np.array(self.get_parameter("target").value)
        self.neighbors = self.get_parameter("neighbors").value
        self.z = np.array(self.get_parameter("z0").value)
        self.r0 = self.get_parameter("r0").value  # Desired barycenter contribution
        
        self.alpha = self.get_parameter("alpha").value
        self.gamma = self.get_parameter("gamma").value

        # Get FULL row of weight matrix A[i, :]
        self.A_row = np.array(self.get_parameter("A_row").value)

        # Extract neighbors (non-zero elements, excluding self-loop)
        self.neighbors = [j for j in range(self.N) if self.A_row[j] > 0 and j != self.agent_id]
        
        # Initialize aggregate variable (sigma)
        self.s = self.z.copy()  

        # Initialize v[0, i] using distributed_aggregative
        _, _, self.v = CostFunction.distributed_aggregative(
            self.z, self.s, self.gamma, self.r0, self.target, self.d, self.N
        )
        
        # Instead of zeros, use own initial values (assumes neighbors start similarly)
        self.neighbor_z = {j: np.zeros(self.d) for j in self.neighbors}
        self.neighbor_s = {j: np.zeros(self.d) for j in self.neighbors}
        self.neighbor_v = {j: np.zeros(self.d) for j in self.neighbors}
        self.data_received = {j: False for j in self.neighbors} 

        self.k = 0
        self.stopped = False

        # Print initialization
        print("\n" + "="*60)
        print(f"🤖 AGENT {self.agent_id} INITIALIZED")
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

        # Subscribe to neighbors
        for j in self.neighbors:
            self.create_subscription(
                MsgFloat,
                f"/topic_{j}",
                self.listener_callback,
                10,
            )

        # Publisher
        self.publisher = self.create_publisher(
            MsgFloat,
            f"/topic_{self.agent_id}",
            10,
        )

        # ✅ NEW: Publisher for cost data
        self.cost_publisher = self.create_publisher(
            MsgFloat,
            f"/cost_{self.agent_id}",
            10,
        )

        # ✅ ADD: Subscribe to stop signal
        self.create_subscription(
            MsgFloat,
            "/stop_signal",
            self.stop_callback,
            10
        )

        # Timer
        self.timer = self.create_timer(dt, self.timer_callback)

    def listener_callback(self, msg):
        """Store received neighbor data"""
        n_id = int(msg.data[0])
        iteration = int(msg.data[1])

        # Parse data into CLEAR variable names (like Nico!)
        self.neighbor_z[n_id] = np.array(msg.data[2:2+self.d])
        self.neighbor_s[n_id] = np.array(msg.data[2+self.d:2+2*self.d])
        self.neighbor_v[n_id] = np.array(msg.data[2+2*self.d:2+3*self.d])
        self.data_received[n_id] = True

    def stop_callback(self, msg):
        """Receive stop signal from plotter"""
        self.stopped = True
        print(f"\n🛑 Agent {self.agent_id}: Received STOP signal - algorithm converged!")
        print(f"   Final position: {self.z.round(3)}")
        print(f"   Target: {self.target.round(3)}")
        print(f"   Final iteration: {self.k}")
        print(f"   Distance to target: {np.linalg.norm(self.z - self.target):.6f}\n")

    def timer_callback(self):
        """Main control loop"""
        

        # ✅ ADD: Check if stopped
        if self.stopped:
            return  # Stop processing but keep node alive
        
        msg = MsgFloat()

        # if self.k == 0:
        #     # Publish initial state: [id, iter, z, s, v]
        #     msg.data = [float(self.agent_id), float(self.k), 
        #                *self.z.tolist(), *self.s.tolist(), *self.v.tolist()]
        #     self.publisher.publish(msg)
            
        #     print(f"Agent {self.agent_id} (k={self.k}): x={self.z.round(3)}, s={self.s.round(3)}")

        #     # Reset data received flags
        #     self.data_received = {j: False for j in self.neighbors}
        #     self.k += 1

        # else:
            
        # Wait for all neighbor data (skip first few iterations)
        if self.k > 0 and len(self.neighbors) > 0:
            if not all(self.data_received.values()):
                return

        # ============================================================
        # EXACT Task2.1 Algorithm (lines 68-80)
        # ============================================================
        
        # Store old z for innovation term
        z_old = self.z.copy()

        # Step 1: Compute cost and gradients at current state
        cost, grad_1, grad_2 = CostFunction.distributed_aggregative(
            self.z, self.s, self.gamma, self.r0, self.target, self.d, self.N
        )

        # Step 2: Update position z[k+1] = z[k] - alpha*(grad_1 + v[k])
        self.z = self.z - self.alpha * (grad_1 + self.v)
        
        # Step 3: Weighted consensus for s and v using A_row
        s_consensus = self.A_row[self.agent_id] * self.s
        v_consensus = self.A_row[self.agent_id] * self.v
        
        for j in self.neighbors:
            s_consensus += self.A_row[j] * self.neighbor_s[j]
            v_consensus += self.A_row[j] * self.neighbor_v[j]
        
        # Step 4: Update s with innovation: s[k+1] = consensus + (z[k+1] - z[k])
        self.s = s_consensus + (self.z - z_old)
        
        # Step 5: Compute new grad_2 for gradient tracking
        _, _, grad_2_next = CostFunction.distributed_aggregative(
            self.z, self.s, self.gamma, self.r0, self.target, self.d, self.N
        )
        
        # Step 6: Update v with innovation: v[k+1] = consensus + (grad_2_next - grad_2)
        self.v = v_consensus + (grad_2_next - grad_2)

        # ============================================================
        
        # Publish updated state: [id, iter, z, s, v]
        msg.data = [float(self.agent_id), float(self.k),
                    *self.z.tolist(), *self.s.tolist(), *self.v.tolist()]
        self.publisher.publish(msg)

        # ✅ Publish cost data: [id, iter, cost_i, grad_norm_i]
        total_grad = grad_1 + self.v  # Total gradient used in update
        grad_norm = np.linalg.norm(total_grad)

        cost_msg = MsgFloat()
        cost_msg.data = [float(self.agent_id), float(self.k), float(cost), float(grad_norm)]
        self.cost_publisher.publish(cost_msg)
        
        # Compute distance to target
        dist_to_target = np.linalg.norm(self.z - self.target)

        print(f"Agent {self.agent_id} (k={self.k}): "
                f"x={self.z.round(3)}, dist_target={dist_to_target:.3f}")
        
        # Reset data received flags
        self.data_received = {j: False for j in self.neighbors}
        self.k += 1

        if self.k > self.maxIters:
            print(f"\n🏁 Agent {self.agent_id}: Done!")
            print(f"   Final pos: {self.z.round(3)}")
            print(f"   Target: {self.target.round(3)}")
            print(f"   Error: {dist_to_target:.3f}\n")
            sleep(2)
            raise SystemExit


def main(args=None):
    rclpy.init(args=args)

    # agent = TestAgent()
    agent = Agent()
    
    agent.get_logger().info(f"Agent {agent.agent_id}: Waiting for synchronization...")
    sleep(1)
    agent.get_logger().info(f"Agent {agent.agent_id}: GO! 🚀")

    try:
        rclpy.spin(agent)
    except SystemExit:
        rclpy.logging.get_logger("Quitting").info(f"Agent {agent.agent_id}: Shutting down ✓")
    finally:
        agent.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
    