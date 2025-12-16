import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray as MsgFloat
from time import sleep
import numpy as np

from task2.helper import CostFunction


class TestAgent(Node):
    def __init__(self):
        super().__init__(
            "agent",
            allow_undeclared_parameters=True,
            automatically_declare_parameters_from_overrides=True,
        )

        # Get parameters from the launch file
        self.agent_id = self.get_parameter("id").value
        self.neighbors = self.get_parameter("neighbors").value
        z = self.get_parameter("xzero").value
        dist = self.get_parameter("dist").value
        self.maxIters = self.get_parameter("maxT").value
        dt = self.get_parameter("dt").value

        self.z = z
        self.dist_ii = dist
        self.k = 0

        # Print agent information
        print("\n" + "="*60)
        print(f"🤖 AGENT {self.agent_id} INITIALIZED")
        print("="*60)
        print(f"Initial position: {self.z}")
        print(f"Neighbors: {self.neighbors}")
        print(f"Desired distances: {self.dist_ii}")
        print(f"Max iterations: {self.maxIters}")
        print(f"Communication time: {dt} seconds")
        print("="*60 + "\n")

        # Subscribe to neighbors' topics
        for j in self.neighbors:
            self.create_subscription(
                MsgFloat,
                f"/topic_{j}",
                self.listener_callback,
                10,
            )
            print(f"Agent {self.agent_id}: ✓ Subscribed to /topic_{j}")

        # Initialize an empty dictionary to store messages from each neighbor
        self.received_data = {j: [] for j in self.neighbors}

        # Create publisher for this agent's topic
        self.publisher = self.create_publisher(
            MsgFloat,
            f"/topic_{self.agent_id}",
            10,
        )
        print(f"Agent {self.agent_id}: ✓ Created publisher /topic_{self.agent_id}")

        # Create timer for periodic communication
        self.timer = self.create_timer(dt, self.timer_callback)

        print(f"\nAgent {self.agent_id}: 🚀 Setup completed! Ready to communicate.\n")

    def listener_callback(self, msg):
        """
        Callback when a message arrives from a neighbor
        """
        j = int(msg.data[0])  # sender agent ID
        iteration = int(msg.data[1])  # iteration number
        position = list(msg.data[2:])  # position data

        # Store the received message
        self.received_data[j].append([iteration, *position])

        # print(f"Agent {self.agent_id}: 📨 Received from Agent {j} (iter {iteration}): position = {position}")

        return None

    def timer_callback(self):
        """
        Periodic callback to publish own state and check received messages
        """
        msg = MsgFloat()

        if self.k == 0:
            # First iteration: just publish initial state
            msg.data = [float(self.agent_id), float(self.k), *self.z]
            self.publisher.publish(msg)
            
            # print(f"\n📤 Agent {self.agent_id} (iter {self.k}): Publishing initial position {self.z}")
            
            self.k += 1

        else:
            # Check if we received messages from all neighbors
            all_received = False
            if all(len(self.received_data[j]) > 0 for j in self.neighbors):
                all_received = all(
                    self.k - 1 == self.received_data[j][0][0] for j in self.neighbors
                )

            if all_received:
                print(f"\n✅ Agent {self.agent_id} (iter {self.k}): Received from ALL neighbors!")
                
                                # Show what was received
                for j in self.neighbors:
                    neighbor_data = self.received_data[j][0]
                    neighbor_pos = neighbor_data[1:]
                    desired_dist = self.dist_ii[j]  # Access distance using neighbor index
                    print(f"   - Neighbor {j}: pos={neighbor_pos}, desired_dist={desired_dist}")
       
                # Clear the buffer (consume the message)
                for j in self.neighbors:
                    self.received_data[j].pop(0)
                
                # Publish current state
                msg.data = [float(self.agent_id), float(self.k), *self.z]
                self.publisher.publish(msg)
                
                print(f"📤 Agent {self.agent_id} (iter {self.k}): Publishing position {self.z}")
                
                # Update iteration counter
                self.k += 1

                # Stop if MAXITERS is exceeded
                if self.k > self.maxIters:
                    print(f"\n🏁 Agent {self.agent_id}: Max iterations reached ({self.maxIters})")
                    sleep(2)
                    raise SystemExit
            else:
                # Still waiting for some neighbors
                waiting_for = [j for j in self.neighbors if len(self.received_data[j]) == 0]
                if waiting_for:
                    print(f"⏳ Agent {self.agent_id} (iter {self.k}): Waiting for neighbors {waiting_for}...")

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
        self.neighbor_z = {j: self.z.copy() for j in self.neighbors}
        self.neighbor_s = {j: self.s.copy() for j in self.neighbors}
        self.neighbor_v = {j: self.v.copy() for j in self.neighbors}
        self.neighbor_iter = {j: 0 for j in self.neighbors}  # ✅ Start at 0, not -1

        self.k = 0

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

        # Timer
        self.timer = self.create_timer(dt, self.timer_callback)

    def listener_callback(self, msg):
        """Store received neighbor data"""
        j = int(msg.data[0])
        iteration = int(msg.data[1])

        # Parse data into CLEAR variable names (like Nico!)
        self.neighbor_z[j] = np.array(msg.data[2:2+self.d])
        self.neighbor_s[j] = np.array(msg.data[2+self.d:2+2*self.d])
        self.neighbor_v[j] = np.array(msg.data[2+2*self.d:2+3*self.d])
        self.neighbor_iter[j] = iteration

    def timer_callback(self):
        """Main control loop"""
        msg = MsgFloat()

        if self.k == 0:
            # Publish initial state: [id, iter, z, s, v]
            msg.data = [float(self.agent_id), float(self.k), 
                       *self.z.tolist(), *self.s.tolist(), *self.v.tolist()]
            self.publisher.publish(msg)
            
            print(f"Agent {self.agent_id} (k={self.k}): x={self.z.round(3)}, s={self.s.round(3)}")
            self.k += 1

        else:
            # Check if all neighbors have sent data for iteration k-1
            if len(self.neighbors) == 0:
                all_received = True
            else:
                all_received = all(self.neighbor_iter[j] >= self.k - 1 for j in self.neighbors)

            if all_received:
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
                
                # Compute distance to target
                dist_to_target = np.linalg.norm(self.z - self.target)

                print(f"Agent {self.agent_id} (k={self.k}): "
                      f"x={self.z.round(3)}, dist_target={dist_to_target:.3f}")
                
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
    