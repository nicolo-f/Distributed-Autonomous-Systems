from launch import LaunchDescription
from launch_ros.actions import Node
import numpy as np

from task2.helper import Digraph

np.random.seed(0)

def generate_launch_description():
    """
    Launch file for aggregative tracking formation control
    """
    
    # =======================
    # CONFIGURATION PARAMETERS
    # =======================

    # Control parameters
    MAXITERS = 1000
    COMM_TIME = 1e-3  # communication time period

    N = 6  # number of agents
    d = 2  # dimension of each x_i (2D positions)

    # Aggregative tracking parameters
    alpha = 1e-2        # step size
    gamma = 0.5         # trade-off parameter

    target_std = 2.0    # standard deviation for target generation

    p_er = 0.5  # Probability for Erdos-Renyi graph

    # =======================
    # INITIAL POSITIONS & TARGETS
    # =======================
    
    robot_positions = np.random.uniform(low=0, high=10, size=(N, d))
    target_positions = robot_positions + np.random.normal(0, target_std, size=(N, d))
    
    # Initialize r0 
    # r0 = robot_positions.copy()  # r0[i] = z[0, i]    # agent-dependent (each has own desired position)
    r0 = np.tile(np.mean(target_positions, axis=0), (N, 1))  # Shared barycenter (mean of targets)
    # r0 = np.full((N, d), [5.0, 5.0])  # All want same constant barycenter
    
    z_init = robot_positions.flatten()

    # =======================
    # COMMUNICATION GRAPH
    # =======================

    digraph = Digraph(N, p_er, 'random')
    A = digraph.get_weight_matrix()  # Get doubly stochastic weight matrix
    G = digraph.get_graph()  # Get NetworkX graph object

    # Extract adjacency matrix (binary connections)
    Adj = (A > 0).astype(bool)
    np.fill_diagonal(Adj, 0)  # Remove self-loops from adjacency

    # =======================
    # CREATE AGENT NODES
    # =======================
    node_list = []  # List to store all nodes

    for i in range(N):

        # Get initial position for agent i
        i_index = i * d + np.arange(d)
        z_init_i = z_init[i_index].flatten().tolist()

        # Create the node with EXACT same structure as formation_launch.py
        node_list.append(
            Node(
                package="task2",
                namespace=f"agent_{i}",
                executable="aggregative_agent",
                parameters=[
                    {
                        "id": i,
                        "N": N,  
                        "d": d,
                        "maxIters": MAXITERS,
                        "dt": COMM_TIME,
                        'alpha': alpha,
                        'gamma': gamma,
                        "A_row": A[i, :].tolist(),
                        "z0": z_init_i,
                        "target": target_positions[i].tolist(),
                        "r0": r0[i].tolist(), 
                    }
                ],
                output="screen",
                # Optional: Uncomment below to open each agent in separate xterm window
                prefix=f'xterm -geometry 120x50 -title "agent_{i}" -fg white -bg black -fs 12 -fa "Monospace" -hold -e',
                # prefix=f'terminator -T "agent_{i}" -e bash -c',
            )
        )

    return LaunchDescription(node_list)