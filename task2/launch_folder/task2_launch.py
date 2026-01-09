from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess, TimerAction, RegisterEventHandler
from launch.event_handlers import OnProcessStart
import numpy as np

from task2.helper import Digraph

np.random.seed(0)

def generate_launch_description():
    """Launch file for aggregative tracking with RViz2 visualization"""
    
    # Configuration parameters
    MAXITERS = 500 
    COMM_TIME = 1e-2

    N = 5
    d = 2

    alpha = 1e-2
    gamma = 0.8
    target_std = 2.0
    p_er = 0.5

    # Initialize robot and target positions
    robot_positions = np.random.uniform(low=0, high=10, size=(N, d))
    target_positions = robot_positions + np.random.normal(0, target_std, size=(N, d))
    z_init = robot_positions.flatten() 

    # Define desired r0

    # All agents share the same desired barycenter (mean of targets)
    # r0 = np.tile(np.mean(target_positions, axis=0), (N, 1))

    # All agents share a specific constant barycenter
    # r0 = np.full((N, d), [5.0, 5.0]) 

    # r0 is agent-dependent (each agent has its own desired position)
    r0 = robot_positions.copy()  # r0[i] = z[0, i]

    # Create strongly connected graph and the associated adjacency matrix
    digraph = Digraph(N, p_er, 'random')
    A = digraph.get_weight_matrix()
    G = digraph.get_graph()

    Adj = (A > 0).astype(bool)
    np.fill_diagonal(Adj, 0)

    # Configure nodes
    
    # RViz2 node
    rviz_config_file = "/home/mirco/das_ros2_ws/src/task2/resource/task2_config.rviz"
    
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_file],
        output='screen'
    )

    # Visualizer node
    vis_params = {
        "N": N,
        "d": d,
        "r0": np.mean(target_positions, axis=0).tolist(),
    }
    
    for i in range(N):
        vis_params[f"target_{i}"] = target_positions[i].tolist()
    
    visualizer_node = Node(
        package="task2",
        executable="task2_visualizer",
        name="task2_visualizer",
        parameters=[vis_params],
        output="screen",
    )

    # Plotter node
    plotter_node = Node(
        package="task2",
        executable="task2_plotter",
        name="task2_plotter",
        parameters=[
            {   "N": N,
                "dt": COMM_TIME,
            }
        ],
        output="screen",
    )

    # Agent nodes
    agent_nodes = []
    
    for i in range(N):
        i_index = i * d + np.arange(d)
        z_init_i = z_init[i_index].flatten().tolist()

        agent_node = Node(
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
            prefix=f'xterm -geometry 120x30 -title "agent_{i}" -fg white -bg black -fs 12 -fa "Monospace" -hold -e',
        )
        
        agent_nodes.append(agent_node)

    # Define launch sequence 

    # Nodes to be launched immediately
    launch_description = [
        rviz_node,
        visualizer_node,
        plotter_node,
    ]
    
    # Delayed start for agent nodes
    for agent_node in agent_nodes:
        delayed_agent = TimerAction(
            period=5.0,  # Wait 5 seconds before starting agents
            actions=[agent_node]
        )
        launch_description.append(delayed_agent)

    return LaunchDescription(launch_description)