import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray as MsgFloat
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from matplotlib.colors import to_rgb
import numpy as np
import os
from ament_index_python.packages import get_package_share_directory

class Task2Visualizer(Node):
    def __init__(self):
        super().__init__(
            "task2_visualizer",
            allow_undeclared_parameters=True,
            automatically_declare_parameters_from_overrides=True,
        )
        # Mesh related variables
        self.use_mesh = False
        self.mesh_path = "package://task2/resource/Quadcopter.stl"

        # Get parameters
        self.N = self.get_parameter("N").value   # Total number of agents
        self.d = self.get_parameter("d").value   # Dimension of decision variable
        
        # Get target positions
        self.targets = []
        for i in range(self.N):
            target = self.get_parameter(f"target_{i}").value
            self.targets.append(np.array(target))
        
        self.r0 = np.array(self.get_parameter("r0").value)  # Desired r0

        # Storage variable for agent data visualization 
        self.agent_positions = {i: np.zeros(self.d) for i in range(self.N)}
        self.agent_trajectories = {i: [] for i in range(self.N)}
        
        # Define a color for each agents
        self.colors = [to_rgb(f'C{i % 10}') for i in range(self.N)]

        # Check if mesh file exists
        try:
            pkg_share = get_package_share_directory('task2')
            mesh_file = os.path.join(pkg_share, 'resource', 'Quadcopter.stl')
            if os.path.exists(mesh_file):
                self.use_mesh = True
        except Exception as e:
            self.get_logger().warn(f'Could not locate mesh file: {e}, using spheres instead')

        # Define publishers and subscribers
        # Create listeners to receive data from all agents
        for i in range(self.N):
            self.create_subscription(
                MsgFloat,
                f"/data_{i}",
                lambda msg, agent_id=i: self.data_callback(msg, agent_id),
                10,
            )

        # Create publisher for visualization markers
        self.marker_pub = self.create_publisher(MarkerArray, "/visualization_marker_array", 10)
        
        self.timer = self.create_timer(0.1, self.publish_markers)   # Timer to publish markers

    def data_callback(self, msg, agent_id):
        """
        Receive agent data and store it
        Message format: [id, iter, z, s, v]
        """
        z = np.array(msg.data[2:2+self.d])
        self.agent_positions[agent_id] = z
        
        self.agent_trajectories[agent_id].append(z.copy())
        # Keep only last 500 positions for trajectory
        if len(self.agent_trajectories[agent_id]) > 500:
            self.agent_trajectories[agent_id].pop(0)

    def publish_markers(self):
        """Publish visualization markers"""
        marker_array = MarkerArray()
        marker_id = 0

        # Define AGENT POSITIONS (mesh or spheres)
        for i, pos in self.agent_positions.items():
            marker = Marker()
            marker.header.frame_id = "world"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "agents"
            marker.id = marker_id
            marker_id += 1
            marker.action = Marker.ADD
            
            # Position
            marker.pose.position.x = float(pos[0])
            marker.pose.position.y = float(pos[1])
            marker.pose.position.z = 0.0
            
            if self.use_mesh:
                # Use MESH (quadcopter file)
                marker.type = Marker.MESH_RESOURCE
                marker.mesh_resource = self.mesh_path
                
                # Orientation (top view require 90deg rotation around x-axis)
                angle = np.pi / 2  # 90 degrees in radians
                marker.pose.orientation.x = np.sin(angle / 2)
                marker.pose.orientation.y = 0.0
                marker.pose.orientation.z = 0.0
                marker.pose.orientation.w = np.cos(angle / 2)
                
                # Scale for mesh
                marker.scale.x = 0.003
                marker.scale.y = 0.003
                marker.scale.z = 0.003
            else:
                # Use SPHERE as fallback
                marker.type = Marker.SPHERE
                marker.pose.orientation.w = 1.0
                
                # Scale for sphere
                marker.scale.x = 0.3
                marker.scale.y = 0.3
                marker.scale.z = 0.3
            
            # Color per agent
            color = self.colors[i % len(self.colors)]
            marker.color.r = color[0]
            marker.color.g = color[1]
            marker.color.b = color[2]
            marker.color.a = 1.0
            
            marker_array.markers.append(marker)

        # Define TARGET POSITIONS (cyliniders)
        for i in range(self.N):
            marker = Marker()
            marker.header.frame_id = "world"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "targets"
            marker.id = marker_id
            marker_id += 1
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD
            
            marker.pose.position.x = float(self.targets[i][0])
            marker.pose.position.y = float(self.targets[i][1])
            marker.pose.position.z = 0.0
            marker.pose.orientation.w = 1.0
            
            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.01
            
            color = self.colors[i % len(self.colors)]
            marker.color.r = color[0]
            marker.color.g = color[1]
            marker.color.b = color[2]
            marker.color.a = 0.5
            
            marker_array.markers.append(marker)

        # Define AGENT TRAJECTORIES (line strips)
        for i in range(self.N):
            if len(self.agent_trajectories[i]) > 1:
                marker = Marker()
                marker.header.frame_id = "world"
                marker.header.stamp = self.get_clock().now().to_msg()
                marker.ns = "trajectories"
                marker.id = marker_id
                marker_id += 1
                marker.type = Marker.LINE_STRIP
                marker.action = Marker.ADD
                
                marker.scale.x = 0.05  # Line width
                
                color = self.colors[i % len(self.colors)]
                marker.color.r = color[0]
                marker.color.g = color[1]
                marker.color.b = color[2]
                marker.color.a = 0.6
                
                for pos in self.agent_trajectories[i]:
                    p = Point()
                    p.x = float(pos[0])
                    p.y = float(pos[1])
                    p.z = 0.0
                    marker.points.append(p)
                
                marker_array.markers.append(marker)

        self.marker_pub.publish(marker_array)   # Publish all markers


def main(args=None):
    rclpy.init(args=args)
    visualizer = Task2Visualizer()
    
    try:
        rclpy.spin(visualizer)
    except KeyboardInterrupt:
        pass
    finally:
        visualizer.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()