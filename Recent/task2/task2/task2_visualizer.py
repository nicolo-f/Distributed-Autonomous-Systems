import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray as MsgFloat
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
import numpy as np


class Task2Visualizer(Node):
    def __init__(self):
        super().__init__(
            "task2_visualizer",
            allow_undeclared_parameters=True,
            automatically_declare_parameters_from_overrides=True,
        )

        # Get parameters
        self.N = self.get_parameter("N").value
        self.d = self.get_parameter("d").value
        
        # Get target positions for visualization
        self.targets = []
        for i in range(self.N):
            target = self.get_parameter(f"target_{i}").value
            self.targets.append(np.array(target))
        
        # Get r0 (desired barycenter)
        self.r0 = np.array(self.get_parameter("r0").value)

        # Storage for agent positions
        self.agent_positions = {i: np.zeros(self.d) for i in range(self.N)}
        self.agent_trajectories = {i: [] for i in range(self.N)}
        
        # Colors for each agent (RGB)
        self.colors = [
            (1.0, 0.0, 0.0),  # Red
            (0.0, 1.0, 0.0),  # Green
            (0.0, 0.0, 1.0),  # Blue
            (1.0, 1.0, 0.0),  # Yellow
            (1.0, 0.0, 1.0),  # Magenta
            (0.0, 1.0, 1.0),  # Cyan
            (1.0, 0.5, 0.0),  # Orange
            (0.5, 0.0, 1.0),  # Purple
        ]

        # Subscribe to all agents
        for i in range(self.N):
            self.create_subscription(
                MsgFloat,
                f"/topic_{i}",
                lambda msg, agent_id=i: self.agent_callback(msg, agent_id),
                10,
            )

        # Publishers for visualization
        self.marker_pub = self.create_publisher(MarkerArray, "/visualization_marker_array", 10)
        
        # Timer for visualization updates
        self.timer = self.create_timer(0.1, self.publish_markers)

        print("\n" + "="*60)
        print("🎨 TASK2 VISUALIZER INITIALIZED")
        print("="*60)
        print(f"Number of agents: {self.N}")
        print(f"Dimension: {self.d}")
        print(f"Desired barycenter r0: {self.r0}")
        print("="*60 + "\n")

    def agent_callback(self, msg, agent_id):
        """Receive agent position and store it"""
        # Message format: [id, iter, z, s, v]
        z = np.array(msg.data[2:2+self.d])
        self.agent_positions[agent_id] = z
        
        # Store trajectory (limit to last 500 points)
        self.agent_trajectories[agent_id].append(z.copy())
        if len(self.agent_trajectories[agent_id]) > 500:
            self.agent_trajectories[agent_id].pop(0)

    def publish_markers(self):
        """Publish visualization markers"""
        marker_array = MarkerArray()
        marker_id = 0

        # 1. Agent positions (spheres)
        for i in range(self.N):
            marker = Marker()
            marker.header.frame_id = "world"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "agents"
            marker.id = marker_id
            marker_id += 1
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            
            marker.pose.position.x = float(self.agent_positions[i][0])
            marker.pose.position.y = float(self.agent_positions[i][1])
            marker.pose.position.z = 0.0
            marker.pose.orientation.w = 1.0
            
            marker.scale.x = 0.3
            marker.scale.y = 0.3
            marker.scale.z = 0.3
            
            color = self.colors[i % len(self.colors)]
            marker.color.r = color[0]
            marker.color.g = color[1]
            marker.color.b = color[2]
            marker.color.a = 1.0
            
            marker_array.markers.append(marker)
            
            # Agent label
            text_marker = Marker()
            text_marker.header.frame_id = "world"
            text_marker.header.stamp = self.get_clock().now().to_msg()
            text_marker.ns = "agent_labels"
            text_marker.id = marker_id
            marker_id += 1
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            
            text_marker.pose.position.x = float(self.agent_positions[i][0])
            text_marker.pose.position.y = float(self.agent_positions[i][1])
            text_marker.pose.position.z = 0.5
            text_marker.pose.orientation.w = 1.0
            
            text_marker.scale.z = 0.3
            text_marker.color.r = 1.0
            text_marker.color.g = 1.0
            text_marker.color.b = 1.0
            text_marker.color.a = 1.0
            
            text_marker.text = f"Agent {i}"
            marker_array.markers.append(text_marker)

        # 2. Target positions (stars)
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

        # 3. Agent trajectories (line strips)
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

        # 4. Current barycenter (computed from agent positions)
        current_barycenter = np.mean(list(self.agent_positions.values()), axis=0)
        marker = Marker()
        marker.header.frame_id = "world"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "barycenter"
        marker.id = marker_id
        marker_id += 1
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        
        marker.pose.position.x = float(current_barycenter[0])
        marker.pose.position.y = float(current_barycenter[1])
        marker.pose.position.z = 0.3
        marker.pose.orientation.w = 1.0
        
        marker.scale.x = 0.4
        marker.scale.y = 0.4
        marker.scale.z = 0.4
        
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 0.8
        
        marker_array.markers.append(marker)

        # 5. Desired barycenter r0 (fixed)
        marker = Marker()
        marker.header.frame_id = "world"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "desired_barycenter"
        marker.id = marker_id
        marker_id += 1
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        
        marker.pose.position.x = float(self.r0[0])
        marker.pose.position.y = float(self.r0[1])
        marker.pose.position.z = 0.0
        marker.pose.orientation.w = 1.0
        
        marker.scale.x = 0.5
        marker.scale.y = 0.5
        marker.scale.z = 0.1
        
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 0.5
        
        marker_array.markers.append(marker)

        # Publish all markers
        self.marker_pub.publish(marker_array)


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