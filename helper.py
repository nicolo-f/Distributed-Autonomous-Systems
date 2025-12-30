import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image, ImageEnhance

class Digraph:
    """Class to manage graph creation and weight computation"""
    
    def __init__(self, N, p_er=0.5, graph_type='random'):
        """Initialize graph with N nodes and double stochastic weights"""
        self.N = N
        self.p_er = p_er # probability for Erdos-Renyi graph
        self.graph_type = graph_type
        self.A = None
        self.G = None
        self.create_graph()
    
    def create_graph(self):
        """Create graph based on specified type"""
        if self.graph_type == 'cycle':
            self.G = nx.cycle_graph(self.N)
            Adj = nx.adjacency_matrix(self.G).toarray()
        elif self.graph_type == 'path':
            self.G = nx.path_graph(self.N)
            Adj = nx.adjacency_matrix(self.G).toarray()
        elif self.graph_type == 'star':
            self.G = nx.star_graph(self.N - 1)
            Adj = nx.adjacency_matrix(self.G).toarray()
        elif self.graph_type == 'random':
            while True:
                self.G = nx.erdos_renyi_graph(self.N, self.p_er)
                Adj = nx.adjacency_matrix(self.G).toarray() # Unweighted adjacency matrix without self-loops
                test = np.linalg.matrix_power(Adj + np.eye(self.N), self.N)
                if np.all(test > 0):  # check strong connectivity
                    break
        else:
            raise ValueError("Unknown topology")
        
        A_unweighted = Adj + np.eye(self.N) # add self-loops
        self.A = self.metropolis_hastings_weights(A_unweighted)
        self.G = nx.from_numpy_array(self.A)  #  G is updated to include self-loops for visualization purposes
    
    def metropolis_hastings_weights(self, A):
        """Compute Metropolis-Hastings weights"""
        deg = np.sum(A, axis=1) # Degree of each node
        A_mh = np.zeros((self.N, self.N)) 
        
        for i in range(self.N): 
            for j in range(self.N): # Compute weights according to Metropolis-Hastings rule
                if A[i, j] == 1 and i != j:
                    A_mh[i, j] = 1.0 / (1 + max(deg[i], deg[j])) 
            
            A_mh[i, i] = 1.0 - np.sum(A_mh[i, :]) # Compute weights for self-loops
            # Self-loops and non-neighbors are considered in the sum but do not affect the results because equals to zero
        
        return A_mh
    
    def get_weight_matrix(self):
        """Return weight matrix A"""
        return self.A
    
    def get_graph(self):
        """Return graph object"""
        return self.G


class CostFunction:
    """Class to manage cost function computations"""

    def quadratic(z, Q, r): 
        """ Compute quadratic cost and gradient """
        val = 0.5 * z.T @ Q @ z + r.T @ z
        grad = Q @ z + r
        return val, grad
    
    def target_localization(z, distances, robot_pos, d, NT):
        """ Compute cost and gradient for target localization """
        z_reshaped = z.reshape((NT, d))
        cost = 0.0
        grad = np.zeros((NT, d))
        
        for tau in range(NT):
            # Compute ||z_τ - p_i||² (squared distance from estimated target to robot)
            diff = z_reshaped[tau] - robot_pos
            squared_dist = np.linalg.norm(diff)**2
            # Residual: d²_iτ - ||z_τ - p_i||²
            residual = distances[tau]**2 - squared_dist
            cost += residual**2
            
            # Gradient: ∂/∂z_τ [(d²_iτ - ||z_τ - p_i||²)²]
            # = 2 * residual * ∂/∂z_τ [-(||z_τ - p_i||²)]
            # = 2 * residual * (-2) * (z_τ - p_i)
            # = -4 * residual * (z_τ - p_i)
            grad[tau] = -4 * residual * diff
    
        return cost, grad.flatten()
    
    def distributed_aggregative(z, bary, gamma, r0, r, d ,N):
        """ Compute cost and gradient for distributed aggregative formation control """
        cost = 0.0
        grad_1 = np.zeros((d))
        grad_2 = np.zeros((d))

        target_dist = z - r
        bary_dist = bary - r0
        # cost += gamma * (np.linalg.norm(target_dist))**2 + (np.linalg.norm(bary_dist))**2
        cost += gamma * (np.linalg.norm(target_dist))**2 + (1-gamma)*(np.linalg.norm(bary_dist))**2
        
        # # Gradient computation (Our calculations)
        # grad_1 = 2*gamma*target_dist + (2.0/N)*bary_dist
        # grad_2 = (2*bary_dist)
        
        # grad_1 = 2*gamma*target_dist + (1-gamma)*(2.0/N)*bary_dist
        # grad_2 = (1-gamma)*(2*bary_dist) 

        # # Gradient computation (other solution)
        grad_1 = 2 * gamma * target_dist
        grad_2 = 2 * (1-gamma) * (1.0/N) * bary_dist

        return cost, grad_1, grad_2
    
    def distributed_aggregative_barrier(z, bary, gamma, r0, r, d ,N, z_all, mu=1.0, threshold=1.0):
        """ Compute cost and gradient for distributed aggregative formation control """
        cost = 0.0
        barrier_cost = 0.0

        delta = 10.0 # formation keeping weight (used for testing)

        grad_1 = np.zeros((d))
        grad_2 = np.zeros((d))
        barrier_grad = np.zeros((d))

        target_dist = z - r
        bary_dist = bary - r0

        for j in range(len(z_all)):
            z_j = z_all[j]
            
            # Skip self
            if np.allclose(z, z_j):
                continue
            
            # Compute squared distance
            diff = z - z_j
            dist_squared = np.linalg.norm(diff)**2
            
            # Barrier argument: ||z_i - z_j||^2 - threshold^2 
            barrier_arg = dist_squared - threshold**2

            barrier_cost += -np.log(barrier_arg)
            barrier_grad += -2 * diff / barrier_arg

        # cost += gamma * (np.linalg.norm(target_dist))**2 + (1-gamma)*(np.linalg.norm(bary_dist))**2 + mu * barrier_cost
        cost += gamma * (np.linalg.norm(target_dist))**2 + delta*(np.linalg.norm(bary_dist))**2 + mu * barrier_cost
        
        # Gradient computation (other solution)
        grad_1 = 2 * gamma * target_dist + mu * barrier_grad
        # grad_2 = 2 * (1-gamma) * (1.0/N) * bary_dist
        grad_2 = 2 * delta * (1.0/N) * bary_dist

        # Barrier function for collision avoidance
        
        

        return cost, grad_1, grad_2


class Plotter:
    """Class to manage all plotting functions"""
    
    def __init__(self, N, d, NT):
        self.N = N
        self.d = d
        self.NT = NT
    
    def plot_graph_and_weights(self, G, A):
        """Plot network graph and weight matrix"""
        fig, axes = plt.subplots(figsize=(10, 5), nrows=1, ncols=2)
        
        # Graph drawing
        ax = axes[0]
        nx.draw_kamada_kawai(G, with_labels=True, ax=ax)   # spring layout for better visualization
        ax.set_title('Network Graph')
        
        # Weight matrix A as annotated heatmap
        ax = axes[1]
        im = ax.matshow(A, cmap='Blues', aspect='auto')
        ax.set_title('Weight Matrix A', pad=20)
        ax.set_xlabel('Agent j')
        ax.set_ylabel('Agent i')
        ax.xaxis.set_ticks_position('bottom')
        ax.set_xticks(range(self.N))
        ax.set_yticks(range(self.N))
        
        # Add A matrix text annotations
        for i in range(self.N):
            for j in range(self.N):
                ax.text(j, i, f'{A[i, j]:.2f}',
                       ha="center", va="center", color="black", fontsize=8)
        
        plt.tight_layout()
        return fig
    
    def plot_cost_and_consensus(self, cost, z, maxIters):
        """Plot cost evolution and consensus error"""

        fig, axes = plt.subplots(figsize=(10, 5), nrows=1, ncols=2)
        
        z_avg = np.mean(z, axis=1)   # average estimate across agents at each iteration
        
        # Cost evolution
        ax = axes[0]
        ax.plot(np.arange(maxIters - 1), cost[:-1])
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Cost')
        ax.set_title('Cost Evolution')
        ax.grid(True)
        
        # Consensus error
        ax = axes[1]
        for i in range(self.N):
            ax.semilogy(np.arange(maxIters), 
                       np.linalg.norm(z[:, i] - z_avg, axis=1), 
                       label=f'Agent {i}')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('||z_i - z_avg||')
        ax.set_title('Consensus Error')
        ax.grid(True)
        ax.legend()
        
        plt.tight_layout()
        return fig
    
    def plot_gradient_norms(self, grad_norm, maxIters):
        """Plot individual and total gradient norms"""

        fig, axes = plt.subplots(figsize=(10, 5), nrows=1, ncols=2)
        
        # Individual gradient norms
        ax = axes[0]
        for i in range(self.N):
            ax.semilogy(np.arange(maxIters), grad_norm[:, i], label=f'Agent {i}')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('||∇f_i(z_i)||')
        ax.set_title('Individual Gradient Norms')
        ax.grid(True)
        
        # Total gradient norm
        ax = axes[1]
        total_grad_norm = np.sum(grad_norm, axis=1)
        ax.semilogy(np.arange(maxIters), total_grad_norm)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Σ||∇f_i(z_i)||')
        ax.set_title('Total Gradient Norm')
        ax.grid(True)
        
        plt.tight_layout()
        return fig
    
    def plot_target_estimation(self, z, true_targets, maxIters):
        """Plot target position estimation error and evolution"""

        fig, axes = plt.subplots(figsize=(10, 5), nrows=1, ncols=2)
        
        # Compute estimation errors
        target_errors = np.zeros((maxIters, self.NT))
        for k in range(maxIters):
            z_mean_k = np.mean(z[k, :, :], axis=0)
            z_mean_reshaped = z_mean_k.reshape((self.NT, self.d))
            
            for tau in range(self.NT):
                target_errors[k, tau] = np.linalg.norm(
                    z_mean_reshaped[tau] - true_targets[tau]
                )
        
        # Plot error evolution
        ax = axes[0]
        for tau in range(self.NT):
            ax.semilogy(np.arange(maxIters), target_errors[:, tau], 
                       label=f'Target {tau}')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('||z̄_τ - p*_τ|| (Estimation Error)')
        ax.set_title('Target Position Estimation Error Evolution')
        ax.legend()
        ax.grid(True)
        
        # Plot position evolution
        ax = axes[1]
        for tau in range(self.NT):
            x_coords = np.zeros(maxIters)
            y_coords = np.zeros(maxIters)
            
            for k in range(maxIters):
                z_mean_k = np.mean(z[k, :, :], axis=0)
                z_mean_reshaped = z_mean_k.reshape((self.NT, self.d))
                x_coords[k] = z_mean_reshaped[tau, 0]
                y_coords[k] = z_mean_reshaped[tau, 1]
            
            color = f'C{tau}'
            
            # Plot trajectory
            ax.plot(x_coords, y_coords, '-', color=color, 
                   label=f'Target {tau} estimate', alpha=0.7)
            
            # Mark initial and final estimates
            ax.scatter(x_coords[0], y_coords[0], s=100, marker='o', 
                      color=color, edgecolors='black', linewidths=1, zorder=3)
            ax.scatter(x_coords[-1], y_coords[-1], s=100, marker='x', 
                      color=color, linewidths=2, zorder=3)
            
            # Plot true target position
            ax.axhline(y=true_targets[tau, 1], color=color, linestyle='--', 
                      alpha=0.5, linewidth=2)
            ax.axvline(x=true_targets[tau, 0], color=color, linestyle='--', 
                      alpha=0.5, linewidth=2)
            ax.scatter(true_targets[tau, 0], true_targets[tau, 1], 
                      s=200, marker='*', color=color, 
                      edgecolors='black', linewidths=1.5, zorder=5,
                      label=f'Target {tau} true')
        
        ax.set_xlabel('X position')
        ax.set_ylabel('Y position')
        ax.set_title('Target Position Estimates Evolution')
        ax.legend()
        ax.grid(True) 
        ax.axis('equal')
        
        plt.tight_layout()
        return fig
    
    def plot_robot_trajectories(self, z, robot_positions, final_positions, target_positions, 
                                               final_barycenter):
        """Plot robot trajectories and barycenter estimation error"""

        # For aggregative problem, we can visualize robot trajectories
        fig, ax = plt.subplots(figsize=(10, 5), nrows=1, ncols=1)

        for i in range(self.N):
            # Get color for this robot
            color = f'C{i}'
            
            # Plot trajectory 
            ax.plot(z[:, i, 0], z[:, i, 1], '-', color=color, alpha=0.5, label=f'Robot {i}')
            
            # Initial position (circle) 
            ax.scatter(robot_positions[i, 0], robot_positions[i, 1], s=100, marker='o', 
                    color=color, edgecolors='black', linewidths=1)
            
            # Final position (cross) 
            ax.scatter(final_positions[i, 0], final_positions[i, 1], s=100, marker='x', 
                    color=color, linewidths=2)
            
            # Target position (star)
            ax.scatter(target_positions[i, 0], target_positions[i, 1], s=150, marker='*', 
                    color=color, edgecolors='black', linewidths=1)

        # # Desired barycenter (red pentagon)
        # ax.scatter(r0[0], r0[1], s=300, marker='P', color='red', 
        #           edgecolors='black', linewidths=2, label='Desired barycenter', zorder=10)

        # Final barycenter (green diamond)
        ax.scatter(final_barycenter[0], final_barycenter[1], s=200, marker='D', 
                color='green', edgecolors='black', linewidths=2, label='Final barycenter', zorder=10)

        ax.set_xlabel('X position')
        ax.set_ylabel('Y position')
        ax.set_title('Robot Trajectories')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True)
        ax.axis('equal')

        plt.tight_layout()
        return fig
    
    def plot_robot_animation(self, z, robot_positions, target_positions, maxIters, dt=0.01, save=False, gif_name='robot_animation',
                              threshold=1.0, use_images=False, robot_image_path=None, image_zoom=0.05):
        """Create an animated plot showing robot trajectories and moving robots"""

        # Compute barycenter at each iteration
        barycenter_trajectory = np.mean(z, axis=1)  
        target_barycenter = np.mean(target_positions, axis=0)  

        # Create figure with grid layout
        fig = plt.figure(figsize=(17, 8))
        gs = fig.add_gridspec(self.N, 2, width_ratios=[1, 1])
        
        time_pointers = []
        time_vector = np.arange(maxIters) * dt
        
        # Individual robot trajectory plots
        for i in range(self.N):
            ax = fig.add_subplot(gs[i, 0])
            time_pointer = ax.axvline(x=0, color='black', linestyle='--', label='_nolegend_')
            time_pointers.append(time_pointer)
            
            # Plot x and y trajectories
            ax.plot(time_vector, z[:, i, 0], 'b-', linewidth=1, alpha=0.7, label='x-pos' if i == 0 else "")
            ax.plot(time_vector, z[:, i, 1], 'r-', linewidth=1, alpha=0.7, label='y-pos' if i == 0 else "")
            
            # Plot target positions as horizontal lines
            ax.axhline(y=target_positions[i, 0], color='b', linestyle='--', alpha=0.5, linewidth=1)
            ax.axhline(y=target_positions[i, 1], color='r', linestyle='--', alpha=0.5, linewidth=1)
            
            ax.grid(True, alpha=0.3)
            if i == self.N - 1:
                ax.set_xlabel('Time (s)')
            ax.set_ylabel(f'Robot {i}')
            if i == 0:
                ax.set_title('Robot Position Evolution')
                ax.legend(['', 'x-pos', 'y-pos'], loc='upper right', ncol=3, fontsize=8)
        
        # 2D animation of robots moving
        ax2 = fig.add_subplot(gs[:, 1])
        
        # Set axis limits with some padding
        all_positions = np.vstack([z.reshape(-1, self.d), robot_positions, target_positions, 
                                   barycenter_trajectory, [target_barycenter]])
        x_min, x_max = all_positions[:, 0].min(), all_positions[:, 0].max()
        y_min, y_max = all_positions[:, 1].min(), all_positions[:, 1].max()
        padding = 0.1 * max(x_max - x_min, y_max - y_min)
        
        ax2.set_xlim(x_min - padding, x_max + padding)
        ax2.set_ylim(y_min - padding, y_max + padding)
        ax2.set_xlabel('X Position')
        ax2.set_ylabel('Y Position')
        ax2.set_title('Robot Animation')
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect('equal')
        
        # Plot static elements (trajectories and targets)
        for i in range(self.N):
            color = f'C{i}'
            # Full trajectory (faded)
            ax2.plot(z[:, i, 0], z[:, i, 1], '-', color=color, alpha=0.2, linewidth=1)
            
            # Target positions (stars)
            ax2.scatter(target_positions[i, 0], target_positions[i, 1], 
                       s=150, marker='*', color=color, edgecolors='black', 
                       linewidths=1, zorder=5, alpha=0.7)
        
        # Plot barycenter trajectory (faded gray line)
        ax2.plot(barycenter_trajectory[:, 0], barycenter_trajectory[:, 1], 
                '--', color='gray', alpha=0.3, linewidth=2, label='Barycenter path')
        
        # Target barycenter (big black star with gold edge)
        ax2.scatter(target_barycenter[0], target_barycenter[1], 
                   s=400, marker='*', color='black', edgecolors='gold', 
                   linewidths=2, label='Target barycenter', zorder=15)
                
        # Initialize animated elements
        robot_artists = []  # List of robot markers (images or dots)
        trajectory_lines = [] 
        collision_circles = []  # List of collision avoidance circles
        
        # Load robot image if using images
        robot_img = []
        if use_images and robot_image_path:
            try:
                base_image = Image.open(robot_image_path)
                print(f"Loaded robot image from: {robot_image_path}")

                # Create colored version for each robot
                for i in range(self.N):
                    # Get RGB color from matplotlib color string
                    color = f'C{i}'
                    # Convert hex to RGB (0-1 range)
                    from matplotlib.colors import to_rgb
                    rgb_color = to_rgb(color)
                    
                    # Colorize the image
                    colored_image = self.color_image(base_image, rgb_color)
                    robot_img.append(colored_image)
                    
                print(f"Created {len(robot_img)} colored drone images")
            except Exception as e:
                print(f"Warning: Could not load image '{robot_image_path}': {e}")
                print("Falling back to dot markers")
                use_images = False
                robot_img = [None] * self.N

        for i in range(self.N):
            color = f'C{i}'
            
            if use_images and robot_img and robot_img[i] is not None:
                # Use the SPECIFIC colored image for this robot
                imagebox = OffsetImage(robot_img[i], zoom=image_zoom)
                ab = AnnotationBbox(imagebox, (robot_positions[i, 0], robot_positions[i, 1]), frameon=False, 
                                   xycoords='data', box_alignment=(0.5, 0.5),
                                   pad=0, zorder=6)
                ax2.add_artist(ab)
                robot_artists.append(ab)
            else:
                # Use dot marker
                dot, = ax2.plot([], [], 'o', color=color, markersize=10, 
                               markeredgecolor='black', markeredgewidth=1, zorder=6)
                robot_artists.append(dot)
            
            # Trajectory line for each robot (regardless of using images or dots)
            line, = ax2.plot([], [], '-', color=color, linewidth=2, alpha=0.8)
            trajectory_lines.append(line)

            # Create collision avoidance circle for each robot
            circle = plt.Circle((robot_positions[i, 0], robot_positions[i, 1]), 
                            threshold, color=color, fill=False, 
                            linestyle='--', linewidth=1.5, alpha=0.5, zorder=4)
            ax2.add_patch(circle)
            collision_circles.append(circle)

        # Add animated barycenter marker (black X)
        barycenter_marker, = ax2.plot([], [], 'D', color='black', markersize=7, 
                                      markeredgewidth=3, zorder=20, 
                                      label='Current barycenter')
        
        # Add animated barycenter trajectory line
        barycenter_line, = ax2.plot([], [], '-', color='black', 
                                    linewidth=2, alpha=0.6, zorder=19)
        
        time_text = ax2.text(0.02, 0.98, '', transform=ax2.transAxes, 
                           fontsize=12, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax2.legend(loc='upper right', fontsize=8)
        
        def init():
            """Initialize animation"""
            for i, artist in enumerate(robot_artists):
                if isinstance(artist, AnnotationBbox):
                    artist.xy = (robot_positions[i, 0], robot_positions[i, 1])
                else:
                    artist.set_data([robot_positions[i, 0]], [robot_positions[i, 1]])
            for i, line in enumerate(trajectory_lines):
                line.set_data([robot_positions[i, 0]], [robot_positions[i, 1]])

            # Initialize barycenter
            barycenter_marker.set_data([barycenter_trajectory[0, 0]], [barycenter_trajectory[0, 1]])
            barycenter_line.set_data([barycenter_trajectory[0, 0]], [barycenter_trajectory[0, 1]])

            time_text.set_text('Time = 0.00 s\nIteration = 0')
            for time_pointer in time_pointers:
                time_pointer.set_xdata([0])

            return (*robot_artists, *trajectory_lines, *collision_circles, 
                    barycenter_marker, barycenter_line, time_text, *time_pointers)
        
        def update(frame):
            """Update animation frame"""
            # Update robot positions and trajectories
            for i in range(self.N):
                # Current position
                if isinstance(robot_artists[i], AnnotationBbox):
                    # Update image position
                    robot_artists[i].xy = (z[frame, i, 0], z[frame, i, 1])
                    robot_artists[i].xybox = (z[frame, i, 0], z[frame, i, 1])
                else:
                    # Update dot position
                    robot_artists[i].set_data([z[frame, i, 0]], [z[frame, i, 1]])
                
                # Trajectory up to current frame
                trajectory_lines[i].set_data(z[:frame+1, i, 0], z[:frame+1, i, 1])

                # Update collision circle position
                collision_circles[i].center = (z[frame, i, 0], z[frame, i, 1])
            
            # Update barycenter position (current position)
            barycenter_marker.set_data([barycenter_trajectory[frame, 0]], 
                                      [barycenter_trajectory[frame, 1]])
            
            # Update barycenter trajectory (path up to current frame)
            barycenter_line.set_data(barycenter_trajectory[:frame+1, 0], 
                                    barycenter_trajectory[:frame+1, 1])
            
            # Update time text
            time_text.set_text(f'Time = {frame * dt:.2f} s\nIteration = {frame}')
            
            # Update time pointers in left plots
            for time_pointer in time_pointers:
                time_pointer.set_xdata([frame * dt])
            
            return (*robot_artists, *trajectory_lines, *collision_circles, barycenter_marker, 
                    barycenter_line, time_text, *time_pointers)
        
        # Create animation
        max_frames = 150
        frames = range(0, maxIters, max(1, maxIters // max_frames))

        ani = FuncAnimation(fig, update, frames=frames, init_func=init, 
                          blit=True, interval=50)
        
        if save:
            print(f'Saving animation as: {gif_name}.gif')
            ani.save(f'{gif_name}.gif', writer='pillow', fps=20)
            print('Animation saved')
        
        fig.suptitle('Robot Trajectory Animation', fontsize=22)

        plt.tight_layout()
        plt.show()
        return fig
    
    def color_image(self, image, color):
        """Colorize a grayscale/black image with a specific color using a better method""" 
        
        # Convert to RGBA if not already
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        
        # Convert to numpy array for easier manipulation
        img_array = np.array(image).astype(float)
        
        # Extract RGB and alpha channels
        rgb = img_array[:, :, :3]
        alpha = img_array[:, :, 3]
        
        # Convert matplotlib color (0-1) to RGB (0-255)
        target_color = np.array([color[0] * 255, color[1] * 255, color[2] * 255])
        
        # Calculate brightness (inverse for black drones)
        # For a black drone on transparent background, we want dark areas to become colored
        brightness = 1.0 - (np.mean(rgb, axis=2) / 255.0)  # Inverted
        
        # Create colored image by applying target color scaled by grayscale intensity
        colored_rgb = np.zeros_like(rgb)
        for i in range(3):
            # Apply target color with brightness, keeping darker areas more saturated
            colored_rgb[:, :, i] = target_color[i] * brightness
        
        # Ensure values are in valid range
        colored_rgb = np.clip(colored_rgb, 0, 255)
        
        # Combine with alpha channel
        colored_array = np.dstack([colored_rgb, alpha]).astype('uint8')
        
        # Convert back to PIL Image
        colored_image = Image.fromarray(colored_array, 'RGBA')
        
        return colored_image