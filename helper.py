import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

class Digraph:
    """Class to manage graph creation and weight computation"""
    
    def __init__(self, N, p_er=0.5, graph_type='random'):
        """Initialize graph with N nodes and double stochastic weights"""
        self.N = N
        self.p_er = p_er
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
                Adj = nx.adjacency_matrix(self.G).toarray()
                test = np.linalg.matrix_power(Adj + np.eye(self.N), self.N)
                if np.all(test > 0):  # check strong connectivity
                    break
        else:
            raise ValueError("Unknown topology")
        
        A_unweighted = Adj + np.eye(self.N)
        self.A = self.metropolis_hastings_weights(A_unweighted)
        self.G = nx.from_numpy_array(self.A)  #  G is updated to include self-loops for visualization purposes
    
    def metropolis_hastings_weights(self, A):
        """Compute Metropolis-Hastings weights"""
        deg = np.sum(A, axis=1)
        A_mh = np.zeros((self.N, self.N))
        
        for i in range(self.N):
            for j in range(self.N):
                if A[i, j] == 1 and i != j:
                    A_mh[i, j] = 1.0 / (1 + max(deg[i], deg[j]))
            
            A_mh[i, i] = 1.0 - np.sum(A_mh[i, :])
            # self-loops and non-neighbors are considered in the sum but do not change the results because equals to zero
        
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
    
    
    def distributed_aggregative(z, bary, gamma, r0, r,N,d):
        """ Compute cost and gradient for distributed aggregative formation control """
        cost = 0.0
        grad_1 = np.zeros((d))
        grad_2 = np.zeros((d))

        target_dist = z - r
        bary_dist = bary - r0
        cost += gamma * (np.linalg.norm(target_dist))**2 + (np.linalg.norm(bary_dist))**2

        # Gradient computation
        grad_1 = 2*gamma*target_dist + (2*bary_dist)/N
        grad_2 = (2*bary_dist) #gradient of the cost function

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
        
        # Add text annotations
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