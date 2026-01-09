import numpy as np
import networkx as nx

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
    
    def distributed_aggregative(z, bary, gamma, r0, r, d):
        """ Compute cost and gradient for distributed aggregative formation control """
        cost = 0.0
        grad_1 = np.zeros((d))
        grad_2 = np.zeros((d))

        target_dist = z - r
        # bary_dist = bary - r0
        bary_dist = z - bary  # corrected direction for formation keeping
        # cost += gamma * (np.linalg.norm(target_dist))**2 + (np.linalg.norm(bary_dist))**2
        cost += gamma * (np.linalg.norm(target_dist))**2 + (1-gamma)*(np.linalg.norm(bary_dist))**2

        # # Gradient computation (other solution)
        grad_1 = 2 * gamma * target_dist + 2 * (1-gamma) * bary_dist
        grad_2 = 2 * (1-gamma) * (-bary_dist)

        return cost, grad_1, grad_2
    
    def distributed_aggregative_barrier(z, bary, gamma, r0, r, d, z_all, mu=1.0, threshold=1.0):
        """ Compute cost and gradient for distributed aggregative formation control """
        cost = 0.0
        barrier_cost = 0.0

        # delta = 10.0 # formation keeping weight (used for testing)

        grad_1 = np.zeros((d))
        grad_2 = np.zeros((d))
        barrier_grad = np.zeros((d))

        target_dist = z - r
        bary_dist = z - bary
        # bary_dist = bary - r0

        # Barrier function for collision avoidance: -log(||z_i - z_j||^2 - threshold^2)
        for j in range(len(z_all)):
            z_j = z_all[j]
            
            # Skip self
            if np.allclose(z, z_j):
                continue

            # Compute squared distances
            diff = z - z_j
            dist = np.linalg.norm(diff)
            # Barrier argument: ||z_i - z_j||^2 - threshold^2
            barrier_arg = max(dist**2 - threshold**2, 1e-6)

            barrier_cost += -np.log(barrier_arg)
            barrier_grad += -2 * diff / barrier_arg

        cost += gamma * (np.linalg.norm(target_dist))**2 + (1-gamma)*(np.linalg.norm(bary_dist))**2 + mu * barrier_cost
        # cost += gamma * (np.linalg.norm(target_dist))**2 + delta*(np.linalg.norm(bary_dist))**2 + mu * barrier_cost
        print(f"Barrier cost: {mu * barrier_cost}")
        
        # Gradient computation (other solution)
        grad_1 = 2 * gamma * target_dist + mu * barrier_grad + 2 * (1-gamma) * bary_dist
        grad_2 = 2 * (1-gamma) * (-bary_dist)
        # grad_2 = 2 * delta * bary_dist

        
        
        return cost, grad_1, grad_2