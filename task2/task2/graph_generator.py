import numpy as np
import networkx as nx

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