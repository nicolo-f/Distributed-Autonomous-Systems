import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

np.random.seed(0)

def metropolis_hastings_weights(Adj):
    NN = Adj.shape[0]
    degrees = np.sum(Adj, axis=1)
    W = np.zeros((NN, NN))
    
    for i in range(NN):
        for j in range(NN):
            if Adj[i, j] == 1 and i != j:
                W[i, j] = 1.0 / (1 + max(degrees[i], degrees[j]))
    
    for i in range(NN):
        W[i, i] = 1.0 - np.sum(W[i, :])
        
    return W

def create_graph(NN, p_er, type='random'):
    if type == 'cycle':
        G = nx.cycle_graph(NN)
        Adj = nx.adjacency_matrix(G).toarray()
    elif type == 'path':
        G = nx.path_graph(NN)
        Adj = nx.adjacency_matrix(G).toarray()
    elif type == 'star':
        G = nx.star_graph(NN - 1) # star_graph(n) has n+1 nodes
        Adj = nx.adjacency_matrix(G).toarray()
    elif type == 'random':
        while 1:
            G = nx.erdos_renyi_graph(NN, p_er)
            Adj = nx.adjacency_matrix(G).toarray()
            test = np.linalg.matrix_power(Adj + np.eye(NN), NN)
            if np.all(test > 0):  # check strong connectivity
                break
    else:
        raise ValueError("Unknown topology")

    A_unweighted = Adj + np.eye(NN)
    A = metropolis_hastings_weights(A_unweighted)

    return Adj, A

def cost_fcn(zz, QQ, rr): 
    val = 0.5 * zz.T @ QQ @ zz + rr.T @ zz
    grad = QQ @ zz + rr
    return val, grad