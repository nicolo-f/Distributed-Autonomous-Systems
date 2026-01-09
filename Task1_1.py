import numpy as np
import matplotlib.pyplot as plt

from helper import Digraph, CostFunction, Plotter

np.random.seed(0)

# Variables for saving figures
save_fig = True
fig_save_path = "./figures/Task1_1/path_graph/"

# Parameters
d = 3           # dimension of the decision variable z
N = 8           # number of robots
p_er = 0.5      # probability for Erdos-Renyi graph
type = 'path'   # possible choices: 'cycle', 'random', 'star', 'path'
maxIters = 1000
alpha = 1e-2    # step size
noise_std = 0.1 # standard deviation of measurement noise

# Robot initializations 
z_init = np.random.normal(size=(N, d))
Q = []
r = []

# Initialize variables for the algorithm
cost = np.zeros((maxIters))
z = np.zeros((maxIters, N, d))
z[0, :, :] = z_init
s = np.zeros((maxIters, N, d))
grad_norm = np.zeros((maxIters, N)) 

for i in range(N):
    Q.append(np.diag(np.random.uniform(size=(d))))
    r.append(np.random.normal(size=(d)))

for i in range(N):
    _, s[0, i] = CostFunction.quadratic(z[0, i], Q[i], r[i])
    grad_norm[0, i] = np.linalg.norm(s[0, i])

# Create strongly connected graph and the associated adjacency matrix
graph = Digraph(N, p_er, type)
A = graph.get_weight_matrix()
G = graph.get_graph()

for k in range(maxIters - 1):
    for i in range(N):
        N_i = np.nonzero(A[i])[0]  
        for j in N_i:
            z[k + 1, i] += A[i, j] * z[k, j]

        z[k + 1, i] -= alpha * s[k, i]

        for j in N_i:
            s[k + 1, i] += A[i, j] * s[k, j]

        _, grad_ell_i_new = CostFunction.quadratic(z[k + 1, i], Q[i], r[i])
        ell_i, grad_ell_i_old = CostFunction.quadratic(z[k, i], Q[i], r[i])
        s[k + 1, i] += grad_ell_i_new - grad_ell_i_old

        grad_norm[k + 1, i] = np.linalg.norm(grad_ell_i_new)  # Store gradient norm
        cost[k] += ell_i  # accumulate global cost

z_avg = np.mean(z, axis=1)

# Create plotter and generate all plots
plotter = Plotter(N, d, N)

fig1 = plotter.plot_graph_and_weights(G, A, save=save_fig, save_path=fig_save_path+"graph.png")
fig2 = plotter.plot_cost_and_consensus(cost, z, maxIters, save=save_fig, save_path=fig_save_path+"cost_consensus.png")
fig3 = plotter.plot_gradient_norms(grad_norm, maxIters, save=save_fig, save_path=fig_save_path+"gradient_norms.png")

plt.show()