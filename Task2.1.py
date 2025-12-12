import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from helper import Digraph, CostFunction, Plotter

np.random.seed(0)

# Parameters
d = 2  # dimension of the decision variable z
N = 8  # number of robots
p_er = 0.5  # probability for Erdos-Renyi graph
type = 'random'  # type of graph
maxIters = 1000
alpha = 1e-3
target_std = 2  # standard deviation to generate target positions
gamma = 1  # trade-off parameter for target attainment vs formation keeping

r0 = np.array([2.0, 2.0]) #distance from barycenter to keep the fleet tight
robot_positions = np.random.uniform(low=0, high=10, size=(N, d))
target_positions = robot_positions + np.random.normal(0, target_std,size=(N, d))


# Initializations
z_init = robot_positions
# barycenter initialization

print(f"\nInitial robot positions:\n", robot_positions)
print(f"\nTarget positions:\n", target_positions)

cost = np.zeros((maxIters))
z = np.zeros((maxIters, N, d))
z[0, :, :] = z_init
s = np.zeros((maxIters, N, d))
v = np.zeros((maxIters, N, d))
grad_norm_s = np.zeros((maxIters, N))
grad_norm_v = np.zeros((maxIters, N)) 

for i in range(N):
    s[0, i] = z[0, i]
    _,_, v[0, i] = CostFunction.distributed_aggregative(z[0, i], s[0, i], robot_positions[i], d, N)

graph = Digraph(N, p_er, type)
A = graph.get_weight_matrix()
G = graph.get_graph()

# Gradient Tracking Algorithm
for k in range(maxIters - 1):
    for i in range(N):
        N_i = np.nonzero(A[i])[0]  
        for j in N_i:
            z[k + 1, i] += A[i, j] * z[k, j]

        z[k + 1, i] -= alpha * s[k, i]

        for j in N_i:
            s[k + 1, i] += A[i, j] * s[k, j]

        _, grad_ell_i_new = CostFunction.target_localization(z[k+1, i], distances[i], robot_positions[i], d, N)
        ell_i, grad_ell_i_old = CostFunction.target_localization(z[k, i], distances[i], robot_positions[i], d, N)
        s[k + 1, i] += grad_ell_i_new - grad_ell_i_old

        grad_norm[k + 1, i] = np.linalg.norm(grad_ell_i_new)  # Store gradient norm
        cost[k] += ell_i  # accumulate global cost

# Extract final estimated target positions
final_estimates = z[-1, 0, :].reshape((N, d))  # Take agent 0's estimate (all should agree)
print(f"\nFinal estimated target positions:\n", final_estimates)
print(f"\nTrue target positions:\n", true_targets)
print(f"\nEstimation errors:\n", final_estimates - true_targets)
print("\n\n")

# Create plotter and generate all plots
plotter = Plotter(N, d, N)

fig1 = plotter.plot_graph_and_weights(G, A)
fig2 = plotter.plot_cost_and_consensus(cost, z, maxIters)
fig3 = plotter.plot_gradient_norms(grad_norm, maxIters)
fig4 = plotter.plot_target_estimation(z, true_targets, maxIters)

plt.show()