import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from Mirco.helper_old import Digraph, CostFunction, Plotter

np.random.seed(0)

# Parameters
d = 2  # dimension of the decision variable z
N = 8  # number of robots
NT = 2  # number of targets
p_er = 0.5  # probability for Erdos-Renyi graph
type = 'random'  # type of graph
maxIters = 1000
alpha = 1e-3
noise_std = 0.1  # standard deviation of measurement noise

# Initialize random positions
z_init = np.random.uniform(low=0, high=10, size=(N, NT*d))
robot_positions = np.random.uniform(low=0, high=10, size=(N, d))
true_targets = np.random.uniform(low=0, high=10, size=(NT, d))

print(f"\nRobot positions (N={N}, d={d}):\n", robot_positions)
print(f"\nTrue target positions (NT={NT}, d={d}):\n", true_targets)

# Generate noisy distance measurements
distances = np.zeros((N, NT))
for i in range(N):
    for tau in range(NT):
        true_dist = np.linalg.norm(true_targets[tau] - robot_positions[i])
        distances[i, tau] = true_dist + np.random.normal(0, noise_std)

print(f"\nDistance measurements (N={N}, NT={NT}):\n", distances)

cost = np.zeros((maxIters))
z = np.zeros((maxIters, N, NT*d))
z[0, :, :] = z_init
s = np.zeros((maxIters, N, NT*d))
grad_norm = np.zeros((maxIters, N)) 

for i in range(N):
    _, s[0, i] = CostFunction.target_localization(z[0, i], distances[i], robot_positions[i], d, NT)
    grad_norm[0, i] = np.linalg.norm(s[0, i])

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

        _, grad_ell_i_new = CostFunction.target_localization(z[k+1, i], distances[i], robot_positions[i], d, NT)
        ell_i, grad_ell_i_old = CostFunction.target_localization(z[k, i], distances[i], robot_positions[i], d, NT)
        s[k + 1, i] += grad_ell_i_new - grad_ell_i_old

        grad_norm[k + 1, i] = np.linalg.norm(grad_ell_i_new)  # Store gradient norm
        cost[k] += ell_i  # accumulate global cost

# Extract final estimated target positions
final_estimates = z[-1, 0, :].reshape((NT, d))  # Take agent 0's estimate (all should agree)
print(f"\nFinal estimated target positions:\n", final_estimates)
print(f"\nTrue target positions:\n", true_targets)
print(f"\nEstimation errors:\n", final_estimates - true_targets)
print("\n\n")

# Create plotter and generate all plots
plotter = Plotter(N, d, NT)

fig1 = plotter.plot_graph_and_weights(G, A)
fig2 = plotter.plot_cost_and_consensus(cost, z, maxIters)
fig3 = plotter.plot_gradient_norms(grad_norm, maxIters)
fig4 = plotter.plot_target_estimation(z, true_targets, maxIters)

plt.show()