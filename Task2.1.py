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
maxIters = 3000
alpha = 1e-4
target_std = 2  # standard deviation to generate target positions
gamma = 1  # trade-off parameter for target attainment vs formation keeping

r0 = np.array([2.0, 2.0]) #distance from barycenter to keep the fleet tight
# robot_positions = np.random.uniform(low=0, high=10, size=(N, d))

# Position robots in a circle
radius = 6.0
angles = np.linspace(0, 2*np.pi, N, endpoint=False)
robot_positions = np.column_stack([radius * np.cos(angles) + 5, radius * np.sin(angles) + 5])
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
    _,_, v[0, i] = CostFunction.distributed_aggregative(z[0, i], s[0, i], gamma, r0, target_positions[i], d, N)

graph = Digraph(N, p_er, type)
A = graph.get_weight_matrix()
G = graph.get_graph()

# Gradient Tracking Algorithm
for k in range(maxIters - 1):

    for i in range(N):
        ell_i,grad_1,grad_2_old = CostFunction.distributed_aggregative(z[k, i], s[k, i], gamma, r0, target_positions[i], d, N)
        #grad_phi = always equal to 1

        z[k + 1, i] = z[k, i] - alpha * (grad_1 + v[k, i])

        N_i = np.nonzero(A[i])[0]  
        for j in N_i:
            s[k + 1, i] += A[i, j] * s[k, j] 
            v[k + 1, i] += A[i, j] * v[k, j]

        s[k + 1, i] +=  z[k + 1, i] - z[k, i]
        _,_, grad_2_new = CostFunction.distributed_aggregative(z[k + 1, i], s[k + 1, i], gamma, r0, target_positions[i], d, N)
        v[k + 1, i] += grad_2_new - grad_2_old

        grad_norm_s[k + 1, i] = np.linalg.norm(grad_2_old)  # Store gradient norm
        grad_norm_v[k + 1, i] = np.linalg.norm(grad_1)  # Store gradient norm
        cost[k] += ell_i


# Create plotter and generate all plots
plotter = Plotter(N, d, N)

fig1 = plotter.plot_graph_and_weights(G, A)
fig3 = plotter.plot_gradient_norms(grad_norm_s, maxIters)
fig4 = plotter.plot_gradient_norms(grad_norm_v, maxIters)
# For aggregative problem, we can visualize robot trajectories
fig2, axes = plt.subplots(figsize=(10, 5), nrows=1, ncols=2)

# Plot 1: Robot trajectories
ax = axes[0]
final_positions = z[-1, :, :]
final_barycenter = np.mean(final_positions, axis=0)

for i in range(N):
    # Get color for this robot
    color = f'C{i}'
    
    # Plot trajectory with same color
    ax.plot(z[:, i, 0], z[:, i, 1], '-', color=color, alpha=0.5, label=f'Robot {i}')
    
    # Initial position (circle) - same color
    ax.scatter(robot_positions[i, 0], robot_positions[i, 1], s=100, marker='o',
              color=color, edgecolors='black', linewidths=1)
    
    # Final position (cross) - same color
    ax.scatter(final_positions[i, 0], final_positions[i, 1], s=100, marker='x',
              color=color, linewidths=2)
    
    # Target position (star) - same color
    ax.scatter(target_positions[i, 0], target_positions[i, 1], s=150, marker='*',
              color=color, edgecolors='black', linewidths=1)

# Desired barycenter (red pentagon)
ax.scatter(r0[0], r0[1], s=300, marker='P', color='red',
          edgecolors='black', linewidths=2, label='Desired barycenter', zorder=10)

# Final barycenter (green diamond)
ax.scatter(final_barycenter[0], final_barycenter[1], s=200, marker='D',
          color='green', edgecolors='black', linewidths=2, label='Final barycenter', zorder=10)

ax.set_xlabel('X position')
ax.set_ylabel('Y position')
ax.set_title('Robot Trajectories')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax.grid(True)
ax.axis('equal')

# Plot 2: Barycenter estimate error
ax = axes[1]
bary_errors = np.zeros(maxIters)
for k in range(maxIters):
    true_bary_k = np.mean(z[k, :, :], axis=0)
    # Use one agent's estimate (they should all converge to same value)
    bary_errors[k] = np.linalg.norm(s[k, 0, :] - true_bary_k)

ax.semilogy(np.arange(maxIters), bary_errors)
ax.set_xlabel('Iteration')
ax.set_ylabel('||s_i - true_barycenter||')
ax.set_title('Barycenter Estimate Error')
ax.grid(True)
 
plt.show()