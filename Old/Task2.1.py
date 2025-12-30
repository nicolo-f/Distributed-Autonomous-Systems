import numpy as np
import matplotlib.pyplot as plt
from helper import Digraph, CostFunction, Plotter
np.random.seed(0) # For reproducibility

# Path to robot image for animation
robot_image_path = 'Mirco/drone.png'

# Parameters
d = 2  # dimension of the decision variable z
N = 8  # number of robots
p_er = 0.5  # probability for Erdos-Renyi graph
type = 'random'  # type of graph
maxIters = 500
alpha = 1e-2 # step-size
target_std = 2  # standard deviation to generate target positions
gamma = 0.5  # trade-off parameter for target attainment vs formation keeping

# Robot and target initializations 

# radius = 6.0
# angles = np.linspace(0, 2*np.pi, N, endpoint=False)
# Possibility to initiate robot positions in circle
# robot_positions = np.column_stack([radius * np.cos(angles) + 5, radius * np.sin(angles) + 5])
robot_positions = np.random.uniform(low=0, high=10, size=(N, d))
target_positions = robot_positions + np.random.normal(0, target_std,size=(N, d))


# Initializations
z_init = robot_positions

print(f"\nInitial robot positions:\n", z_init)
print(f"\nTarget positions:\n", target_positions)

# Initialize variables for the algorithm
cost = np.zeros((maxIters))
z = np.zeros((maxIters, N, d))
z[0, :, :] = z_init 
r0 = np.zeros((N, d))
s = np.zeros((maxIters, N, d))
v = np.zeros((maxIters, N, d))
grad_norm_2 = np.zeros((maxIters, N))
grad_norm_1 = np.zeros((maxIters, N)) 

# Define desired r0

# All agents share the same desired barycenter (mean of targets)
# r0 = np.tile(np.mean(target_positions, axis=0), (N, 1))

# All agents share a specific constant barycenter
# r0 = np.full((N, d), [2.0, 2.0]) 

# r0 is agent-dependent (each agent has its own desired position)
for i in range(N):
    r0[i] = z[0, i]

# Initialize s (local barycenter estimates) and v (local gradients_2 estimates)
for i in range(N):
    s[0, i] = z[0, i]
    _,_, v[0, i] = CostFunction.distributed_aggregative(z[0, i], s[0, i], gamma, r0[i], target_positions[i], d, N)
    # _,_, v[0, i] = CostFunction.distributed_aggregative(z[0, i], s[0, i], gamma, r0, target_positions[i], d, N)

    grad_norm_1[0, i] = np.linalg.norm(v[0, i])  # norm of gradient_2

# Create strongly connected graph and the associated adjacency matrix
graph = Digraph(N, p_er, type)
A = graph.get_weight_matrix()
G = graph.get_graph()

# Aggregative Tracking Distributed Optimization Algorithm
for k in range(maxIters - 1):
    for i in range(N):
        ell_i,grad_1,grad_2 = CostFunction.distributed_aggregative(z[k, i], s[k, i], gamma, r0[i], target_positions[i], d, N)
        # ell_i,grad_1,grad_2 = CostFunction.distributed_aggregative(z[k, i], s[k, i], gamma, r0, target_positions[i], d, N)
        # notice: grad_phi always equal to 1 so we dont add it in the code
 
        z[k + 1, i] = z[k, i] - alpha * (grad_1 + v[k, i])  # Update robot position
 
        N_i = np.nonzero(A[i])[0]   # Neighbors of robot i

        for j in N_i:
            s[k + 1, i] += A[i, j] * s[k, j]
            v[k + 1, i] += A[i, j] * v[k, j]
 
        s[k + 1, i] +=  z[k + 1, i] - z[k, i]   # Update local barycenter estimate
        _,_, grad_2_next = CostFunction.distributed_aggregative(z[k + 1, i], s[k + 1, i], gamma, r0[i], target_positions[i], d, N)
        # _,_, grad_2_next = CostFunction.distributed_aggregative(z[k + 1, i], s[k + 1, i], gamma, r0, target_positions[i], d, N)
        v[k + 1, i] += grad_2_next - grad_2   # Update local gradient_2 estimate with innovation term
 
        grad_norm_1[k + 1, i] = np.linalg.norm(grad_1)  
        grad_norm_2[k + 1, i] = np.linalg.norm(grad_2)  
        cost[k] += ell_i

# Compute final metrics
final_positions = z[-1, :, :]
final_barycenter = np.mean(final_positions, axis=0) # True final barycenter
final_aggregate_estimates = s[-1, :, :]  # Each agent's estimate of barycenter at final iteration 

# Print some informations
print(f"\nFinal robot positions:\n", final_positions)
print(f"\nTarget positions:\n", target_positions)
print(f"\nPosition errors:\n", final_positions - target_positions)
print(f"\nTrue final barycenter:\n", final_barycenter)
print(f"\nDesired barycenter r0:\n", r0)
print(f"\nBarycenter error:\n", final_barycenter - r0)
print(f"\nAgents' barycenter estimates (should all agree):\n", final_aggregate_estimates)
print("\n\n")

# Create plotter and generate all plots
plotter = Plotter(N, d, N)

fig1 = plotter.plot_graph_and_weights(G, A)
fig2 = plotter.plot_cost_and_consensus(cost, z, maxIters)
fig3 = plotter.plot_gradient_norms(grad_norm_1, maxIters)
fig4 = plotter.plot_robot_trajectories(z, robot_positions, final_positions, target_positions, final_barycenter)

fig5 = plotter.plot_robot_animation(z, robot_positions, target_positions, 
                                     maxIters, dt=0.01, save=False, 
                                     gif_name='robot_formation_animation',
                                     use_images=True, robot_image_path=robot_image_path,
                                     image_zoom=0.05)

plt.show()