import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

np.random.seed(0)

def metropolis_hastings_weights(A, N):
    deg = np.sum(A, axis=1)
    A_mh = np.zeros((N, N))
    
    for i in range(N):
        for j in range(N):
            if A[i, j] == 1 and i != j:
                A_mh[i, j] = 1.0 / (1 + max(deg[i], deg[j]))
            
        A_mh[i, i] = 1.0 - np.sum(A_mh[i, :])
    
    return A_mh

def create_graph(N, p_er, type='random'):
    if type == 'cycle':
        G = nx.cycle_graph(N)
        Adj = nx.adjacency_matrix(G).toarray()
    elif type == 'path':
        G = nx.path_graph(N)
        Adj = nx.adjacency_matrix(G).toarray()
    elif type == 'star':
        G = nx.star_graph(N - 1) # star_graph(n) has n+1 nodes
        Adj = nx.adjacency_matrix(G).toarray()
    elif type == 'random':
        while 1:
            G = nx.erdos_renyi_graph(N, p_er)
            Adj = nx.adjacency_matrix(G).toarray()
            test = np.linalg.matrix_power(Adj + np.eye(N), N)
            if np.all(test > 0):  # check strong coNectivity
                break
    else:
        raise ValueError("Unknown topology")

    A_unweighted = Adj + np.eye(N)
    A = metropolis_hastings_weights(A_unweighted, N)
    
    G = nx.from_numpy_array(A)
    return A, G

# New cost function for target localization
def cost_fcn(z, dist, p, d, NT):

    z_reshaped = z.reshape((NT, d))  # Reshape to (NT, d) for easier computation
    
    cost = 0.0
    grad = np.zeros((NT, d))
    
    for tau in range(NT):
        # Compute ||z_τ - p_i||² (squared distance from estimated target to robot)
        diff = z_reshaped[tau] - p  # Difference vector
        squared_dist = np.linalg.norm(diff)**2
        
        # Residual: d²_iτ - ||z_τ - p_i||²
        residual = dist[tau]**2 - squared_dist
        
        # Cost contribution: residual²
        cost += residual**2
        
        # Gradient: ∂/∂z_τ [(d²_iτ - ||z_τ - p_i||²)²]
        # = 2 * residual * ∂/∂z_τ [-(||z_τ - p_i||²)]
        # = 2 * residual * (-2) * (z_τ - p_i)
        # = -4 * residual * (z_τ - p_i)
        grad[tau] = -4 * residual * diff
    
    return cost, grad.flatten()  # Return flattened gradient (d*NT,)

d = 2  # dimension of the decision variable z
N = 8  # number of robots
NT = 2  # number of targets
p_er = 0.5  # probability for Erdos-Renyi graph
type = 'random'  # type of graph
maxIters = 1000
alpha = 1e-3
z_init = np.random.uniform(low=0, high=10, size=(N, NT*d))
noise_std = 0.1  # standard deviation of measurement noise

# Generate robot position 
robot_positions = np.random.uniform(low=0, high=10, size=(N, d))
print(f"\nRobot positions (N={N}, d={d}):\n", robot_positions)

# Generate true target positions
true_targets = np.random.uniform(low=0, high=10, size=(NT, d))
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
    _, s[0, i] = cost_fcn(z[0, i], distances[i], robot_positions[i], d, NT)
    grad_norm[0, i] = np.linalg.norm(s[0, i])

A, G = create_graph(N, 0.5, type)

for k in range(maxIters - 1):
    for i in range(N):
        N_i = np.nonzero(A[i])[0]  
        for j in N_i:
            z[k + 1, i] += A[i, j] * z[k, j]

        z[k + 1, i] -= alpha * s[k, i]

        for j in N_i:
            s[k + 1, i] += A[i, j] * s[k, j]

        _, grad_ell_i_new = cost_fcn(z[k+1, i], distances[i], robot_positions[i], d, NT)
        ell_i, grad_ell_i_old = cost_fcn(z[k, i], distances[i], robot_positions[i], d, NT)
        s[k + 1, i] += grad_ell_i_new - grad_ell_i_old

        grad_norm[k + 1, i] = np.linalg.norm(grad_ell_i_new)  # Store gradient norm
        cost[k] += ell_i  # accumulate global cost

z_avg = np.mean(z, axis=1)

# print(f"\nFinal estimated target positions of each agent: {z[-1, :, :].reshape((N, NT, d))}")

# Extract final estimated target positions
final_estimates = z[-1, 0, :].reshape((NT, d))  # Take agent 0's estimate (all should agree)
print(f"\nFinal estimated target positions:\n", final_estimates)
print(f"\nTrue target positions:\n", true_targets)
print(f"\nEstimation errors:\n", final_estimates - true_targets)

# First figure: Graph and Weight Matrix
fig1, axes1 = plt.subplots(figsize=(10, 5), nrows=1, ncols=2)

# Graph drawing
ax = axes1[0]
nx.draw_kamada_kawai(G, with_labels=True, ax=ax)
ax.set_title('Network Graph')

# Weight matrix A as annotated heatmap
ax = axes1[1]
im = ax.matshow(A, cmap='Blues', aspect='auto')
ax.set_title('Weight Matrix A', pad=20)
ax.set_xlabel('Agent j')
ax.set_ylabel('Agent i')
ax.xaxis.set_ticks_position('bottom')
ax.set_xticks(range(N))
ax.set_yticks(range(N))

# Add text annotations
for i in range(N):
    for j in range(N):
        text = ax.text(j, i, f'{A[i, j]:.2f}',
                      ha="center", va="center", color="black", fontsize=8)

plt.tight_layout()

# Second figure: Cost and Convergence plots
fig2, axes2 = plt.subplots(figsize=(10, 5), nrows=1, ncols=2)

# Cost evolution
ax = axes2[0]
ax.plot(np.arange(maxIters - 1), cost[:-1])
ax.set_xlabel('Iteration')
ax.set_ylabel('Cost')
ax.set_title('Cost Evolution')
ax.grid(True)

# Convergence (deviation from average)
ax = axes2[1]
for i in range(N):
    ax.semilogy(np.arange(maxIters), np.linalg.norm(z[:, i] - z_avg, axis=1), label=f'Agent {i}')
ax.set_xlabel('Iteration')
ax.set_ylabel('||z_i - z_avg||')
ax.set_title('Consensus Error')
ax.grid(True)
ax.legend() 

# Third figure: Agents and total Gradient Norm Evolution
fig3, axes3 = plt.subplots(figsize=(10, 5), nrows=1, ncols=2)

# Agents gradient norm evolution
ax = axes3[0]
for i in range(N):
    ax.semilogy(np.arange(maxIters), grad_norm[:, i], label=f'Agent {i}')
ax.set_xlabel('Iteration')
ax.set_ylabel('||∇f_i(z_i)||')
ax.set_title('Individual Gradient Norms')
ax.grid(True)
# ax.legend() 

# Total gradient norm (sum across all agents)
ax = axes3[1]
total_grad_norm = np.sum(grad_norm, axis=1)
ax.semilogy(np.arange(maxIters), total_grad_norm)
ax.set_xlabel('Iteration')
ax.set_ylabel('Σ||∇f_i(z_i)||')
ax.set_title('Total Gradient Norm')
ax.grid(True)

# Fourth figure: Target Position Estimation Error Evolution
fig4, axes4 = plt.subplots(figsize=(10, 5), nrows=1, ncols=2)

ax = axes4[0]
# Compute estimation error for each target over iterations
# Use mean estimate across all agents at each iteration
target_errors = np.zeros((maxIters, NT))

for k in range(maxIters):
    # Get mean estimate across all agents at iteration k
    z_mean_k = np.mean(z[k, :, :], axis=0)  # Average across agents
    z_mean_reshaped = z_mean_k.reshape((NT, d))  # Reshape to (NT, d)
    
    # Compute error for each target
    for tau in range(NT):
        target_errors[k, tau] = np.linalg.norm(z_mean_reshaped[tau] - true_targets[tau])

# Plot error evolution for each target
for tau in range(NT):
    ax.semilogy(np.arange(maxIters), target_errors[:, tau], label=f'Target {tau}')
    # ax.semilogy(np.arange(maxIters), [maxIters *true_targets], label=f'Target {tau}')

ax.set_xlabel('Iteration')
ax.set_ylabel('||z̄_τ - p*_τ|| (Estimation Error)')
ax.set_title('Target Position Estimation Error Evolution')
ax.legend()
ax.grid(True)

ax = axes4[1]

# Plot evolution of estimated target positions over iterations
# Use mean estimate across all agents
for tau in range(NT):
    # Extract x and y coordinates for target tau over all iterations
    x_coords = np.zeros(maxIters)
    y_coords = np.zeros(maxIters)
    
    for k in range(maxIters):
        z_mean_k = np.mean(z[k, :, :], axis=0)  # Average across agents
        z_mean_reshaped = z_mean_k.reshape((NT, d))  # Reshape to (NT, d)
        x_coords[k] = z_mean_reshaped[tau, 0]
        y_coords[k] = z_mean_reshaped[tau, 1]

    # Get color for this target
    color = f'C{tau}'
    
    # Plot trajectory of estimated position
    ax.plot(x_coords, y_coords, '-', color=color, label=f'Target {tau} estimate', alpha=0.7)
    
    # Mark initial estimate
    ax.scatter(x_coords[0], y_coords[0], s=100, marker='o', color=color, edgecolors='black', linewidths=1, zorder=3)
    
    # Mark final estimate
    ax.scatter(x_coords[-1], y_coords[-1], s=100, marker='x', color=color, edgecolors='black', linewidths=1, zorder=3)
    
    # Plot true target position with dotted line (horizontal and vertical)
    ax.axhline(y=true_targets[tau, 1], color=color, linestyle='--', 
               alpha=0.5, linewidth=2)
    ax.axvline(x=true_targets[tau, 0], color=color, linestyle='--', 
               alpha=0.5, linewidth=2)
    
    # Mark true target position
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
plt.show()

