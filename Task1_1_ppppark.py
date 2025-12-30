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

def cost_fcn(z, Q, r): 
    val = 0.5 * z.T @ Q @ z + r.T @ z
    grad = Q @ z + r
    return val, grad

d = 3  # dimension of the decision variable z
N = 8  # number of agents
p_er = 0.5  # probability for Erdos-Renyi graph
type = 'random'  # type of graph
maxIters = 1000
alpha = 1e-1
z_init = np.random.normal(size=(N, d))

Q = []
r = []
for i in range(N):
    Q.append(np.diag(np.random.uniform(size=(d))))
    r.append(np.random.normal(size=(d)))

cost = np.zeros((maxIters))
z = np.zeros((maxIters, N, d))
z[0, :, :] = z_init
s = np.zeros((maxIters, N, d))
grad_norm = np.zeros((maxIters, N)) 

for i in range(N):
    _, s[0, i] = cost_fcn(z[0, i], Q[i], r[i])
    grad_norm[0, i] = np.linalg.norm(s[0, i])

A, G = create_graph(N, 0.5, type)
# print("Weight matrix A:\n", A)
# print("Graph edges:\n", G)

for k in range(maxIters - 1):
    for i in range(N):
        N_i = np.nonzero(A[i])[0]  
        for j in N_i:
            z[k + 1, i] += A[i, j] * z[k, j]

        z[k + 1, i] -= alpha * s[k, i]

        for j in N_i:
            s[k + 1, i] += A[i, j] * s[k, j]

        _, grad_ell_i_new = cost_fcn(z[k + 1, i], Q[i], r[i])
        ell_i, grad_ell_i_old = cost_fcn(z[k, i], Q[i], r[i])
        s[k + 1, i] += grad_ell_i_new - grad_ell_i_old

        grad_norm[k + 1, i] = np.linalg.norm(grad_ell_i_new)  # Store gradient norm
        cost[k] += ell_i  # accumulate global cost

z_avg = np.mean(z, axis=1)

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

plt.tight_layout()
plt.show()

