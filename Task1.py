import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

np.random.seed(0)

def metropolis_hastings_weights(A):
    N = A.shape[0]
    deg = np.sum(A, axis=1)
    A_mh = np.zeros((N, N))
    # print("Degrees:", deg)
    
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
    A = metropolis_hastings_weights(A_unweighted)

    return A, G

def cost_fcn(zz, QQ, rr): 
    val = 0.5 * zz.T @ QQ @ zz + rr.T @ zz
    grad = QQ @ zz + rr
    # print("gradient shape:", grad.shape)
    return val, grad

d = 3  # dimension of the decision variable z
N = 10  # number of agents
p_er = 0.5  # probability for Erdos-Renyi graph
type = 'random'  # type of graph
maxIters = 500
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

for i in range(N):
    _, s[0, i] = cost_fcn(z[0, i], Q[i], r[i])

A, G = create_graph(N, 0.5, type)
print("Weight matrix A:\n", A)

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

        cost[k] += ell_i  # accumulate global cost

z_avg = np.mean(z, axis=1)

fig, axes = plt.subplots(figsize=(8, 6), nrows=2, ncols=2)
ax = axes[0, 0]
# ax.semilogy(np.arange(maxIters - 1), np.abs(cost[:-1] - cost_opt))
nx.draw_kamada_kawai(G, with_labels=True, ax=ax)
# ax.plot(np.arange(maxIters - 1), cost_opt * np.ones((maxIters - 1)), "r--")
ax = axes[1, 0]
ax.plot(np.arange(maxIters - 1), cost[:-1])

ax = axes[1, 1]
for i in range(N):
    ax.semilogy(np.arange(maxIters), np.abs(z[:, i] - z_avg))
    # ax.plot(np.arange(maxIters), z[:, i] - z_avg)


plt.show()

