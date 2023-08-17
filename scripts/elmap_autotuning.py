import numpy as np

def assign_clusters(data, nodes):
    (n_data, n_dims) = np.shape(data)
    (n_nodes, n_dims) = np.shape(nodes)
    clusters = [[] for m in range(n_nodes)]
    for i in range(n_data):
        dists = []
        for j in range(n_nodes):
            dists.append(np.linalg.norm(data[i] - nodes[j]))
        clusters[np.argmin(dists)].append(i)
    return clusters

    
def calc_Uy(data, nodes, weights):
    clusters = assign_clusters(data, nodes)
    cost = 0.
    for i in range(len(clusters)): #i contains index of map_node
        for j in range(len(clusters[i])): #j contains index of traj node of cluster. clusters[i][j] gives the index of node in traj & w
            cost = cost + weights[clusters[i][j]] * np.linalg.norm(data[clusters[i][j]] - nodes[i])
    return cost / np.sum(weights)

def calc_Ue(nodes):
    (n_nodes, n_dims) = np.shape(nodes)
    cost = 0.
    for i in range(n_nodes - 1):
        cost = cost + np.linalg.norm(nodes[i] - nodes[i + 1])
    return cost
    
def calc_Ur(nodes):
    (n_nodes, n_dims) = np.shape(nodes)
    cost = 0.
    for i in range(1, n_nodes - 1):
        cost = cost + np.linalg.norm(nodes[i-1] + nodes[i + 1] - 2 * nodes[i])
    return cost   

def kv_est(nodes, Uy):
    (n_nodes, n_dims) = np.shape(nodes)
    sum_v = 0.
    sum_rho = 0.
    for i in range(1, n_nodes - 1):
        xt1 = nodes[i + 1] - nodes[i]
        xt2 = nodes[i-1] + nodes[i + 1] - 2 * nodes[i]
        if np.linalg.norm(xt1) != 0 and np.linalg.norm(xt2) != 0 and (np.linalg.norm(xt1)**2 * np.linalg.norm(xt2)**2) != (np.dot(xt1, xt2)**2):
            rho = np.linalg.norm(xt1)*3 / (np.linalg.norm(xt1)**2 * np.linalg.norm(xt2)**2 - np.dot(xt1, xt2)**2)**0.5
            sum_v = sum_v + np.linalg.norm(xt1)
            sum_rho = sum_rho + rho**(1/3)
    kv = (sum_v - Uy) / sum_rho
    return kv
    
def estimate_stretch_bend(data, init, weights=None):
    (n_data, n_dims) = np.shape(data)
    if weights is None:
        weights = np.ones((n_data, 1))
        
    sf = np.amax(data) - np.amin(data)
        
    uy = calc_Uy(data, init, weights)
    ue_est = calc_Ue(init)
    ur_est = calc_Ur(init)
    stretch = 5 * (uy / ue_est)
    bend = 5 * (uy / ur_est)
    #stretch = 1.2 * sf * (uy / ue_est)
    #bend = 1.1 * sf * (uy / ur_est)
    return stretch, bend
    
def estimate_crv(data, init, weights=None):
    (n_data, n_dims) = np.shape(data)
    if weights is None:
        weights = np.ones((n_data, 1))
        
    #sf = np.amax(data) - np.amin(data)
        
    uy = calc_Uy(data, init, weights)
    crv = kv_est(init, uy)
    return crv
    