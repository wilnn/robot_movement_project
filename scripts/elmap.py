import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

#function to downsample a 1 dimensional trajectory to n points
#arguments
#traj: nxd vector, where n is number of points and d is number of dims
#n (optional): the number of points in the downsampled trajectory. Default is 100.
#returns the trajectory downsampled to n points
def downsample_traj(traj, n=100):
    n_pts, n_dims = np.shape(traj)
    npts = np.linspace(0, n_pts - 1, n)
    out = np.zeros((n, n_dims)) 
    for i in range(n):
        out[i][:] = traj[int(npts[i])][:]
    return out
    

class elastic_map(object):

    def __init__(self, given_data, init=None, stretch=1.0, bend=1.0, crv=1.0, termination='iter', termination_condition=10):
        self.traj = given_data
        (self.n_data, self.n_dims) = np.shape(self.traj)
        
        self.w = np.ones((self.n_data))
        
        self.term = termination
        self.term_cond = termination_condition
        
        self.map_nodes = np.vstack((self.traj[0], self.traj[-1])) if init is None else init
        self.map_size = 2 if init is None else len(init)
        
        self.lmbda = stretch
        self.mu = bend
        self.kv = crv
        
        self.iter = 0
        
    def calc_grid_uh(self, plot):
        if (plot):
            self.fig = plt.figure()
            plt.plot(self.traj[:, 0], self.traj[:, 1], 'k', lw=5)
            l, =  plt.plot(self.map_nodes[:, 0], self.map_nodes[:, 1], 'r.-', lw=4, ms=20)
            self.fig.canvas.draw_idle()
            plt.pause(0.001)
            #print('Press Enter to start')
            #input()
            
        while not self.is_termination():
            self.iter = self.iter + 1
            print("Iter: ", self.iter)
            if (plot):
                l.set_xdata(self.map_nodes[:, 0])
                l.set_ydata(self.map_nodes[:, 1])
                self.fig.canvas.draw_idle()
                plt.pause(0.001)
            
            #self.insert_node(plot)
            self.assign_clusters()
            self.optimize_map_uh()
        self.iter = 0
        if (plot):
            print('Finished!')
            plt.show()
        
        return self.map_nodes
    
    def calc_grid(self, plot):
        if (plot):
            self.fig = plt.figure()
            plt.plot(self.traj[:, 0], self.traj[:, 1], 'k', lw=5)
            l, =  plt.plot(self.map_nodes[:, 0], self.map_nodes[:, 1], 'r.-', lw=4, ms=20)
            self.fig.canvas.draw_idle()
            plt.pause(0.001)
            #print('Press Enter to start')
            #input()
            
        while not self.is_termination():
            self.iter = self.iter + 1
            print("Iter: ", self.iter)
            if (plot):
                l.set_xdata(self.map_nodes[:, 0])
                l.set_ydata(self.map_nodes[:, 1])
                self.fig.canvas.draw_idle()
                plt.pause(0.001)
        
            #self.insert_node(plot)
            self.assign_clusters()
            self.optimize_map()
        self.iter = 0
        if (plot):
            print('Finished!')
            plt.show()
        
        return self.map_nodes
		
    def assign_clusters(self):
        self.clusters = [[] for m in range(self.map_size)]
        for i in range(self.n_data):
            dists = []
            for j in range(self.map_size):
                dists.append(np.linalg.norm(self.traj[i] - self.map_nodes[j]))
            self.clusters[np.argmin(dists)].append(i)
    '''    
    def calc_largest_load(self):
        #print(self.clusters)
        edge_loads=[]
        for j in range(self.map_size - 1):
            edge_loads.append(len(self.clusters[j][:]) + len(self.clusters[j+ 1][:]))
        print(edge_loads)
        return np.argmax(edge_loads)
        
    def insert_node(self, plot):
        self.assign_clusters()
        edge_num = self.calc_largest_load()
        edge = self.map_nodes[edge_num:edge_num+2]
        print('edge')
        print(edge)
        print('')
        #if (plot):
        #    l2, = plt.plot(edge[:, 0], edge[:, 1], 'b.-', lw=3, ms=15)
        #    self.fig.canvas.draw_idle()
        #    plt.pause(0.001)
        #    l2.set_xdata([])
        #    l2.set_ydata([])
        #    self.fig.canvas.draw_idle()
        #    plt.pause(0.001)
        new_node_pos = np.mean(edge, axis=0)
        self.map_nodes = np.insert(self.map_nodes, edge_num+1, new_node_pos, axis=0)
        self.map_size = self.map_size + 1
    '''
    
    def calc_Uy(self):
        cost = 0.
        for i in range(len(self.clusters)): #i contains index of map_node
            for j in range(len(self.clusters[i])): #j contains index of traj node of cluster. clusters[i][j] gives the index of node in traj & w
                cost = cost + self.w[self.clusters[i][j]] * np.linalg.norm(self.traj[self.clusters[i][j]] - self.map_guess[i])
        return cost / np.sum(self.w) 
        
    def calc_Ue(self):
        cost = 0.
        for i in range(self.map_size - 1):
            cost = cost + np.linalg.norm(self.map_guess[i] - self.map_guess[i + 1])
        return cost *  self.lmbda
        
    def calc_Ur(self):
        cost = 0.
        for i in range(1, self.map_size - 1):
            cost = cost + np.linalg.norm(self.map_guess[i-1] + self.map_guess[i + 1] - 2 * self.map_guess[i])
        return cost * self.mu
    
    def calc_Uh(self):
        cost = 0.
        for i in range(1, self.map_size - 1):
            xt1 = self.map_guess[i + 1] - self.map_guess[i]
            xt2 = self.map_guess[i-1] + self.map_guess[i + 1] - 2 * self.map_guess[i]
            if np.any(xt2 == 0):
                continue
            cost = cost + np.linalg.norm(xt1 - self.kv* np.abs(((1 + xt1**2)**(3/2))/xt2)**(1/3))
        return cost
        
    def calc_Uh2(self):
        cost = 0.
        for i in range(1, self.map_size - 1):
            xt1 = self.map_guess[i + 1] - self.map_guess[i]
            xt2 = self.map_guess[i-1] + self.map_guess[i + 1] - 2 * self.map_guess[i]
            if np.linalg.norm(xt1) != 0 and np.linalg.norm(xt2) != 0 and (np.linalg.norm(xt1)**2 * np.linalg.norm(xt2)**2) != (np.dot(xt1, xt2)**2):
                rho = np.linalg.norm(xt1)*3 / (np.linalg.norm(xt1)**2 * np.linalg.norm(xt2)**2 - np.dot(xt1, xt2)**2)**0.5
                cost = cost + np.abs(np.linalg.norm(xt1) - self.kv * rho**(1/3))
        return cost
    
    def calc_costs(self, X):
        self.map_guess = np.reshape(X, ((self.map_size, self.n_dims)))
        return self.calc_Ue() + self.calc_Ur() + self.calc_Uy()

    def calc_costs_uh(self, X):
        self.map_guess = np.reshape(X, ((self.map_size, self.n_dims)))
        #return self.calc_Ue() + self.calc_Ur() + self.calc_Uy() + self.calc_Uh()
        return self.calc_Uy() + self.calc_Uh2()
    
    def optimize_map(self):
        init_guess = np.reshape(self.map_nodes, ((self.map_size*self.n_dims, )))
        res = minimize(self.calc_costs, init_guess, tol=1e-2)
        self.map_nodes = np.reshape(res.x, ((self.map_size, self.n_dims)))
    
    def optimize_map_uh(self):
        init_guess = np.reshape(self.map_nodes, ((self.map_size*self.n_dims, )))
        res = minimize(self.calc_costs_uh, init_guess, tol=1e-2)
        self.map_nodes = np.reshape(res.x, ((self.map_size, self.n_dims)))

    def is_termination(self):
        if (self.term == 'iter'):
            return self.iter >= self.term_cond
        print('No termination found!')
        return False


def main_single_demo():
    N = 1000
    t = np.linspace(0, 10, N).reshape((N, 1))
    x = np.sin(t) + 0.01 * t**2 - 0.05 * (t-5)**2
    traj = np.hstack((t, x))
    
    traj_ds = downsample_traj(traj, n=25)
    elmap = elastic_map(given_data=traj, init=traj_ds, stretch=0.01, bend=0.05)
    elmap.calc_grid(plot=True)

if __name__ == '__main__':
    main_single_demo()