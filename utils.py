import numpy as np
import similaritymeasures
import h5py
import matplotlib.pyplot as plt

def save_plot(fpath, demo, repro_uh, repro):
    fig = plt.figure()
    plt.plot(demo[:, 0], demo[:, 1], 'k', lw=3, label='Demonstration')
    plt.plot(repro_uh[:, 0], repro_uh[:, 1], 'r', lw=3, label='Reproduction with Uh')
    plt.plot(repro[:, 0], repro[:, 1], 'b', lw=3, label='Reproduction without Uh')

    plt.grid(True)
    plt.legend()
    fig.savefig(fpath + 'plot.png', bbox_inches='tight', dpi=300)
    plt.close(fig)

def get_lasa_trajN(shape_name, n=1):
    #ask user for the file which the playback is for
    #filename = raw_input('Enter the filename of the .h5 demo: ')
    #open the file
    filename = 'lasa_dataset.h5'
    hf = h5py.File(filename, 'r')
    #navigate to necessary data and store in numpy arrays
    shape = hf.get(shape_name)
    demo = shape.get('demo' + str(n))
    pos_info = demo.get('pos')
    pos_data = np.array(pos_info)
    y_data = np.delete(pos_data, 0, 1)
    x_data = np.delete(pos_data, 1, 1)
    #close out file
    hf.close()
    x = np.reshape(x_data, ((len(x_data), 1)))
    y = np.reshape(y_data, ((len(y_data), 1)))
    return np.hstack((x, y))
    
def align_sse(exp_data, num_data):
    #print(np.shape(exp_data))
    #print(np.shape(num_data))
    exp = exp_data - exp_data[0]
    num = num_data - num_data[0]
    #assume exp data is always larger
    new_exp = [exp[0]]
    indeces = [i for i in range(1, len(exp) - 1)]
    for i in range(1, len(num) - 1):
        best_ind = indeces[0]
        best_val = np.linalg.norm(exp[indeces[0]] - num[i])
        for j in indeces:
            val = np.linalg.norm(exp[j] - num[i])
            if val < best_val:
                best_val = val
                best_ind = j
        #print([best_ind, exp[best_ind]])
        new_exp.append(exp[best_ind])
        indeces.remove(best_ind)
    new_exp.append(exp[-1])
    new_exp = np.array(new_exp)
    #plt.figure()
    #plt.plot(exp[:, 0], exp[:, 1], 'r.')
    #plt.plot(new_exp[:, 0], new_exp[:, 1], 'g.')
    #plt.plot(num[:, 0], num[:, 1], 'b.')
    #plt.show()
    return sum_of_squared_error(new_exp, num) 

def sum_of_squared_error(exp_data, num_data):
    #naive approach
    (n_points, n_dims) = np.shape(exp_data)
    if not np.shape(exp_data) == np.shape(num_data):
        print('Array dims must match!')
    sum = 0.
    for i in range(n_points):
        sum += (np.linalg.norm(exp_data[i] - num_data[i]))**2
    return sum
    
def calc_jerk(traj):
    (n_pts, n_dims) = np.shape(traj)
    ttl = 0.
    for i in range(2, n_pts - 2):
        ttl += np.linalg.norm(traj[i - 2] + 2*traj[i - 1] - 2 * traj[i+1] - traj[i+2])
    return ttl
    
def align_ang_sim(exp_data, num_data):
    #print(np.shape(exp_data))
    #print(np.shape(num_data))
    exp = exp_data - exp_data[0]
    num = num_data - num_data[0]
    #assume exp data is always larger
    new_exp = [exp[0]]
    indeces = [i for i in range(1, len(exp) - 1)]
    for i in range(1, len(num) - 1):
        best_ind = indeces[0]
        best_val = np.linalg.norm(exp[indeces[0]] - num[i])
        for j in indeces:
            val = np.linalg.norm(exp[j] - num[i])
            if val < best_val:
                best_val = val
                best_ind = j
        #print([best_ind, exp[best_ind]])
        new_exp.append(exp[best_ind])
        indeces.remove(best_ind)
    new_exp.append(exp[-1])
    new_exp = np.array(new_exp)
    #plt.figure()
    #plt.plot(exp[:, 0], exp[:, 1], 'r.')
    #plt.plot(new_exp[:, 0], new_exp[:, 1], 'g.')
    #plt.plot(num[:, 0], num[:, 1], 'b.')
    #plt.show()
    return angular_similarity(new_exp, num)    

def angular_similarity(exp_data, num_data):
    (n_points, n_dims) = np.shape(exp_data)
    #print(np.shape(exp_data))
    #print(np.shape(num_data))
    if not (np.shape(exp_data) == np.shape(num_data)):
        print('Array dims must match!')
    sum = 0.
    for i in range(n_points - 1):
        v1 = exp_data[i + 1] - exp_data[i]
        v2 = num_data[i + 1] - num_data[i]
        #calc cosine similarity
        costheta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        #calc angular distance
        ang_dist = np.arccos(costheta) / np.pi if not np.isnan(costheta) else 0
        sum = sum + ang_dist
    return sum / n_points
    
def calc_sim_measures(exp_data, num_data):
    sse = align_sse(exp_data, num_data)
    frech = similaritymeasures.frechet_dist(exp_data, num_data)
    ang = align_ang_sim(exp_data, num_data)
    jerk = calc_jerk(num_data)
    return [sse, frech, ang, jerk]