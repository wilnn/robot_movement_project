import numpy as np
import os
import matplotlib.pyplot as plt
from elmap import *
from utils import *

def main():
    # define testing data
    #
    lasa_names = ['Angle','BendedLine','CShape','DoubleBendedLine','GShape', \
                'heee','JShape','JShape_2','Khamesh','Leaf_1', \
                'Leaf_2','Line','LShape','NShape','PShape', \
                'RShape','Saeghe','Sharpc','Sine','Snake', \
                'Spoon','Sshape','Trapezoid','Worm','WShape', \
                'Zshape']
    
    #lasa_names = ['Angle']
    num_demo = 1
    demo_points = 250 #M
    
    # define parameters
    stretch_const = 0.01 #lambda
    bend_const = 1.4 #mu
    num_points = 50 #N
    
    # for each shape
    for num_demo in range(1, 2):
        for name in lasa_names:
            # create save folder
            save_fpath = './experiment_Uh2_26/' + name + str(num_demo) + '/'
            try:
                os.makedirs(save_fpath)
            except OSError:
                print ("Creation of the directory %s failed" % save_fpath)
            else:
                print ("Successfully created the directory %s" % save_fpath)
                
            # get lasa data
            demo = get_lasa_trajN(name, num_demo)
            
            # get reproduction
            #demo_ds = downsample_traj(demo, n=demo_points)
            init_ds = downsample_traj(demo, n=num_points)
            EM = elastic_map(given_data=demo, init=init_ds, stretch=stretch_const, bend=bend_const)
            repro_uh = EM.calc_grid_uh(plot=False)
            repro = EM.calc_grid(plot = False)

            # measure reproduction
            [sse, frech, ang, jerk] = calc_sim_measures(demo, repro_uh)
            [sse1, frech1, ang1, jerk1] = calc_sim_measures(demo, repro)

            # save results
            np.savetxt(save_fpath + 'repro_uh.txt', repro_uh) #reproduction
            np.savetxt(save_fpath + 'repro.txt', repro)
            fp = open(save_fpath + 'parameters_results.txt', 'w')
            fp.writelines(['Shape, Demo_number, Demo_points, Stretch, Bend, N, SSE, Frechet, Angular, Jerk\n', 
                           'With Uh:\n', 
                            str([name, num_demo, demo_points, stretch_const, bend_const, num_points, sse, frech, ang, jerk]).replace('[', '').replace(']', '') + '\n',
                            'without uh:\n',
                            str([name, num_demo, demo_points, stretch_const, bend_const, num_points, sse1, frech1, ang1, jerk1]).replace('[', '').replace(']', '') + '\n']) #parameters & metrics
            fp.close()
            save_plot(save_fpath, demo, repro_uh, repro) #plot

if __name__ == '__main__':
    main()