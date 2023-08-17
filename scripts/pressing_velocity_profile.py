import numpy as np
import math
import matplotlib.pyplot as plt
from elmap import *

def repro():
    data = np.loadtxt("../pressing/reproduction/repro.txt")
    data = downsample_traj(data, 50)
    magnitudes = []
    for i in range(0, len(data)-1):
        result = data[i+1] - data[i]
        result2 = math.sqrt(sum(result**2))
        magnitudes.append(result2)
    
    fig = plt.figure()
    plt.plot(magnitudes, 'k', lw=3, label='velocity')

    data1 = np.loadtxt("../pressing/reproduction/repro_uh.txt")
    data1 = downsample_traj(data1, 50)
    magnitudes1 = []
    for i in range(0, len(data1)-1):
        result = data1[i+1] - data1[i]
        result2 = math.sqrt(sum(result**2))
        magnitudes1.append(result2)
    
    data2 = np.loadtxt("../pressing/succ2_demo.txt")
    data2 = downsample_traj(data2, 50)
    magnitudes2 = []
    for i in range(0, len(data2)-1):
        result = data2[i+1] - data2[i]
        result2 = math.sqrt(sum(result**2))
        magnitudes2.append(result2)

    fig = plt.figure()
    plt.plot(magnitudes, 'b', lw=3, label='velocity profile without Uh')
    plt.plot(magnitudes1, 'r', lw=3, label='velocity profile with Uh')
    plt.plot(magnitudes2, 'k', lw=3, label='velocity')

    plt.grid(True)
    plt.legend()
    fig.savefig("../pressing/reproduction/" + 'velocity_profile.png', bbox_inches='tight', dpi=300)
    plt.close(fig)

def main():
    repro()

if __name__ == "__main__":
    main()

