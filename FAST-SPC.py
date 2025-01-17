import matplotlib.pyplot as plt
import matplotlib as mpl
import math
import numpy as np
from numpy import genfromtxt
import csv
import pandas
import time
import json
import os
import time as t
from numba import jit, prange
from typing import List
from pathlib import Path


def load_data(run_num):
    f = np.load(f"MC_charge_spread/MC_DATA/data_set_{run_num}.npy")
    return f

def auto_threshold(dataset, bin_min, bin_max):
    A = np.asarray(dataset).reshape(-1)
    hist = np.histogram(A,bins=bin_max-bin_min,range=(bin_min,bin_max))
    
    plt.hist(A,bins=bin_max-bin_min,range=(bin_min,bin_max))
    plt.xlabel("ADU")
    plt.ylabel("Pixel counts")
    plt.yscale('log')
    plt.show()


    Threshold2 = 0  # 3 sigma of the background noise

    # find the maximum index of the histogram using argmax
    max_index = np.argmax(hist[0])

    for i in range(max_index,-2*bin_min):
        if hist[0][i] < np.exp(-1/2)*hist[0][max_index]: # sigma level
            Threshold2 = 3*hist[1][i] + 1
            #print(Threshold2)
            break
            
    print('3 sigma: %d' %(Threshold2))
    return Threshold2

single_kernel = np.array([[0, 0, 0],
                          [0, 1, 0],
                          [0, 0, 0]])

single_checker = np.array([[1, 1, 1],
                           [1, 0, 1],
                           [1, 1, 1]])

single_kernels = [[single_kernel, single_checker]]

# Double pixel kernels for horizontal and vertical adjacency
double_kernel = np.array([[0, 0, 0, 0],
                          [0, 1, 1, 0],
                          [0, 0, 0, 0]])

double_checker = np.array([[1, 1, 1, 1],
                           [1, 0, 0, 1],
                           [1, 1, 1, 1]])

double_kernels = [
    [
        double_kernel,
        double_checker
    ],
    [
        np.rot90(double_kernel),
        np.rot90(double_checker)
    ]
]

# Triple pixel kernels for L-shape adjacency
triple_kernel = np.array([[0, 0, 0, 0],
                          [0, 1, 1, 0],
                          [0, 0, 1, 0],
                          [0, 0, 0, 0]])

triple_checker = np.array([[1, 1, 1, 1],
                           [1, 0, 0, 1],
                           [1, 1, 0, 1],
                           [0, 1, 1, 1]])

triple_kernels = [
    [
        triple_kernel,
        triple_checker
    ],
    [
        np.rot90(triple_kernel, k=1),
        np.rot90(triple_checker, k=1)
    ],
    [
        np.rot90(triple_kernel, k=2),
        np.rot90(triple_checker, k=2)
    ],
    [
        np.rot90(triple_kernel, k=3),
        np.rot90(triple_checker, k=3)
    ]
]

# Quad pixel kernels for L-shape adjacency
quad_kernel = np.array([[0, 0, 0, 0],
                        [0, 1, 1, 0],
                        [0, 1, 1, 0],
                        [0, 0, 0, 0]])

quad_checker = np.array([[1, 1, 1, 1],
                         [1, 0, 0, 1],
                         [1, 0, 0, 1],
                         [1, 1, 1, 1]])

quad_kernels = [
    [
        quad_kernel,
        quad_checker
    ]
]

# store all kernels and checkers in a dictionary
kernels = {
    "single": single_kernels,
    "double": double_kernels,
    "triple": triple_kernels,
    "quad": quad_kernels
        }

def check_roi(roi, kernel_group_name, kernel_index, kernels = kernels):
    kernel, checker = kernels[kernel_group_name][kernel_index]
    # Check if every non-zero element in the kernel is also non-zero in the corresponding location in the ROI
    if not (np.all(roi[kernel > 0] > 0) and ((roi * checker).sum() == 0)):
        return False
    elif kernel_group_name == "single" or kernel_group_name == "double":
        return True
    elif kernel_group_name == "triple":
        roi_rot = np.rot90(roi, k=-kernel_index)
        return roi_rot[1][2] > roi_rot[1][1] and roi_rot[1][2] > roi_rot[2][2]
    elif kernel_group_name == "quad":
        min_val, max_val = np.min(roi[1:3, 1:3]), np.max(roi[1:, 1:3])
        min_pos = np.argwhere(roi == min_val)[0]
        max_pos = np.argwhere(roi == max_val)[0]
        # Check that the min and max are diagonally positioned
        if np.abs(max_pos[0] - min_pos[0]) == np.abs(max_pos[1] - min_pos[1]) == 1:
            # Check that the other two values are smaller than the max
            adj_vals = [roi[1, 1], roi[1, 2], roi[2, 1], roi[2, 2]]
            adj_vals.remove(min_val)
            adj_vals.remove(max_val)
            return adj_vals[0] < max_val and adj_vals[1] < max_val
        else:
            return False
        
        
def cluster(image, thresh_1, thresh_2, kernels = kernels):
    single_clusters: List[float] = []
    double_clusters: List[float] = []
    triple_clusters: List[float] = []
    quad_clusters: List[float] = []
    
    cluster_lists = {
        "single": single_clusters,
        "double": double_clusters,
        "triple": triple_clusters,
        "quad": quad_clusters,
    }
    
    if not np.any(image > thresh_1/4):
        # If none of the pixels are over thresh_2, no clusters present
        return cluster_lists
    
    # Only need to loop over spots that are non-zero and not at the edges
    guesses = np.argwhere((image[2:-2, 2:-2] > thresh_1//4))
    #print(guesses)
    
    for x_shift, y_shift in guesses:
        x = x_shift + 2
        y = y_shift + 2
        for kernel_group_name, kernel_group in kernels.items():
            for kernel_index, (kernel_mat, checker) in enumerate(kernel_group):
                padding_x = kernel_mat.shape[0] // 2  # Half kernel width in x direction
                padding_y = kernel_mat.shape[1] // 2  # Half kernel width in y direction
                
                # Region of interest
                roi = image[
                    x - padding_x : x - padding_x + kernel_mat.shape[0],
                    y - padding_y : y - padding_y + kernel_mat.shape[1]
                ]

                energy = (roi * kernel_mat).sum()  # Energy given by kernel convolution

                if energy > thresh_1 and check_roi(roi, kernel_group_name, kernel_index):
                    cluster_lists[kernel_group_name].append(energy)
                        
    return cluster_lists


thresh_1 = 90
ADU_min = -10
ADU_max = 500


# detector area limits
y_min=0
y_max=1000
x_min=0
x_max=1000

# image selection
first_img = None
last_img = None
step_img = 1

# output dir
out_path = Path("MC_charge_spread/SPC_outputs")

run = 14
t_pre = t.time()
#try:
dataset = load_data(run)
#except:
    #print("\tNot found. Skipping...")
    #continue
t_0 = t.time()
print(f"Data loading takes {round( t_0 - t_pre, 2)}s")
thresh_2 = auto_threshold(dataset,bin_min=ADU_min, bin_max=ADU_max)
t_1 = t.time()
print(f"Auto-threshold takes {round(t_1 - t_0, 2)}s")


run_details = []
for i, image in enumerate(dataset):
    print(f'\r{"▋" * (i*20//len(dataset))}  {int((i) * 100 // len(dataset))}%', end='')
    image = image[y_min:y_max, x_min:x_max]  # crop to detector area
    image[image < thresh_2] = 0  # mask out background
    if np.any(image > 0):
        shot_data = cluster(image, thresh_1, thresh_2)
    run_details.append(shot_data)
    # change a line if finished, move the progress bar to 100%
    if i == len(dataset) - 1:
        print(f'\r{"▋" * (20//len(dataset))}  {100}%', end='')

t_2 = t.time()
print("")
print(f"Clustering takes {round(t_2 - t_1, 2)}s")
with open(out_path / f"{str(run).zfill(4)}_photons.json", "w", encoding="utf-8") as out_file:
    json.dump(run_details, out_file)
t_3 = t.time()
print(f"Saving takes {round(t_3 - t_2, 2)}s")
print("")
