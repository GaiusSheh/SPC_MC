import matplotlib.pyplot as plt
import numpy as np
import random
import time
from numba import jit, prange

import matplotlib as mpl

mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
mpl.rcParams['font.size'] = 10
mpl.rcParams['figure.figsize'] = (4, 3)
mpl.rcParams['figure.dpi'] = 300
plt.rcParams['lines.linewidth'] = 1.0

# Define constants
ADU = np.linspace(100, 500, 401)
ADU_bin = np.linspace(0, 501, 502)
prob = (ADU - 100) ** 0.5 * np.exp(-(ADU - 0) / 30)
prob += 1 * np.exp(-((ADU - 200)) ** 2 / 25)

photon_fraction = 5  # in %
number_of_frames = 100
Y, X = 1000, 1000
r = 0.4  # radius of charge cloud, in pixels
fine_grid = 100
noise_level = 5 # standard deviation of the noise, in ADU

@jit(nopython=True, parallel=True)
def distribute_photons(photons, fine_grid, r, X, Y):
    mat = np.zeros((Y, X))
    fine_r = r * fine_grid

    # Precompute subpixel offsets
    subpixel_offsets = np.array([(x, y) for x in range(fine_grid) for y in range(fine_grid)])

    for photon in photons:
        x_i_c = random.randint(1 * fine_grid, (X - 1) * fine_grid)
        y_i_c = random.randint(1 * fine_grid, (Y - 1) * fine_grid)
        x_i, y_i = x_i_c // fine_grid, y_i_c // fine_grid
        # print(x_i, y_i, photon)

        #define a matrix to store the overlap of the charge cloud with the pixels
        S_mat = np.zeros((min(X, x_i + 2) - max(0, x_i - 1), min(Y, y_i + 2) - max(0, y_i - 1)))

        for x_n in range(max(0, x_i - 1), min(X, x_i + 2)):
            for y_n in range(max(0, y_i - 1), min(Y, y_i + 2)):
                # Compute distances for all subpixels at once
                distances = np.sqrt((x_i_c - x_n * fine_grid - subpixel_offsets[:, 0]) ** 2 +
                                    (y_i_c - y_n * fine_grid - subpixel_offsets[:, 1]) ** 2)
                # store the overlap of the charge cloud with the pixels to the matrix
                S_mat[x_n - max(0, x_i - 1)][y_n - max(0, y_i - 1)] = np.sum(distances <= fine_r)
        
        S_total = np.sum(S_mat)
        # loop again to add the charge to the pixels
        for x_n in range(max(0, x_i - 1), min(X, x_i + 2)):
            for y_n in range(max(0, y_i - 1), min(Y, y_i + 2)):
                ratio = S_mat[x_n - max(0, x_i - 1)][y_n - max(0, y_i - 1)] / S_total
                mat[x_n][y_n] += photon * ratio
                
        # print(temp_mat)  
    return mat

def generate_data(prob, ADU, ADU_bin, photon_fraction, number_of_frames, Y, X, plot):
    data = np.zeros((number_of_frames, Y, X))
    number_of_photons_per_frame = int(Y * X * photon_fraction * 0.01)
    all_photons = np.zeros(number_of_photons_per_frame * number_of_frames)

    # Normalize probabilities
    prob /= np.sum(prob)

    for frame in range(number_of_frames):
        print(f'\r{"▇" * ((frame)* 20//number_of_frames)}  {int((frame) * 100 // number_of_frames)}%', end='')

        photons = np.random.choice(ADU, size=number_of_photons_per_frame, p=prob)
        data[frame] = distribute_photons(photons, fine_grid, r, X, Y)

        # add random noise to the data
        data[frame] += np.random.normal(0, noise_level, (Y, X))

        all_photons[frame * number_of_photons_per_frame:(frame + 1) * number_of_photons_per_frame] = photons

        # if plot the frame, only do once
        if plot and frame == 0:
            plt.figure()
            plt.imshow(data[0])
            plt.colorbar()
            if X >= 200:
                plt.xlim(99.5, 199.5)
                plt.ylim(99.5, 199.5)
            elif 100 < X < 200:
                plt.xlim(X-100.5, X-0.5)
                plt.ylim(X-100.5, X-0.5)
            plt.title('Simulated Frame')
            plt.show()

            plt.figure()
            plt.hist(all_photons[all_photons>0], bins=ADU_bin, alpha=0.7, label='Photon distribution')
            plt.plot(ADU, prob * number_of_photons_per_frame, label='Probability distribution')
            plt.legend()
            plt.title('Photon ADU Distribution')
            plt.show()
        # change a line if finished, move the progress bar to 100%
        if frame == number_of_frames - 1:
            print(f'\r{"▇" * 20//number_of_frames}  {100}%', end='')
            print()


    return data

plot = True
save_data = True
num_of_files = 1

for file_no in range(num_of_files):
    t1 = time.time()
    print(f'{file_no}/{num_of_files}')
    data_set = generate_data(prob, ADU, ADU_bin, photon_fraction, number_of_frames, Y, X, plot)
    data_set = np.floor(data_set)

    if save_data:
        np.save(f'data_set_{file_no}.npy', data_set)

    t2 = time.time()
    print(f'\nDone in {t2 - t1:.2f} seconds')