import matplotlib.pyplot as plt
import h5py
import numpy as np

run = 4 # give a set of run numbers
noise = 3
frac = 1

show_switch = 2

if show_switch ==1:
    path_new = "SPC data analysis\\New\\r_0.4\\noise_%s\\frac_%s\\%s\\"%(str(noise),str(frac),str(run))
    f = h5py.File(path_new+"MC_%d_valid.hdf5"%(run),'r') 
if show_switch == 2:
    path = "New_data\\r_0.4\\Noise_%s\\frac_%s\\"%(str(noise),str(frac))
    f = h5py.File(path+"MC_%d.hdf5"%(run),'r')
dataset = f['data'] 

A = np.asarray(dataset).reshape(-1) # to make a histogram of the raw data
plt.figure()
hist = plt.hist(A,bins=500,range=(-100,400)) # to make a histogram of the raw data. Limit set to -100 ADU
plt.xlabel('# of ADUs') # to generate a histogram, not in use any longer
plt.ylabel('Frequency')
plt.yscale("log")
plt.show()
plt.close()

plt.imshow(dataset[0])
plt.show()