import matplotlib.pyplot as plt
import numpy as np
import random
import time



ADU = np.linspace(100,500,401)
ADU_bin = np.linspace(0,501,502)
prob = (ADU-100)**0.5 * np.exp(-(ADU-0)/50)
prob += 1 * np.exp(-((ADU-200))**2/25)
noise_std = 0 # standard dev

photon_fraction = 1 # in %
number_of_frames = 10
Y = 1000
X = 1000
r = 0.1 # radii of charge cloud, in pixels

def generate_data(prob,ADU,ADU_bin,noise_std,photon_fraction,number_of_frames,Y,X,plot):
    data = np.zeros((number_of_frames,Y,X))
    number_of_photons_per_frame = int(Y*X*photon_fraction*0.01)
    all_photons = np.zeros((number_of_photons_per_frame*number_of_frames))

    for frame in range(len(data)):
        print('\r'+'▇'*(frame//2)+"  "+str(int(frame*100//number_of_frames))+'%', end='')
        prob /= np.sum(prob)
        photons = np.zeros((number_of_photons_per_frame))
        for i in range(len(photons)):
            random_number = np.random.choice(ADU, p=prob)
            photons[i] = random_number

        mat = np.zeros((X,Y))
        for y_pix in range(Y):
            #print(mat[y_pix])
            mat[y_pix] = np.random.normal(0,noise_std,X)
        for i_photon in range(number_of_photons_per_frame):
            
            # generate random x,y coordinates
            x_i_c = random.randint(0,X*10-1)
            y_i_c = random.randint(0,Y*10-1)
            x_i = x_i_c//10
            y_i = y_i_c//10
            for x_n in range(x_i-1,x_i+2):
                for y_n in range(y_i-1,y_i+2):
                    S=0.0
                    for p in range(0,10):
                        for q in range(0,10):
                            if (10*x_n+p-x_i_c)**2+(10*y_n+q-y_i_c)**2<=100*r**2:
                                S+=1.0
                    S/=100
                    S/=(np.pi*r**2)
                    mat[y_n][x_n] += photons[i_photon] * S
        data[frame] = mat
        all_photons[frame*number_of_photons_per_frame:(frame+1)*number_of_photons_per_frame] = photons
    
    if plot == 1:
        plt.imshow(data[0])
        plt.show()
        if plot ==1:
            fig = plt.figure()
            ax  = fig.add_subplot(111)
            a = ax.hist(all_photons,bins = ADU_bin)
            h = ax.plot(ADU,prob*number_of_photons_per_frame*number_of_frames)
            plt.show()
    return data



plot = 1
Save = 0
Num_of_files = 1
for File_No in range(0,Num_of_files):
    t1 = time.time()
    print(File_No,'/',Num_of_files)
    data_set = generate_data(prob,ADU,ADU_bin,noise_std,photon_fraction,number_of_frames,Y,X,plot)
    data_set = np.floor(data_set)
    #print(data)
    if Save:
        f = h5py.File("MC_data\\Multi\\noise_%d\\r_%.1f\\frac_%d\\MC_%d.hdf5"%(noise_std,r,photon_fraction,File_No), "w")
        f.create_dataset("data", data=data_set)
        #print(np.array(f["data"]))
    t2 = time.time()
    print('')
    print("Done in",(t2-t1))
        