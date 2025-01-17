import matplotlib.pyplot as plt
import numpy as np
import random
import h5py
import time
import os




def mkdir(path):
    path=path.strip()
    path=path.rstrip("\\")
    isExists=os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return True
    else:
        return False

ADU = np.linspace(100,500,401)
ADU_bin = np.linspace(0,501,502)
prob = 0.1*(ADU-90)**0.5 * np.exp(-(ADU-0)/50)
prob += 0.2 * np.exp(-((ADU-120))**2/25)
#prob += 0.1 * np.exp(-((ADU-180))**2/25)
prob += 0.5 * np.exp(-((ADU-300))**2/25)
noise_std = 3 # standard dev
photon_fractions = [1,2,3,4,5]


number_of_frames = 10
Y = 704
X = 768
r = 0.4 # radii of charge cloud, in pixels
gran = 50
gran_inv = 1/gran

def generate_data(prob,ADU,ADU_bin,noise_std,photon_fraction,number_of_frames,Y,X,plot):
    data = np.zeros((number_of_frames,Y,X))
    number_of_photons_per_frame = int(Y*X*photon_fraction*0.01)
    all_photons = np.zeros((number_of_photons_per_frame*number_of_frames))

    for frame in range(len(data)):
        print('\r'+'▋'*(frame*10//number_of_frames)+"  "+str(frame*100//number_of_frames)+'%', end='')
        prob /= np.sum(prob)
        photons = np.zeros((number_of_photons_per_frame))
        for i in range(len(photons)):
            random_number = np.random.choice(ADU, p=prob)
            photons[i] = random_number

        mat = np.zeros((Y,X))
        for y_pix in range(Y):
            #print(mat[y_pix])
            mat[y_pix] =  np.random.normal(0,noise_std,X)
        for i_photon in range(number_of_photons_per_frame):
            y_i_c = random.randrange(gran*1,gran*703)
            x_i_c = random.randrange(gran*1,gran*767)
            x_i = x_i_c//gran
            y_i = y_i_c//gran
            for x_n in range(x_i-1,x_i+2):
                for y_n in range(y_i-1,y_i+2):
                    S=0.0
                    for p in range(0,gran):
                        for q in range(0,gran):
                            if (x_n+gran_inv*p-gran_inv*x_i_c)**2+(y_n+gran_inv*q-gran_inv*y_i_c)**2<=r**2:
                                S+=1.0
                    S/=gran**2
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


plot = 0
Save = 1
Num_of_files = 6
start_num = 5

for photon_fraction in photon_fractions:
    path = "22Jan\\r_%s\\Noise_%s\\frac_%s"%(str(r),str(noise_std),str(photon_fraction))
    print(photon_fraction)
    try:
        mkdir(path)
    except:
        print("path exist")
    for File_No in range(start_num,Num_of_files):
        t1 = time.time()
        print(File_No,'/',Num_of_files)
        data_set = generate_data(prob,ADU,ADU_bin,noise_std,photon_fraction,number_of_frames,Y,X,plot)
        data_set = np.floor(data_set)
        #print(data)
        if Save:
            f = h5py.File(path+"\\MC_%d.hdf5"%(File_No), "w")
            f.create_dataset("data", data=data_set)
            #print(np.array(f["data"]))
        t2 = time.time()
        print("")
        print("Done in %.1f"%(t2-t1))
        print("")

        
