import matplotlib
import matplotlib.pyplot as plt

import math
import numpy as np
from numpy import genfromtxt
import csv
import h5py
from numpy.core.fromnumeric import shape
import pandas
import time
from scipy import integrate
from scipy import optimize

def convolution(a,b):
    ans = 0
    if a.shape[0] == b.shape[0]:
        length = a.shape[0]
        ans = np.convolve(a,b,mode='full')[0:length]
    else:
        print('array length error')
    return ans

def Auto_Convolution(f,n):
    ans = []
    if n == 1:
        ans = f
    if n > 1:
        g = Auto_Convolution(f,n-1)
        #print('g:',g)
        ans = convolution(f,g)
        #print('ans',ans)
    return ans

def A(m,Poisson_lambda):
    return ((-1) ** (m-1)) * (1/ m)

def reconstruct(I,Poisson_Lambda,order):
    ans = I *0
    for n in range(1,order):
        ans +=  (A(n,Poisson_Lambda)/λ) * np.exp(n * Poisson_Lambda) * Auto_Convolution(I,n)
    return ans


def num_d(I,Poisson_Lambda,order):
    ans = np.zeros((len(I),len(I)))
    for i in range(0,len(I)):
        print(i)
        for j in range(0,len(I)):
            Delta_I_j = 0.01 * I[j] + 1E-10
            new_I = I+0
            new_I[j] = I[j] + Delta_I_j
            Delta_f_i = 1.0 * (reconstruct(new_I,Poisson_Lambda,order)[i] - reconstruct(I,Poisson_Lambda,order)[i])
            ans[i][j] = Delta_f_i/Delta_I_j
    return ans

def plot(key,log,lower,upper):
    global reconstructed_spec
    global Observed_spec
    colour = ['red','orange','y','green','cyan','blue','purple']
    if key == 'signal to noise':
        print("no signal to noise")
    else:  
        plt.plot(ADU,prob*number_of_photons_per_frame*number_of_frames,label='Original')
        if key == 'reconstructed':
            plt.plot(ADU_grid,Observed_spec,'red',label="Observed Specturm ")
            for i in range(0,len(reconstructed_spec_all)):
                #print(Energy_grid)
                #print(Observed_spec)
                plt.plot(ADU_grid,reconstructed_spec_all[i],colour[i+1],label="Reconstructed with λ=%.4g%%"%(100*λ_set[i]))
                
            #plt.plot(Energy_grid,Transmission_rate[100:500],'black',label="Observed Specturm ")          
        
        if key == 'noise':
            plt.plot(ADU_grid,noise,'r',label="Noise")
            plt.plot(ADU_grid,reconstructed_spec,'g',label="Reconstructed Spectrum")
        if key == 'observed':
            plt.plot(ADU_grid,Observed_spec,'r',label="Clustered")
        if log == 1:
            plt.yscale('log')
        plt.ylim(1e0,1e5)
        plt.xlim(lower,upper)
        plt.xlabel('ADUs')
        plt.ylabel('Spectrum (Photon/ADU)')
        plt.legend()
        plt.show()





###############################################################################################################################################

###############################################################################################################################################

###############################################################################################################################################



ADU = np.linspace(100,500,401)
ADU_bin = np.linspace(0,501,502)
prob = 0.1*(ADU-100)**0.5 * np.exp(-(ADU-0)/50)
prob += 0.2 * np.exp(-((ADU-100))**2/25)
prob += 0.1 * np.exp(-((ADU-180))**2/25)
prob += 0.5 * np.exp(-((ADU-320))**2/25)
prob /= np.sum(prob)
number_of_frames = 10
Y = 704
X = 768
photon_fraction = 4
noise = 3
number_of_photons_per_frame = int(Y*X*photon_fraction*0.01)

run_num = [5]






    


path = 'SPC data analysis\\22Jan\\r_0.4\\noise_%s\\frac_%s\\'%(str(noise),str(photon_fraction))
for run in run_num:
    upper_grid = 1000
    lower_grid = 70
    order = 6



    Fractions = pandas.read_excel(path+'%d\\%d - frac.xlsx'%(run,run),sheet_name='Fractions')
    #print(Fractions)
    fill_frac = Fractions['Illuminated Pixels'][1]/100
    print("fill frac:",fill_frac)
    λ = - np.log(1-fill_frac)
    print("λ0:",λ)
    #λ = 1 - np.exp(-4 * λ)
    #print("λ:",λ)

    Observed = pandas.read_excel(path+"%d\\%d - spec.xlsx"%(run,run),sheet_name='Spectrum')
    ADU_grid = np.array((Observed['ADUs']))
    #print(Observed)
    reconstructed_spec = np.zeros((upper_grid))
    reconstructed_spec_all = []
    
    ADU_grid = np.append(np.linspace(0,(lower_grid-1),lower_grid),ADU_grid)[lower_grid:upper_grid]
    #print(Energy_grid.shape)
    

    λ_set = [0.01*photon_fraction,0.03*photon_fraction]

    for λ in λ_set:
        correction = 4
        Zeros = np.zeros((lower_grid+correction))
        Observed_spec = np.array((Observed['Valid Detections']),'float32')
        Observed_spec = np.append(Zeros,Observed_spec)[0:upper_grid]
        N = np.sum(Observed_spec)
        print('N:',N)
        Normalisation_factor = N /(λ*np.exp(-λ))
        Observed_spec /= Normalisation_factor
        print(λ,'Observed Integral:',np.sum(Observed_spec))
        reconstructed_spec = reconstruct(Observed_spec,λ,order)
        print(λ,'Reconstrcuted Integral:',np.sum(reconstructed_spec))
        
        error = np.sqrt(Observed_spec+1)
        error /= Normalisation_factor
        signal_to_noise = 0
        if signal_to_noise == 1:
            noise = error + 0
            derivative = num_d(Observed_spec,λ,order)
            for  i in range(785,786):
                temp = error * derivative[i]
                temp = temp ** 2
                temp = np.sum(temp)
                temp = temp ** 0.5
                noise[i] = temp
            #print(noise)
   
        recon = reconstructed_spec[lower_grid:upper_grid]
        recon *= N
        reconstructed_spec_all.append(recon)
        Observed_spec = Observed_spec[lower_grid:upper_grid] * Normalisation_factor


    


    key = [1,2]  # orders
    tb = 'transform back'
    rs = 'reconstructed'
    ns = 'noise'
    stn = 'signal to noise'
    ob = 'observed'
    log = 1
    plot(rs,log,70,500)
    
    Reconstructed_Spectra = {}
    a = [str(i) for i in λ_set]
    #print (a)
    for i in range(len(a)):
        Reconstructed_Spectra[a[i]] = reconstructed_spec_all[i]
    #print(Reconstructed_Spectra)
    df2 = pandas.DataFrame(Reconstructed_Spectra, columns = a) 
    df2.to_excel(path+"%d\\%d - deconvolution.xlsx"%(run,run),sheet_name='deconvolution')
    
    



