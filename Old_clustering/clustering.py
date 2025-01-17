import matplotlib.pyplot as plt
import matplotlib as mpl
import math
import numpy as np
from numpy import genfromtxt
import csv
import h5py
import pandas
import time

# TERM DEFINATION:
# Neighbouring refers to the 9 pixels around one pixel
# Adjacent refers to the 4 pixels around one pixel


def Adjacents(mat,i,j): # the four neighbours of mat[i][j]
    return [mat[i-1][j],mat[i+1][j],mat[i][j-1],mat[i][j+1]]

def Adjacents_Except_East(mat,i,j): # the neighbours of mat[i][j] except the east one
    return [mat[i-1][j],mat[i+1][j],mat[i][j-1]]

def Adjacents_Except_West(mat,i,j): # the neighbours of mat[i][j] except the west one
    return [mat[i-1][j],mat[i+1][j],mat[i][j+1]]

def Adjacents_Except_South(mat,i,j): # the neighbours of mat[i][j] except the south one
    return [mat[i-1][j],mat[i][j-1],mat[i][j+1]]

def Adjacents_Except_North(mat,i,j): #  the neighbours of mat[i][j] except the north one
    return [mat[i+1][j],mat[i][j-1],mat[i][j+1]]

def Adjacents_SE(mat,i,j):
    return [mat[i][j+1],mat[i+1][j]]

def Adjacents_NE(mat,i,j):
    return [mat[i][j+1],mat[i-1][j]]

def Adjacents_NW(mat,i,j):
    return [mat[i][j-1],mat[i-1][j]]

def Adjacents_SW(mat,i,j):
    return [mat[i][j-1],mat[i+1][j]]

def Neighbouring(mat,i,j): # neighbours of a single pixel
    ans  = set()
    for m in range(-1,2):
        for n in range(-1,2):
            ans.add((i+m,j+n))
    return ans

def Neighbouring_Cluster(mat,input): # neighbours of a cluster
    result = set()
    ans = []
    for pixel in input:
        i = pixel[0]
        j = pixel[1]
        neighbour = Neighbouring(mat,i,j)
        for pixel_2 in neighbour:
            p=pixel_2[0]
            q=pixel_2[1]
            result.add((p,q))
    for pixel in input:        
        i = pixel[0]
        j = pixel[1]
        result.remove((i,j))
    result = list(result)
    for pixel in result:
        i = pixel[0]
        j = pixel[1]
        ans.append(mat[i][j])
    return ans

def moment(mat,pixels):
    centre_i = 0
    centre_j = 0
    Weighted_Sum_i = 0
    Weighted_Sum_j = 0
    Sum = 0
    centre_i = 0.0
    centre_j = 0.0
    ans = 0.0
    for pixel in pixels:
        #print(pixel)
        i = pixel[0]
        j = pixel[1]
        #print(i)
        #print(j)
        Weighted_Sum_i += mat[i][j] * i
        Weighted_Sum_j += mat[i][j] * j
        Sum += mat[i][j]
    centre_i = Weighted_Sum_i/Sum
    centre_j = Weighted_Sum_j/Sum
    #print(centre_i,centre_j)
    for pixel in pixels:
        i = pixel[0]
        j = pixel[1]
        ans += mat[i][j] * ((i - centre_i) **2 + (j-centre_j) **2)
    ans /= Sum
    return ans


#clustering algorithm
def cluster(mat,threshold_1,threshold_2):
    global all_detections_photons
    global all_detections_pixels
    global all_single
    global valid_detections_photons
    global valid_single
    global valid_double

    global valid_triple
    global valid_quad

    for i in range(2,702):
        for j in range(2,766):
            # Single pixel events:
            if mat[i][j] > threshold_1:
                if np.max(Adjacents(mat,i,j)) < threshold_2:
                    all_detections_pixels[i][j] = mat[i][j]
                    all_detections_photons[i][j] = mat[i][j]
                    all_single[i][j] = mat[i][j]
                    cluster = [[i,j]]
                    if np.max(Neighbouring_Cluster(mat,cluster)) < threshold_2:
                        valid_single[i][j] = mat[i][j]
                        valid_detections_photons[i][j] = mat[i][j]


            # Double pixel events: 
            if 2 * mat[i][j] > threshold_1 and mat[i][j] > np.max(Adjacents(mat,i,j)): # pick a center peak
                if np.max(Adjacents_Except_East(mat,i,j)) < threshold_2 and mat[i][j+1] > threshold_2: # double eastwise
                    if mat[i][j] + mat[i][j+1] > threshold_1: # ensure that it is a photon hit
                        all_detections_pixels[i][j] = mat[i][j]
                        all_detections_pixels[i][j+1] = mat[i][j+1]
                        all_detections_photons[i][j] = mat[i][j] + mat[i][j+1]
                        cluster = [[i,j],[i,j+1]]
                        if np.max(Neighbouring_Cluster(mat,cluster)) < threshold_2: # validity check
                            valid_double[i][j] = mat[i][j]
                            valid_double[i][j+1] = mat[i][j+1]
                            if Double_statistics == True:
                                pixels = [[i,j],[i,j+1]]
                                double_moment[i][j] = moment(mat,pixels)
                                double_E[i][j] = mat[i][j]
                                double_E[i][j+1] = mat[i][j+1]
                                centre_i = mat[i][j] * i + mat[i+1][j] * (i+1)
                if np.max(Adjacents_Except_West(mat,i,j)) < threshold_2 and mat[i][j-1] > threshold_2: # double westwise
                    if mat[i][j] + mat[i][j-1] > threshold_1:
                        all_detections_pixels[i][j] = mat[i][j]
                        all_detections_pixels[i][j-1] = mat[i][j-1]
                        all_detections_photons[i][j] = mat[i][j] + mat[i][j-1]
                        cluster = [[i,j],[i,j-1]]
                        if np.max(Neighbouring_Cluster(mat,cluster)) < threshold_2:
                            valid_double[i][j] = mat[i][j]
                            valid_double[i][j-1] = mat[i][j-1]
                            if Double_statistics == True:
                                pixels = [[i,j],[i,j-1]]
                                double_moment[i][j] = moment(mat,pixels)
                                double_W[i][j] = mat[i][j]
                                double_W[i][j-1] = mat[i][j-1]
                            valid_detections_photons[i][j] = mat[i][j] + mat[i][j-1]
                if np.max(Adjacents_Except_North(mat,i,j)) < threshold_2 and mat[i-1][j] > threshold_2: # double northwise
                    if mat[i][j] + mat[i-1][j] > threshold_1:
                        all_detections_pixels[i][j] = mat[i][j]
                        all_detections_pixels[i-1][j] = mat[i-1][j]
                        all_detections_photons[i][j] = mat[i][j] + mat[i-1][j]
                        cluster = [[i,j],[i-1,j]]
                        if np.max(Neighbouring_Cluster(mat,cluster)) < threshold_2:   
                            valid_double[i][j] = mat[i][j]
                            valid_double[i-1][j] = mat[i-1][j]
                            if Double_statistics == True:
                                pixels = [[i,j],[i-1,j]]
                                double_moment[i][j] = moment(mat,pixels)
                                double_N[i][j] = mat[i][j]
                                double_N[i-1][j] = mat[i-1][j]
                            valid_detections_photons[i][j] = mat[i][j] + mat[i-1][j]
                if np.max(Adjacents_Except_South(mat,i,j)) < threshold_2 and mat[i+1][j] > threshold_2: # double southwise
                    if mat[i+1][j] + mat[i][j] > threshold_1:
                        all_detections_pixels[i][j] = mat[i][j]
                        all_detections_pixels[i+1][j] = mat[i+1][j]
                        all_detections_photons[i][j] = mat[i][j] + mat[i+1][j]
                        cluster = [[i+1,j],[i,j]]
                        if np.max(Neighbouring_Cluster(mat,cluster)) < threshold_2:
                            valid_double[i][j] = mat[i][j]
                            valid_double[i+1][j] = mat[i+1][j]
                            if Double_statistics == True:
                                pixels = [[i,j],[i+1,j]]
                                double_moment[i][j] = moment(mat,pixels)
                                double_S[i][j] = mat[i][j]
                                double_S[i+1][j] = mat[i+1][j]
                            valid_detections_photons[i][j] = mat[i+1][j] + mat[i][j]
            
            # Triple pixel events and quad pixel events
            if 3* mat[i][j] > threshold_1 and mat[i][j] > np.max(Adjacents(mat,i,j)):
                if np.max(Adjacents_NW(mat,i,j)) < threshold_2 and np.min(Adjacents_SE(mat,i,j)) > threshold_2 and mat[i+1][j+1] < threshold_2:
                    if mat[i][j] + mat[i][j+1] + mat[i+1][j] > threshold_1:
                        all_detections_pixels[i][j] = mat[i][j]
                        all_detections_pixels[i][j+1] = mat[i][j+1]
                        all_detections_pixels[i+1][j] = mat[i+1][j]
                        all_detections_photons[i][j] = mat[i][j] + mat[i+1][j] + mat[i][j+1]
                        cluster = [[i,j],[i,j+1],[i+1,j]]
                        if np.max(Neighbouring_Cluster(mat,cluster)) < threshold_2:
                            valid_triple[i][j] = mat[i][j]
                            valid_triple[i][j+1] = mat[i][j+1]
                            valid_triple[i+1][j] = mat[i+1][j]
                            if Triple_statistics == True:
                                pixels = [[i,j],[i,j+1],[i+1,j]]
                                triple_moment[i][j] = moment(mat,pixels)
                                triple_SE[i][j] = mat[i][j]
                                triple_SE[i][j+1] = mat[i][j+1]
                                triple_SE[i+1][j] = mat[i+1][j]
                            valid_detections_photons[i][j] = mat[i][j] + mat[i+1][j] + mat[i][j+1]
                if np.max(Adjacents_SE(mat,i,j)) < threshold_2 and np.min(Adjacents_NW(mat,i,j))>threshold_2 and mat[i-1][j-1] < threshold_2: # triple NW
                    if mat[i][j] + mat[i][j-1] + mat[i-1][j] > threshold_1:  
                        all_detections_pixels[i][j] = mat[i][j]
                        all_detections_pixels[i][j-1] = mat[i][j-1]
                        all_detections_pixels[i-1][j] = mat[i-1][j]
                        all_detections_photons[i][j] = mat[i][j] + mat[i-1][j] + mat[i][j-1]                    
                        cluster = [[i,j],[i,j-1],[i-1,j]]
                        if np.max(Neighbouring_Cluster(mat,cluster)) < threshold_2:
                            valid_triple[i][j] = mat[i][j]
                            valid_triple[i][j-1] = mat[i][j-1]
                            valid_triple[i-1][j] = mat[i-1][j]
                            if Triple_statistics == True:
                                pixels = [[i,j],[i,j-1],[i-1,j]]
                                triple_moment[i][j] = moment(mat,pixels)
                                triple_NW[i][j] = mat[i][j]
                                triple_NW[i][j-1] = mat[i][j-1]
                                triple_NW[i-1][j] = mat[i-1][j]
                            valid_detections_photons[i][j] = mat[i][j] + mat[i][j-1] + mat[i-1][j]
                if np.max(Adjacents_NE(mat,i,j)) < threshold_2 and np.min(Adjacents_SW(mat,i,j))>threshold_2 and mat[i+1][j-1] < threshold_2: # triple SW
                    if mat[i][j] + mat[i][j-1] + mat[i+1][j] > threshold_1:
                        all_detections_pixels[i][j] = mat[i][j]
                        all_detections_pixels[i][j-1] = mat[i][j-1]
                        all_detections_pixels[i+1][j] = mat[i+1][j]
                        all_detections_photons[i][j] = mat[i][j] + mat[i][j-1] + mat[i+1][j]
                        cluster = [[i,j],[i,j-1],[i+1,j]]
                        if np.max(Neighbouring_Cluster(mat,cluster)) < threshold_2:
                            valid_triple[i][j] = mat[i][j]
                            valid_triple[i][j-1] = mat[i][j-1]
                            valid_triple[i+1][j] = mat[i+1][j]
                            if Triple_statistics == True:
                                pixels = [[i,j],[i,j-1],[i+1,j]]
                                triple_moment[i][j] = moment(mat,pixels)
                                triple_SW[i][j] = mat[i][j]
                                triple_SW[i][j-1] = mat[i][j-1]
                                triple_SW[i+1][j] = mat[i+1][j]
                            valid_detections_photons[i][j] = mat[i][j] + mat[i][j-1] + mat[i+1][j]
                if np.max(Adjacents_SW(mat,i,j)) < threshold_2 and np.min(Adjacents_NE(mat,i,j))>threshold_2 and mat[i-1][j+1] < threshold_2: # triple NE
                    if mat[i][j] + mat[i][j+1] + mat[i-1][j] > threshold_1:
                        all_detections_pixels[i][j] = mat[i][j]
                        all_detections_pixels[i][j+1] = mat[i][j+1]
                        all_detections_pixels[i-1][j] = mat[i-1][j]
                        all_detections_photons[i][j] = mat[i][j] + mat[i][j+1] + mat[i-1][j]
                        cluster = [[i,j],[i,j+1],[i-1,j]]
                        if np.max(Neighbouring_Cluster(mat,cluster)) < threshold_2:
                            valid_triple[i][j] = mat[i][j]
                            valid_triple[i][j+1] = mat[i][j+1]
                            valid_triple[i-1][j] = mat[i-1][j]
                            if Triple_statistics == True:
                                pixels = [[i,j],[i,j+1],[i-1,j]]
                                triple_moment[i][j] = moment(mat,pixels)
                                triple_NE[i][j] = mat[i][j]
                                triple_NE[i][j+1] = mat[i][j+1]
                                triple_NE[i-1][j] = mat[i-1][j]
                            valid_detections_photons[i][j] = mat[i][j] + mat[i][j+1] + mat[i-1][j]
            
            # quad pixel events
            if 4* mat[i][j] > threshold_1 and mat[i][j] > np.max(Adjacents(mat,i,j)):  
                if np.max(Adjacents_NW(mat,i,j)) < threshold_2 and np.min(Adjacents_SE(mat,i,j)) > threshold_2 and mat[i+1][j+1] > threshold_2: # quad SE
                    if mat[i][j] + mat[i][j+1] + mat[i+1][j] +mat[i+1][j+1] > threshold_1: 
                        all_detections_pixels[i][j] = mat[i][j]
                        all_detections_pixels[i][j+1] = mat[i][j+1]
                        all_detections_pixels[i+1][j] = mat[i+1][j]
                        all_detections_pixels[i+1][j+1] = mat[i+1][j+1]
                        all_detections_photons[i][j] = mat[i][j] + mat[i+1][j] + mat[i][j+1] + mat[i+1][j+1]
                        cluster = [[i,j],[i,j+1],[i+1,j],[i+1,j+1]]
                        if np.max(Neighbouring_Cluster(mat,cluster)) < threshold_2:
                            valid_quad[i][j] = mat[i][j]
                            valid_quad[i][j+1] = mat[i][j+1]
                            valid_quad[i+1][j] = mat[i+1][j]
                            valid_quad[i+1][j+1] = mat[i+1][j+1]
                            if Quad_statistics == True:
                                pixels = [[i,j],[i,j+1],[i+1,j],[i+1,j+1]]
                                quad_moment[i][j] = moment(mat,pixels)
                                quad_SE[i][j] = mat[i][j]
                                quad_SE[i][j+1] = mat[i][j+1]
                                quad_SE[i+1][j] = mat[i+1][j]
                                quad_SE[i+1][j+1] = mat[i+1][j+1]
                            valid_detections_photons[i][j] = mat[i][j] + mat[i+1][j] + mat[i][j+1] + mat[i+1][j+1] 
                if np.max(Adjacents_SE(mat,i,j)) < threshold_2 and np.min(Adjacents_NW(mat,i,j)) > threshold_2 and mat[i-1][j-1] > threshold_2: # quad NW
                    if mat[i][j] + mat[i][j-1] + mat[i-1][j] +mat[i-1][j-1] > threshold_1: 
                        all_detections_pixels[i][j] = mat[i][j]
                        all_detections_pixels[i][j-1] = mat[i][j-1]
                        all_detections_pixels[i-1][j] = mat[i-1][j]
                        all_detections_pixels[i-1][j-1] = mat[i-1][j-1]
                        all_detections_photons[i][j] = mat[i][j] + mat[i-1][j] + mat[i][j-1] + mat[i-1][j-1]
                        cluster = [[i,j],[i,j-1],[i-1,j],[i-1,j-1]]
                        if np.max(Neighbouring_Cluster(mat,cluster)) < threshold_2:
                            valid_quad[i][j] = mat[i][j]
                            valid_quad[i][j-1] = mat[i][j-1]
                            valid_quad[i-1][j] = mat[i-1][j]
                            valid_quad[i-1][j-1] = mat[i-1][j-1]
                            if Quad_statistics == True:
                                pixels = [[i,j],[i,j-1],[i-1,j],[i-1,j-1]]
                                quad_moment[i][j] = moment(mat,pixels)
                                quad_NW[i][j] = mat[i][j]
                                quad_NW[i][j-1] = mat[i][j-1]
                                quad_NW[i-1][j] = mat[i-1][j]
                                quad_NW[i-1][j-1] = mat[i-1][j-1]
                            valid_detections_photons[i][j] = mat[i][j] + mat[i-1][j] + mat[i][j-1] + mat[i-1][j-1]  
                if np.max(Adjacents_NE(mat,i,j)) < threshold_2 and np.min(Adjacents_SW(mat,i,j)) > threshold_2 and mat[i+1][j-1] > threshold_2: # quad SW
                    if mat[i][j] + mat[i][j-1] + mat[i+1][j] +mat[i+1][j-1] > threshold_1: 
                        all_detections_pixels[i][j] = mat[i][j]
                        all_detections_pixels[i][j-1] = mat[i][j-1]
                        all_detections_pixels[i+1][j] = mat[i+1][j]
                        all_detections_pixels[i+1][j-1] = mat[i+1][j-1]
                        all_detections_photons[i][j] = mat[i][j] + mat[i+1][j] + mat[i][j-1] + mat[i+1][j-1]
                        cluster = [[i,j],[i,j-1],[i+1,j],[i+1,j-1]]
                        if np.max(Neighbouring_Cluster(mat,cluster)) < threshold_2:
                            valid_quad[i][j] = mat[i][j]
                            valid_quad[i][j-1] = mat[i][j-1]
                            valid_quad[i+1][j] = mat[i+1][j]
                            valid_quad[i+1][j-1] = mat[i+1][j-1]
                            if Quad_statistics == True:
                                pixels = [[i,j],[i,j-1],[i+1,j],[i+1,j-1]]
                                quad_moment[i][j] = moment(mat,pixels)
                                quad_NE[i][j] = mat[i][j]
                                quad_NE[i][j-1] = mat[i][j-1]
                                quad_NE[i+1][j] = mat[i+1][j]
                                quad_NE[i+1][j-1] = mat[i+1][j-1]
                            valid_detections_photons[i][j] = mat[i][j] + mat[i+1][j] + mat[i][j-1] + mat[i+1][j-1]                        
                if np.max(Adjacents_SW(mat,i,j)) < threshold_2 and np.min(Adjacents_NE(mat,i,j)) > threshold_2 and mat[i+1][j+1] > threshold_2: # quad NE
                    if mat[i][j] + mat[i][j+1] + mat[i-1][j] +mat[i-1][j+1] > threshold_1: 
                        all_detections_pixels[i][j] = mat[i][j]
                        all_detections_pixels[i][j+1] = mat[i][j+1]
                        all_detections_pixels[i-1][j] = mat[i-1][j]
                        all_detections_pixels[i-1][j+1] = mat[i-1][j+1]
                        all_detections_photons[i][j] = mat[i][j] + mat[i-1][j] + mat[i][j+1] + mat[i-1][j+1]
                        cluster = [[i,j],[i,j+1],[i-1,j],[i-1,j+1]]
                        if np.max(Neighbouring_Cluster(mat,cluster)) < threshold_2:
                            print("a")
                            valid_quad[i][j] = mat[i][j]
                            valid_quad[i][j+1] = mat[i][j+1]
                            valid_quad[i-1][j] = mat[i-1][j]
                            valid_quad[i-1][j+1] = mat[i-1][j+1]
                            if Quad_statistics == True:
                                pixels = [[i,j],[i,j+1],[i-1,j],[i-1,j+1]]
                                quad_moment[i][j] = moment(mat,pixels)
                                quad_SW[i][j] = mat[i][j]
                                quad_SW[i][j+1] = mat[i][j+1]
                                quad_SW[i+1][j] = mat[i+1][j]
                                quad_SW[i+1][j+1] = mat[i+1][j+1]
                            valid_detections_photons[i][j] = mat[i][j] + mat[i-1][j] + mat[i][j+1] + mat[i-1][j+1] 




                            

def overlay_check(matA,matB): # to check if any of the detections have overlay
    overlay = np.zeros((704,768)) 
    for i in range(1,703):
        for j in range(1,767):
            if matA[i][j] > 0 and matB[i][j] > 0 and matA[i][j] == matB[i][j]:
                overlay[i][j] = 1
    return overlay


def make_hist_and_plot(mat): # to make histigram of an image and show that image
    A = np.asarray(mat.reshape(-1))
    plt.figure()
    plt.hist(A,bins=500,range=(-100,400),log=True)
    plt.xlabel('# of ADUs')
    plt.ylabel('Frequency')
    plt.show()
    plt.imshow(mat)
    plt.colorbar()
    plt.axis()
    plt.show()



def moment_hist(moment):
    hist = plt.hist(moment,bins=100,range=(0,1),log=False)
    ans = (hist[0],hist[1])
    return ans

def mkdir(path):
    import os
    path=path.strip()
    path=path.rstrip("\\")
    isExists=os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return True
    
    else:

        return False





####################################################################################################
##                                                                                                ##
##                               THIS IS THE MAIN PART OF THE CODES                               ##
##                                                                                                ##
####################################################################################################






run_range = range(0,1) # give a set of run numbers
noise = 3
frac = 1
print("frac:",frac)
path = "New_data\\r_0.3\\Noise_%s\\frac_%s\\"%(str(noise),str(frac))
#-------------------------- STARTS HERE --------------------------
for run in run_range: # the set of images used to construct a spectrum
    t1 = time.time() # only for timming


    #====================================================== import the raw data ======================================================


    f = h5py.File(path+"MC_%d.hdf5"%(run),'r')    # where the raw data is, change to anyting like
    #f = h5py.File('F:\\nc\\run0%d.nc'%(run),'r')
    #print(list(f.keys()))
    dataset = f['data']      #import raw data
    #print(dataset.shape)
    total_image_number = dataset.shape[0]  # how many images in one run, should be 342 for majority
    #print(total_image_number)

    
    #====================================================== import the raw data ======================================================




    #-------------------------- raw data statistics --------------------------
    
    A = np.asarray(dataset).reshape(-1) # to make a histogram of the raw data
    plt.figure()
    hist = plt.hist(A,bins=500,range=(-100,400)) # to make a histogram of the raw data. Limit set to -100 ADU
    #plt.xlabel('# of ADUs') # to generate a histogram, not in use any longer
    #plt.ylabel('Frequency')
    #plt.yscale("log")
    #plt.show()
    #make_hist_and_plot(dataset[100])


    Threshold_2 = 0  # 3 sigma of the background noise
    peak = np.argmax(hist[0][200:300]) + 100 # peak at XFEL Energy, should be 9keV. need +100 as the histogram starts at -100
    print('peak is %f' %(peak))
    for i in range(100,150):
        if hist[0][i] < 2.71828**(-4.5) * np.max(hist[0]): # find the three sigma from a Gaussian fit
            Threshold_2 = i - 100 # the histogram starts at -100 ADU
            break
    print('3 sigma: %d' %(Threshold_2))

    Threshold_1 = 70 # threshold for valid photon detection, temperary set to be 70, therefore 70ADU * 60 eV/ADU = 4200 eV
    
    #-------------------------- raw data statistics --------------------------

    


    #-------------------------- clustering algorithm --------------------------


    all_detections_photons_3D = np.zeros((total_image_number,704,768)) # All detections recognized as photons attributed to the central one pixel
    all_detections_pixels_3D = np.zeros((total_image_number,704,768)) # All detections recognized as photons with chargespread
    all_single_3D = np.zeros((total_image_number,704,768)) # All single pixel events
    
    valid_detections_photons_3D = np.zeros((total_image_number,704,768)) # Aingle photon clusters attributed the central one pixel
    
    valid_single_3D = np.zeros((total_image_number,704,768)) # Valid single pixel events


    Double_statistics = True
    Triple_statistics = True
    Quad_statistics = True

    valid_double_3D = np.zeros((total_image_number,704,768)) # Double pixel events
    if Double_statistics == True:
        double_E_3D = np.zeros((total_image_number,704,768))
        double_W_3D = np.zeros((total_image_number,704,768))
        double_S_3D = np.zeros((total_image_number,704,768))
        double_N_3D = np.zeros((total_image_number,704,768))
        double_moment_3D = np.zeros((total_image_number,704,768))

    valid_triple_3D = np.zeros((total_image_number,704,768)) # Triple pixel events
    if Triple_statistics == True:
        triple_SW_3D = np.zeros((total_image_number,704,768))
        triple_SE_3D = np.zeros((total_image_number,704,768))
        triple_NW_3D = np.zeros((total_image_number,704,768))
        triple_NE_3D = np.zeros((total_image_number,704,768))
        triple_moment_3D = np.zeros((total_image_number,704,768))

    valid_quad_3D = np.zeros((total_image_number,704,768)) # Quad pixel events
    if Quad_statistics == True:
        quad_SW_3D = np.zeros((total_image_number,704,768))
        quad_SE_3D = np.zeros((total_image_number,704,768))
        quad_NW_3D = np.zeros((total_image_number,704,768))
        quad_NE_3D = np.zeros((total_image_number,704,768))
        quad_moment_3D = np.zeros((total_image_number,704,768))




    for i in range(0,total_image_number):  #SPC clustering algorithm image by image
        t3 = time.time() # just for timming
        matrix = dataset[i] # import from dataset, the 704*768 pixel ADU values
         

        all_detections_photons = np.zeros((704,768))  # All detections recognized as photon detections
        all_detections_pixels = np.zeros((704,768))
        all_single =  np.zeros((704,768)) # all single pixel events on that image
        
        valid_detections_photons = np.zeros((704,768)) # single photon clusters attributed the central one pixel on that image
        
        valid_single = np.zeros((704,768)) # valid single pixel events on that image

        valid_double = np.zeros((704,768)) # double pixel events on that image
        if Double_statistics == True:
            double_E = np.zeros((704,768))
            double_W = np.zeros((704,768))
            double_S = np.zeros((704,768))
            double_N = np.zeros((704,768))
            double_moment = np.zeros((704,768))

        valid_triple = np.zeros((704,768)) # triple pixel events on that image
        if Triple_statistics == True:
            triple_SW = np.zeros((704,768)) 
            triple_SE = np.zeros((704,768)) 
            triple_NW = np.zeros((704,768)) 
            triple_NE = np.zeros((704,768))
            triple_moment = np.zeros((704,768))

        valid_quad = np.zeros((704,768)) # quad pixel events on that image
        if Quad_statistics == True:
            quad_SW = np.zeros((704,768)) 
            quad_SE = np.zeros((704,768)) 
            quad_NW = np.zeros((704,768)) 
            quad_NE = np.zeros((704,768))
            quad_moment = np.zeros((704,768))       

        cluster(matrix,Threshold_1,Threshold_2) # clustering algorithm
        
        all_detections_photons_3D[i] = all_detections_photons
        all_detections_pixels_3D[i] = all_detections_pixels
        all_single_3D[i] = all_single

        valid_detections_photons_3D[i] = valid_detections_photons

        valid_single_3D[i] = valid_single

        valid_double_3D[i] = valid_double
        if Double_statistics == True:
            double_E_3D[i] = double_E
            double_W_3D[i] = double_W
            double_S_3D[i] = double_S
            double_N_3D[i] = double_N
            double_moment_3D[i] = double_moment

        valid_triple_3D[i] = valid_triple
        if Triple_statistics == True:
            triple_SW_3D[i] = triple_SW
            triple_SE_3D[i] = triple_SE
            triple_NW_3D[i] = triple_NW
            triple_NE_3D[i] = triple_NE
            triple_moment_3D[i] = triple_moment

        valid_quad_3D[i] = valid_quad
        if Quad_statistics == True:
            quad_SW_3D[i] = quad_SW
            quad_SE_3D[i] = quad_SE
            quad_NW_3D[i] = quad_NW
            quad_NE_3D[i] = quad_NE
            quad_moment_3D[i] = triple_moment
        t4 = time.time()
        print('\r'+'▋'*((i*10)//total_image_number)+"  "+str((i*100)//total_image_number)+'%', end='')
        #print('iteration: %d, time: %f'%(i,t4-t3))

    valid_detections_pixels_3D = valid_single_3D + valid_double_3D + valid_triple_3D + valid_quad_3D

    #-------------------------- clustering algorithm --------------------------
    path_new = "SPC data analysis\\New\\r_0.3\\noise_%s\\frac_%s\\%s\\"%(str(noise),str(frac),str(run))
    mkdir(path_new)
    file_valid = h5py.File(path_new+"MC_%d_valid.hdf5"%(run), "w")
    file_valid.create_dataset("data", data=valid_detections_photons_3D)
    




    t2 = time.time()
    print("")
    print('totaltime: %f'%(t2-t1))

    




