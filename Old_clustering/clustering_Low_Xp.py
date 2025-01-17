


##################################################################################################
#                                                                                                #
#                          SINGLE PHOTON COUNTING CLUSTERING ALGORITHM                           #
#                                                                                                #
##################################################################################################

##          YUANFENG SHI 石元峰(c) ALL RIGHTS RESERVED  




import matplotlib.pyplot as plt
import matplotlib as mpl
import math
import numpy as np
from numpy import genfromtxt
import csv
import h5py
import pandas
import time
import os

# TERM DEFINATION:

# Neighbouring refers to the 8 pixels around one pixel
# Adjacent refers to the 4 pixels around one pixel
# East: j+1, West: j-1
# North: i-1, South: i+1




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

def moment(mat,pixels): # moment cefined similar to moment of inertia, i.e., Σ_i((m_i)(r_i^2)), where m_i's are ADU's
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
def cluster(mat,threshold1,threshold2):

    for i in range(2,702):
        for j in range(2,766):
            # Single pixel events:
            if mat[i][j] > threshold1:
                if np.max(Adjacents(mat,i,j)) < threshold2:
                    all_detections_pixels.append(mat[i][j])
                    all_detections_photons.append(mat[i][j])
                    all_single.append(mat[i][j])
                    cluster = [[i,j]]
                    if np.max(Neighbouring_Cluster(mat,cluster)) < threshold2:
                        valid_single.append(mat[i][j])
                        valid_detections_photons.append(mat[i][j])


            # Double pixel events: 
            if 2 * mat[i][j] > threshold1 and mat[i][j] > np.max(Adjacents(mat,i,j)): # pick a center peak
                if np.max(Adjacents_Except_East(mat,i,j)) < threshold2 and mat[i][j+1] > threshold2: # double eastwise
                    if mat[i][j] + mat[i][j+1] > threshold1: # ensure that it is a photon hit
                        all_detections_pixels.append(mat[i][j])
                        all_detections_pixels.append(mat[i][j+1])
                        all_detections_photons.append(mat[i][j] + mat[i][j+1])
                        cluster = [[i,j],[i,j+1]]
                        if np.max(Neighbouring_Cluster(mat,cluster)) < threshold2:
                            valid_detections_photons.append(mat[i][j] + mat[i][j+1])
                            valid_double.append(mat[i][j]+mat[i][j+1])
                            double_E.append(mat[i][j])
                            double_E.append(mat[i][j+1])
                            pixels = [[i,j],[i,j+1]]
                            double_moment.append(moment(mat,pixels))
                            if moment(mat,pixels) > 0.24:
                                double_moment_max.append(mat[i][j])

                if np.max(Adjacents_Except_West(mat,i,j)) < threshold2 and mat[i][j-1] > threshold2: # double westwise
                    if mat[i][j] + mat[i][j-1] > threshold1:
                        all_detections_pixels.append(mat[i][j])
                        all_detections_pixels.append(mat[i][j-1])
                        all_detections_photons.append(mat[i][j] + mat[i][j-1])
                        cluster = [[i,j],[i,j-1]]
                        if np.max(Neighbouring_Cluster(mat,cluster)) < threshold2:
                            valid_detections_photons.append(mat[i][j] + mat[i][j-1])
                            valid_double.append(mat[i][j]+mat[i][j-1])
                            double_W.append(mat[i][j])
                            double_W.append(mat[i][j-1])
                            pixels = [[i,j],[i,j-1]]
                            double_moment.append(moment(mat,pixels))
                            if moment(mat,pixels) > 0.24:
                                double_moment_max.append(mat[i][j])

                if np.max(Adjacents_Except_North(mat,i,j)) < threshold2 and mat[i-1][j] > threshold2: # double northwise
                    if mat[i][j] + mat[i-1][j] > threshold1:
                        all_detections_pixels.append(mat[i][j])
                        all_detections_pixels.append(mat[i-1][j])
                        all_detections_photons.append(mat[i][j] + mat[i-1][j])
                        cluster = [[i,j],[i-1,j]]
                        if np.max(Neighbouring_Cluster(mat,cluster)) < threshold2:   
                            valid_detections_photons.append(mat[i][j] + mat[i-1][j])
                            valid_double.append(mat[i][j]+mat[i-1][j])
                            double_N.append(mat[i][j])
                            double_N.append(mat[i-1][j])
                            pixels = [[i,j],[i-1,j]]
                            double_moment.append(moment(mat,pixels))
                            if moment(mat,pixels) > 0.24:
                                double_moment_max.append(mat[i][j])

                if np.max(Adjacents_Except_South(mat,i,j)) < threshold2 and mat[i+1][j] > threshold2: # double southwise
                    if mat[i+1][j] + mat[i][j] > threshold1:
                        all_detections_pixels.append(mat[i][j])
                        all_detections_pixels.append(mat[i+1][j])
                        all_detections_photons.append(mat[i][j] + mat[i+1][j])
                        cluster = [[i+1,j],[i,j]]
                        if np.max(Neighbouring_Cluster(mat,cluster)) < threshold2:
                            valid_detections_photons.append(mat[i+1][j] + mat[i][j])
                            valid_double.append(mat[i][j]+mat[i+1][j])
                            double_S.append(mat[i][j])
                            double_S.append(mat[i+1][j])
                            pixels = [[i,j],[i+1,j]]
                            double_moment.append(moment(mat,pixels))
                            if moment(mat,pixels) > 0.24:
                                double_moment_max.append(mat[i][j])

                            
            
            # Triple pixel events
            if 3* mat[i][j] > threshold1 and mat[i][j] > np.max(Adjacents(mat,i,j)):
                if np.max(Adjacents_NW(mat,i,j)) < threshold2 and np.min(Adjacents_SE(mat,i,j)) > threshold2 and mat[i+1][j+1] < threshold2:
                    if mat[i][j] + mat[i][j+1] + mat[i+1][j] > threshold1:
                        all_detections_pixels.append(mat[i][j])
                        all_detections_pixels.append(mat[i][j+1])
                        all_detections_pixels.append(mat[i+1][j])
                        all_detections_photons.append(mat[i][j] + mat[i+1][j] + mat[i][j+1])
                        cluster = [[i,j],[i,j+1],[i+1,j]]
                        if np.max(Neighbouring_Cluster(mat,cluster)) < threshold2:
                            valid_detections_photons.append(mat[i][j] + mat[i+1][j] + mat[i][j+1])
                            valid_triple.append(mat[i][j] + mat[i+1][j] + mat[i][j+1])
                            triple_SE.append(mat[i][j])
                            triple_SE.append(mat[i][j+1])
                            triple_SE.append(mat[i+1][j])
                            pixels = [[i,j],[i,j+1],[i+1,j]]
                            triple_moment.append(moment(mat,pixels))
                
                if np.max(Adjacents_SE(mat,i,j)) < threshold2 and np.min(Adjacents_NW(mat,i,j))>threshold2 and mat[i-1][j-1] < threshold2: # triple NW
                    if mat[i][j] + mat[i][j-1] + mat[i-1][j] > threshold1:  
                        all_detections_pixels.append(mat[i][j])
                        all_detections_pixels.append(mat[i][j-1])
                        all_detections_pixels.append(mat[i-1][j])
                        all_detections_photons.append(mat[i][j] + mat[i-1][j] + mat[i][j-1])                    
                        cluster = [[i,j],[i,j-1],[i-1,j]]
                        if np.max(Neighbouring_Cluster(mat,cluster)) < threshold2:
                            valid_detections_photons.append(mat[i][j] + mat[i][j-1] + mat[i-1][j])
                            valid_triple.append(mat[i][j] + mat[i][j-1] + mat[i-1][j])
                            triple_NW.append(mat[i][j])
                            triple_NW.append(mat[i][j-1])
                            triple_NW.append(mat[i-1][j])
                            pixels = [[i,j],[i,j-1],[i-1,j]]
                            triple_moment.append(moment(mat,pixels))

                if np.max(Adjacents_NE(mat,i,j)) < threshold2 and np.min(Adjacents_SW(mat,i,j))>threshold2 and mat[i+1][j-1] < threshold2: # triple SW
                    if mat[i][j] + mat[i][j-1] + mat[i+1][j] > threshold1:
                        all_detections_pixels.append(mat[i][j])
                        all_detections_pixels.append(mat[i][j-1])
                        all_detections_pixels.append(mat[i+1][j])
                        all_detections_photons.append(mat[i][j] + mat[i][j-1] + mat[i+1][j])
                        cluster = [[i,j],[i,j-1],[i+1,j]]
                        if np.max(Neighbouring_Cluster(mat,cluster)) < threshold2:
                            valid_detections_photons.append(mat[i][j] + mat[i][j-1] + mat[i+1][j])
                            valid_triple.append(mat[i][j] + mat[i][j-1] + mat[i+1][j])
                            triple_SW.append(mat[i][j])
                            triple_SW.append(mat[i][j-1])
                            triple_SW.append(mat[i+1][j])
                            pixels = [[i,j],[i,j-1],[i+1,j]]
                            triple_moment.append(moment(mat,pixels))
                            
                if np.max(Adjacents_SW(mat,i,j)) < threshold2 and np.min(Adjacents_NE(mat,i,j))>threshold2 and mat[i-1][j+1] < threshold2: # triple NE
                    if mat[i][j] + mat[i][j+1] + mat[i-1][j] > threshold1:
                        all_detections_pixels.append(mat[i][j])
                        all_detections_pixels.append(mat[i][j+1])
                        all_detections_pixels.append(mat[i-1][j])
                        all_detections_photons.append(mat[i][j] + mat[i][j+1] + mat[i-1][j])
                        cluster = [[i,j],[i,j+1],[i-1,j]]
                        if np.max(Neighbouring_Cluster(mat,cluster)) < threshold2:
                            valid_detections_photons.append(mat[i][j] + mat[i][j+1] + mat[i-1][j])
                            valid_triple.append(mat[i][j] + mat[i][j+1] + mat[i-1][j])
                            triple_NE.append(mat[i][j])
                            triple_NE.append(mat[i][j+1])
                            triple_NE.append(mat[i-1][j])
                            pixels = [[i,j],[i,j+1],[i-1,j]]
                            triple_moment.append(moment(mat,pixels))

                if np.max([mat[i+1][j],mat[i-1][j]]) < threshold2 and np.max([mat[i][j-1],mat[i][j+1]]) > threshold2: # testing triple row
                    if mat[i][j-1] + mat[i][j] + mat[i][j+1] > threshold1:
                        all_detections_pixels.append(mat[i][j])
                        all_detections_pixels.append(mat[i][j-1])
                        all_detections_pixels.append(mat[i][j+1])
                        all_detections_photons.append(mat[i][j]+mat[i][j-1]+mat[i][j+1])
                        cluster = [[i,j],[i,j-1],[i,j+1]]
                        if np.max(Neighbouring_Cluster(mat,cluster)) < threshold2:
                            #valid_detections_photons.append(mat[i][j]+mat[i][j+1]+mat[i][j-1])
                            triple_row.append(mat[i][j])
                            triple_row.append(mat[i][j-1])
                            triple_row.append(mat[i][j+1])
                            pixels = [[i,j],[i,j-1],[i,j+1]]
                            triple_moment.append(moment(mat,pixels))
                
                if np.max([mat[i][j-1],mat[i][j+1]]) < threshold2 and np.max([mat[i-1][j],mat[i+1][j]]) > threshold2: # testing triple bar
                    if mat[i-1][j] + mat[i][j] + mat[i+1][j] > threshold1:
                        all_detections_pixels.append(mat[i][j])
                        all_detections_pixels.append(mat[i-1][j])
                        all_detections_pixels.append(mat[i+1][j])
                        all_detections_photons.append(mat[i][j]+mat[i-1][j]+mat[i+1][j])
                        cluster = [[i,j],[i-1,j],[i+1,j]]
                        if np.max(Neighbouring_Cluster(mat,cluster)) < threshold2:
                            #valid_detections_photons.append(mat[i][j]+mat[i-1][j]+mat[i+1][j])
                            triple_bar.append(mat[i][j])
                            triple_bar.append(mat[i-1][j])
                            triple_bar.append(mat[i+1][j])    
                            pixels = [[i,j],[i-1,j],[i+1,j]]
                            triple_moment.append(moment(mat,pixels))

            
            # quad pixel events
            if 4* mat[i][j] > threshold1 and mat[i][j] > np.max(Adjacents(mat,i,j)):  
                if np.max(Adjacents_NW(mat,i,j)) < threshold2 and np.min(Adjacents_SE(mat,i,j)) > threshold2 and mat[i+1][j+1] > threshold2: # quad SE
                    if mat[i][j] + mat[i][j+1] + mat[i+1][j] +mat[i+1][j+1] > threshold1: 
                        all_detections_pixels.append(mat[i][j])
                        all_detections_pixels.append(mat[i][j+1])
                        all_detections_pixels.append(mat[i+1][j])
                        all_detections_pixels.append(mat[i+1][j+1])
                        all_detections_photons.append(mat[i][j] + mat[i+1][j] + mat[i][j+1] + mat[i+1][j+1])
                        cluster = [[i,j],[i,j+1],[i+1,j],[i+1,j+1]]
                        if np.max(Neighbouring_Cluster(mat,cluster)) < threshold2:
                            valid_detections_photons.append(mat[i][j] + mat[i+1][j] + mat[i][j+1] + mat[i+1][j+1])
                            valid_quad.append(mat[i][j] + mat[i+1][j] + mat[i][j+1] + mat[i+1][j+1])
                            quad_SE.append(mat[i][j])
                            quad_SE.append(mat[i][j+1])
                            quad_SE.append(mat[i+1][j])
                            quad_SE.append(mat[i+1][j+1])
                            pixels = [[i,j],[i,j+1],[i+1,j],[i+1,j+1]]
                            quad_moment.append(moment(mat,pixels))

                if np.max(Adjacents_SE(mat,i,j)) < threshold2 and np.min(Adjacents_NW(mat,i,j)) > threshold2 and mat[i-1][j-1] > threshold2: # quad NW
                    if mat[i][j] + mat[i][j-1] + mat[i-1][j] +mat[i-1][j-1] > threshold1: 
                        all_detections_pixels.append(mat[i][j])
                        all_detections_pixels.append(mat[i][j-1])
                        all_detections_pixels.append(mat[i-1][j])
                        all_detections_pixels.append(mat[i-1][j-1])
                        all_detections_photons.append(mat[i][j] + mat[i-1][j] + mat[i][j-1] + mat[i-1][j-1])
                        cluster = [[i,j],[i,j-1],[i-1,j],[i-1,j-1]]
                        if np.max(Neighbouring_Cluster(mat,cluster)) < threshold2:
                            valid_detections_photons.append(mat[i][j] + mat[i-1][j] + mat[i][j-1] + mat[i-1][j-1])
                            valid_quad.append(mat[i][j] + mat[i-1][j] + mat[i][j-1] + mat[i-1][j-1])
                            quad_NW.append(mat[i][j])
                            quad_NW.append(mat[i][j-1])
                            quad_NW.append(mat[i-1][j])
                            quad_NW.append(mat[i-1][j-1])
                            pixels = [[i,j],[i,j-1],[i-1,j],[i-1,j-1]]
                            quad_moment.append(moment(mat,pixels))

                if np.max(Adjacents_NE(mat,i,j)) < threshold2 and np.min(Adjacents_SW(mat,i,j)) > threshold2 and mat[i+1][j-1] > threshold2: # quad SW
                    if mat[i][j] + mat[i][j-1] + mat[i+1][j] +mat[i+1][j-1] > threshold1: 
                        all_detections_pixels.append(mat[i][j])
                        all_detections_pixels.append(mat[i][j-1])
                        all_detections_pixels.append(mat[i+1][j])
                        all_detections_pixels.append(mat[i+1][j-1])
                        all_detections_photons.append(mat[i][j] + mat[i+1][j] + mat[i][j-1] + mat[i+1][j-1])
                        cluster = [[i,j],[i,j-1],[i+1,j],[i+1,j-1]]
                        if np.max(Neighbouring_Cluster(mat,cluster)) < threshold2:
                            valid_detections_photons.append(mat[i][j] + mat[i+1][j] + mat[i][j-1] + mat[i+1][j-1])
                            valid_quad.append(mat[i][j] + mat[i+1][j] + mat[i][j-1] + mat[i+1][j-1])
                            quad_SW.append(mat[i][j])
                            quad_SW.append(mat[i][j-1])
                            quad_SW.append(mat[i+1][j])
                            quad_SW.append(mat[i+1][j-1])
                            pixels = [[i,j],[i,j-1],[i+1,j],[i+1,j-1]]
                            quad_moment.append(moment(mat,pixels))
                                                
                if np.max(Adjacents_SW(mat,i,j)) < threshold2 and np.min(Adjacents_NE(mat,i,j)) > threshold2 and mat[i-1][j+1] > threshold2: # quad NE\
                    if mat[i][j] + mat[i][j+1] + mat[i-1][j] +mat[i-1][j+1] > threshold1: 
                        all_detections_pixels.append(mat[i][j])
                        all_detections_pixels.append(mat[i][j+1])
                        all_detections_pixels.append(mat[i-1][j])
                        all_detections_pixels.append(mat[i-1][j+1])
                        all_detections_photons.append(mat[i][j] + mat[i-1][j] + mat[i][j+1] + mat[i-1][j+1])
                        cluster = [[i,j],[i,j+1],[i-1,j],[i-1,j+1]]
                        if np.max(Neighbouring_Cluster(mat,cluster)) < threshold2:
                            valid_detections_photons.append(mat[i][j] + mat[i-1][j] + mat[i][j+1] + mat[i-1][j+1])
                            valid_quad.append(mat[i][j] + mat[i-1][j] + mat[i][j+1] + mat[i-1][j+1])
                            quad_NE.append(mat[i][j])
                            quad_NE.append(mat[i][j+1])
                            quad_NE.append(mat[i-1][j])
                            quad_NE.append(mat[i-1][j+1])
                            pixels = [[i,j],[i,j+1],[i-1,j],[i-1,j+1]]
                            quad_moment.append(moment(mat,pixels))

                if np.min(Adjacents_Except_West(mat,i,j)) > threshold2 and mat[i][j-1] < threshold2:
                    if mat[i][j] + mat[i][j+1] + mat[i-1][j] + mat[i+1][j] > threshold1:
                        cluster = [[i,j],[i,j+1],[i-1,j],[i+1,j]]
                        if np.max(Neighbouring_Cluster(mat,cluster)) < threshold2:
                            quad_E.append(mat[i][j])
                            quad_E.append(mat[i][j+1])
                            quad_E.append(mat[i-1][j])
                            quad_E.append(mat[i+1][j])
                
                if np.min(Adjacents_Except_East(mat,i,j)) > threshold2 and mat[i][j+1] < threshold2:
                    if mat[i][j] + mat[i][j-1] + mat[i+1][j] + mat[i-1][j] > threshold2:
                        cluster = [[i,j],[i,j-1],[i+1,j],[i-1,j]]
                        if np.max(Neighbouring_Cluster(mat,cluster)) < threshold2:
                            quad_W.append(mat[i][j])
                            quad_W.append(mat[i][j-1])
                            quad_W.append(mat[i-1][j])
                            quad_W.append(mat[i+1][j])

                if np.min(Adjacents_Except_South(mat,i,j)) > threshold2 and mat[i+1][j] < threshold2:
                    if mat[i][j] + mat[i][j-1] + mat[i][j+1] + mat[i-1][j] > threshold2:
                        cluster = [[i,j],[i,j-1],[i-1,j],[i,j+1]]
                        if np.max(Neighbouring_Cluster(mat,cluster)) < threshold2:
                            quad_N.append(mat[i][j])
                            quad_N.append(mat[i][j-1])
                            quad_N.append(mat[i-1][j])
                            quad_N.append(mat[i][j+1])   

                if np.min(Adjacents_Except_North(mat,i,j)) > threshold2 and mat[i-1][j] < threshold2:
                    if mat[i][j] + mat[i][j-1] + mat[i+1][j] + mat[i][j+1] > threshold2:
                        cluster = [[i,j],[i,j-1],[i+1,j],[i,j+1]]
                        if np.max(Neighbouring_Cluster(mat,cluster)) < threshold2:
                            quad_S.append(mat[i][j])
                            quad_S.append(mat[i][j-1])
                            quad_S.append(mat[i+1][j])
                            quad_S.append(mat[i][j+1])

                            




                            

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

def make_hist(mat): # to make histogram of an image
    global ADU_Max
    #A = np.asarray(mat.reshape(-1))
    #plt.figure() 
    hist = plt.hist(mat,bins=ADU_Max,range=(0,ADU_Max),log=True) 
    #plt.show()
    ans = (hist[0],hist[1])
    return ans

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





















run_range = range(4,6) # give a set of run numbers
noise = 3
frac = 2
print("frac:",frac)
for run in run_range: # the set of images used to construct a spectrum
    print("run number", run)
    t1 = time.time() # for timming perpose


    #====================================================== import the raw data ======================================================


    f = h5py.File("22Jan\\r_0.4\\Noise_%s\\frac_%s\\MC_%d.hdf5"%(str(noise),str(frac),run),'r')    # where the raw data is, change to anyting like
    #f = h5py.File('F:\\nc\\run0%d.nc'%(run),'r')
    #print(list(f.keys()))
    dataset = f['data']      #import raw data
    #print(dataset.shape)
    total_image_number = dataset.shape[0]  # how many images in one run, should be 342 for majority
    print(total_image_number)

    
    #====================================================== import the raw data ======================================================
    
    ##################################################################################################################################
    
    #====================================================== raw data statistics =====================================================


    A = np.asarray(dataset).reshape(-1) # to make a histogram of the raw data
    plt.figure()
    hist = plt.hist(A,bins=500,range=(-100,400)) # to make a histogram of the raw data. Limit set to -100 ADU
    #plt.xlabel('# of ADUs') # to generate a histogram, not in use any longer
    #plt.ylabel('Frequency')
    #plt.yscale("log")
    #plt.show()
    #make_hist_and_plot(dataset[100])


    Threshold2 = 0  # 3 sigma of the background noise
    peak = np.argmax(hist[0][200:300]) + 100 # peak at XFEL Energy, should be 9keV. need +100 as the histogram starts at -100
    print('peak is %f' %(peak))
    for i in range(100,150):
        if hist[0][i] < 2.71828**(-4.5) * np.max(hist[0]): # find the three sigma from a Gaussian fit
            Threshold2 = i - 100 # the histogram starts at -100 ADU
            break
    print('3 sigma: %d' %(Threshold2))

    Threshold1 = 70 # threshold for valid photon detection, temperary set to be 70, therefore 70ADU * 60 eV/ADU = 4200 eV


    # ====================================================== raw data statistics  =====================================================
    
    ###################################################################################################################################
    
    #====================================================== clustering algorithm ======================================================


    all_detections_photons = [] # All detections recognized as photons attributed to the central pixel
    all_detections_pixels = [] # All detections recognized as photons with chargespread
    all_single = [] # All single pixel events
    
    valid_detections_photons = [] # Aingle photon clusters attributed the central one pixel
    valid_single = [] # Valid single pixel events
    valid_double = []
    valid_triple = []
    valid_quad = []
    
    double_E = [] # valid double pixel event
    double_W = []
    double_S = []
    double_N = []
    double_moment = []
    double_moment_max = []

    triple_SW = [] # valid triple pixel event
    triple_SE = []
    triple_NW = []
    triple_NE = []
    triple_row = []
    triple_bar = []
    triple_moment = []


    quad_SW = []
    quad_SE = []
    quad_NW = []
    quad_NE = []
    quad_W = []
    quad_E = []
    quad_S = []
    quad_N = []
    quad_moment = []



    for i in range(0,total_image_number):  #SPC clustering algorithm image by image
        t3 = time.time() # for timming perpose
        matrix = dataset[i] # import from dataset, the 704*768 pixel ADU values       

        cluster(matrix,threshold1= Threshold1,threshold2= Threshold2) # clustering algorithm
        #print(double_moment_max)
        
        t4 = time.time()
        print('\r'+'▋'*((i*10)//total_image_number)+"  "+str((i*100)//total_image_number)+'%', end='')
        #print('iteration: %d, time: %f'%(i,t4-t3)) # show which iteration we are in now
    
    valid_double_pixels = double_E+double_N+double_S+double_W # count the number of valid double events
    valid_triple_pixels = triple_NE + triple_NW + triple_SE + triple_SW
    valid_quad_pixels = quad_NE + quad_NW + quad_SE + quad_SW
    valid_detections_pixels = valid_single + valid_double + valid_triple + valid_quad


    #====================================================== clustering algorithm ======================================================
    
    ###################################################################################################################################






    mk = "SPC data analysis\\22Jan\\r_0.4\\noise_%s\\frac_%s\\%d"%(str(noise),str(frac),run)
    
    mkdir(mk)

    
    #======================================================    making spectrum   ======================================================
    
    
    ADU_Max = 1000
    all_detection_hist = make_hist(all_detections_photons) # histogram of all detections
    all_single_hist = make_hist(all_single)
    valid_detection_hist = make_hist(valid_detections_photons) 
    valid_single_hist = make_hist(valid_single) 
    valid_double_hist = make_hist(valid_double)
    valid_triple_hist = make_hist(valid_triple)
    valid_quad_hist = make_hist(valid_quad)

    
    hist_length = ADU_Max - Threshold1

    Spectrum = {} # prepare to record the spectrum 

    #recording the spectrum
    Spectrum['ADUs'] = valid_detection_hist[1][Threshold1:ADU_Max] #X Photon Energy
    Spectrum['All Detections'] = all_detection_hist[0][Threshold1:ADU_Max]
    Spectrum['All Single'] = all_single_hist[0][Threshold1:ADU_Max]
    Spectrum['Valid Detections'] = valid_detection_hist[0][Threshold1:ADU_Max] 
    Spectrum['Valid Single'] = valid_single_hist[0][Threshold1:ADU_Max]
    Spectrum['Valid Double'] = valid_double_hist[0][Threshold1:ADU_Max]
    Spectrum['Valid Triple'] = valid_triple_hist[0][Threshold1:ADU_Max]
    Spectrum['Valid Quad'] = valid_quad_hist[0][Threshold1:ADU_Max]


    # save the spectrum data
    df1 = pandas.DataFrame(Spectrum, columns = ['ADUs', 'All Detections','Valid Detections','All Single','Valid Single',\
        'Valid Double','Valid Triple','Valid Quad'])
    df1.to_excel (mk+"\\%d - spec.xlsx"%(run),sheet_name='Spectrum')


    #====================================================== making spectrum ======================================================
    
    ##############################################################################################################################
    
    #====================================================== Fill fractions =======================================================
    
    
    Threshold2_Mat = Threshold1 * np.ones((total_image_number,704,768))

    all_detections_photon_count = len(all_detections_photons)
    all_detections_pixel_count = len(all_detections_pixels) # # of pixels in all photon detections
    all_single_count = len(all_single)
    valid_detections_photon_count = len(valid_detections_photons)
    valid_detections_pixel_count = len(valid_detections_pixels)
    valid_single_count = len(valid_single)

    illuminated_pixels_count = np.sum(dataset > Threshold2_Mat)
    

    all_detections_photon_fraction = 100 * (all_detections_photon_count/(704 * 768 * total_image_number))
    all_detections_pixel_fraction = 100 * (all_detections_pixel_count/(704 * 768 * total_image_number))
    all_single_fraction = 100 * (all_single_count/(704 * 768 * total_image_number))
    valid_detections_photon_fraction = 100 * (valid_detections_photon_count/(704 * 768 * total_image_number))
    valid_detections_pixel_fraction = 100 * (valid_detections_pixel_count/(704 * 768 * total_image_number))
    valid_single_fraction = 100 * (valid_single_count/(704 * 768 * total_image_number))
    illuminated_pixels_fraction = 100 * (illuminated_pixels_count/(704 * 768 * total_image_number))
    

    Fill_Fraction = {
        'All Detections Photons': [all_detections_photon_count,all_detections_photon_fraction],
        'All Detections Pixels': [all_detections_pixel_count,all_detections_pixel_fraction],
        'All Single': [all_single_count,all_single_fraction],
        'Valid Detections Photons': [valid_detections_photon_count,valid_detections_photon_fraction],
        'Valid Detections Pixels': [valid_detections_pixel_count,valid_detections_pixel_fraction],
        'Valid Single': [valid_single_count,valid_single_fraction],
        'Illuminated Pixels':[illuminated_pixels_count,illuminated_pixels_fraction]
    }

    df2 = pandas.DataFrame(Fill_Fraction, columns = ['All Detections Photons', 'All Detections Pixels', \
        'All Single','Valid Detections Photons','Valid Detections Pixels','Valid Single','Illuminated Pixels'])
    df2.to_excel(mk+"\\%d - frac.xlsx"%(run),sheet_name='Fractions')


    #======================================================     Fill fractions     ======================================================
    
    #####################################################################################################################################
    
    #====================================================== Statistics of clusters ======================================================
    
    
    #Count the number of each events
    Double_E_count = len(double_E)/2
    Double_W_count = len(double_W)/2
    Double_S_count = len(double_S)/2
    Double_N_count = len(double_N)/2

    Triple_SW_count = len(triple_SW)/3
    Triple_SE_count = len(triple_SE)/3
    Triple_NW_count = len(triple_NW)/3
    Triple_NE_count = len(triple_NE)/3

    Triple_row_count = len(triple_row)/3
    Triple_bar_count = len(triple_bar)/3

    Quad_SW_count = len(quad_SW)/4
    Quad_SE_count = len(quad_SE)/4
    Quad_NW_count = len(quad_NW)/4
    Quad_NE_count = len(quad_NE)/4

    Quad_E_count = len(quad_E)/4
    Quad_W_count = len(quad_W)/4
    Quad_S_count = len(quad_S)/4
    Quad_N_count = len(quad_N)/4

    Statistics = {
        "Single": [valid_single_count],
        "Double E": [Double_E_count],
        "Double W": [Double_W_count], 
        "Double S": [Double_S_count],
        "Double N": [Double_N_count],
        "Triple SW": [Triple_SW_count],
        "Triple SE": [Triple_SE_count],
        "Triple NW": [Triple_NW_count], 
        "Triple NE": [Triple_NE_count],
        "Triple Row": [Triple_row_count],
        "Triple Bar": [Triple_bar_count],
        "Quad SW": [Quad_SW_count],
        "Quad SE": [Quad_SE_count],
        "Quad NW": [Quad_NW_count],
        "Quad NE": [Quad_NE_count],
        "Quad N": [Quad_N_count],
        "Quad E": [Quad_E_count],
        "Quad S": [Quad_S_count],
        "Quad W": [Quad_W_count]

    }


    df3 = pandas.DataFrame(Statistics,columns=["Single","Double E","Double W","Double S","Double N",\
        "Triple SW","Triple SE","Triple NW","Triple NE","Triple Row","Triple Bar",\
        "Quad SW","Quad SE","Quad NW","Quad NE","Quad N","Quad E", "Quad S", "Quad W"])
    df3.to_excel(mk+"\\%d - statistics.xlsx"%(run),sheet_name='statistics')


    #====================================================== Statistics of clusters =================================================
    
    ################################################################################################################################
    
    #====================================================== Moment statistics ======================================================

    double_moment_hist = moment_hist(double_moment)
    triple_monment_hist = moment_hist(triple_moment)
    quad_moment_hist = moment_hist(quad_moment)
    


    Moment_hist = {}
    Moment_hist['moment'] = double_moment_hist[1][0:100]
    Moment_hist['double'] = double_moment_hist[0]
    Moment_hist['triple'] = triple_monment_hist[0]
    Moment_hist['quad'] = quad_moment_hist[0]

    df4 = pandas.DataFrame(Moment_hist, columns = ['moment', 'double', 'triple','quad']) 
    df4.to_excel(mk+"\\%d - moment.xlsx"%(run),sheet_name='Moments')


    #====================================================== Moment statistics ======================================================  
    
    ################################################################################################################################
    
    #====================================================== Moment 0.25 statistics =================================================

    double_moment_max_hist = make_hist(double_moment_max)
    #print(double_moment_max)
    #print(double_moment_max_hist)
    

    


    Moment_max_hist = {}
    Moment_max_hist['ADUs'] = double_moment_max_hist[1][Threshold1:ADU_Max]
    Moment_max_hist['Distribution'] = double_moment_max_hist[0][Threshold1:ADU_Max]


    df5 = pandas.DataFrame(Moment_max_hist, columns = ['ADUs', 'Distribution']) 
    df5.to_excel(mk+"\\%d - moment max.xlsx"%(run),sheet_name='Moment max')


    #====================================================== Moment 0.25 statistics =================================================


    t2 = time.time()
    print("")
    print('totaltime: %f'%(t2-t1))

   
    




