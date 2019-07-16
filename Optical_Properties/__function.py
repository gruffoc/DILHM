#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 19 16:08:33 2019

@author: claudriel
"""

import sys
import os
import matplotlib
from pylab import *
from scipy.ndimage import measurements
from PIL import Image
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from PIL import ImageStat
import holopy as hp
from holopy.core.process import bg_correct, subimage, normalize,center_find
import cv2
import imutils
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours

"""
All the functions that I used
"""

def mediana(directory_sample, directory_save_bg, directory_save_correct, a, b, N):
    """
    Calculates the medians of a data set and subtracs them to each image.
    Parameters
    ----------
    directory_sample: str
       Path of the directory of the data set
    directory_save_bg: str
        Path of the directory where it saves the background (the medians)
    directory_save_correct: str
       Path of the directory where it saves the final coorected images
    a = int
        Initaila range of the time you want repeat the median
    b = int
        Final range of the time you want repeat the median
    N = int
        Number of the data set lengh
    
    Returns
    -------
    0 : The images are automatically saved in the path and they are ready to
    the use
    """  
    img_list= os.listdir(directory_sample)
    img_list.sort()
    
    for i in range(0,N):
        I_array = []
    
        for k in range(a,b):
            
            I_array.append(Image.open(directory_sample + img_list[k]).convert("L"))
        I_array = np.array([np.asarray(im) for im in I_array])      
        img_median = np.median(I_array,axis=0)
                  
        result = Image.fromarray(img_median.astype('uint8'))
        result.save(directory_save_bg+str(a)+'_'+str(b)+'.tiff')
        
        for j in range(a,b):
            print(j)
            
            im = Image.open(directory_sample + img_list[j]).convert("L")
            I = np.asarray(im)
            
            I_correct = I - img_median
            minimo = np.ones((1024,1280))*np.abs(np.amin(I_correct))
            I_correct = I_correct + minimo
            
            result = Image.fromarray(I_correct.astype('uint8'))
            result.save(directory_save_correct + str(j) + '.tiff')
           
        a = a + 10
        b = b + 10
    return (0)


def propagation_module(z, rec_vol, lim):
    """
    Calculates the module of the hologram along the optical axis, only in the 
    center of the hologram.
    Parameters
    ----------
    rec_vol : :class:`.Image` or :class:`.VectorGrid`
       Hologram in function of x,y,z
    z: float or list of floats
       Distance to propagate. 
    lim: int 
        Center of the hologram to propagate
    
    Returns
    -------
    module_arr : np.array
       The module of the hologram progagated to a distance d from its current
       location calculated at the center of the hologram.
    """  
    module_arr=np.array([])   
    for j in range(0,len(z)):
            
        module=np.abs(rec_vol[lim][lim][j])
        module_arr=np.append(module_arr,module)
    return(module_arr)

      
def propagation_phase(phase, p, z, lim):
    """
    Calculates the phase of the hologram along the optical axis, only in the 
    center of the hologram.
    Parameters
    ----------
    phase : :class:`.Image` or :class:`.VectorGrid`
        Phase of the Hologram in function of x,y,z
    p: np.array
        Reference wave, that hasn't scattered
    z:  float or list of floats
       Distance to propagate. 
    lim: int 
        Center of the hologram to propagate
    
    Returns
    -------
    phase_arr : np.array
       The phase of the hologram progagated to a distance d from its current
       location calculated at the center of the hologram with respect to the
       reference wave.
    """  
    p_arr=np.array([])
    phase_arr=np.array([])
    for j in range(0,len(z)):
        
        diff=phase[lim-2:lim+2, lim-2:lim+2, j] - p[j]
        diff= np.mean(diff)
        phase_arr=np.append(phase_arr,diff)
        p_arr=np.append(p_arr,p[j])
    
        phase_arr[phase_arr>np.pi] = 0
        phase_arr[phase_arr<-np.pi] = 0
    return(phase_arr)
    
    
def maximum_minimum(array, z):
    """
    Calculates the max and min value of an array
    Parameters
    ----------
    array: np.array, float
    z:  float or list of floats
        Distance to propagate  
    
    Returns
    -------
    d_max: float
        Distance[pixels] at which the array have the maximum value
    d_min: float
        Distance[pixels] at which the array have the minimun value
    z_max: int
        Array position at which the array have the maximum value
    z_min: int
        Array position at which the array have the minium value    
    """
    
    max_array = np.amax(array[1:])
    d_max = z[np.where(array == max_array)[0]]
    z_max = np.where(array == max_array)[0]
    
    min_array = np.amin(array[1:])
    d_min = z[np.where(array == min_array)[0]]
    z_min = np.where(array == min_array)[0]
    return(d_max, d_min, z_max, z_min)
    
    
def plot_twin_propagation(z, module_arr, phase_arr, directory_graph):
    """
    Calculates the plot of the propagation of the hologram along the optical
    axis and at the center of the hologram both studing the intensity of the
    field and the phase of the field. 
    Parameters
    ----------
    z: float or list of floats
        Distance to propagate  
    module_arr: np.array, float 
        Array of the intensity of the field propagated
    phase_arr: np.array, float 
        Array of the phase of the field propagated
    directory_graph: str
        Path where the graph is saved         
        
    Returns
    -------
    0: the graph is saved authomatically.
        By the graph the point of discontinuity can be seen and it can be possibkìle
        calculate the z position of the particle     
    """
    plt.figure(1)
    fig, ax1 = plt.subplots()
    ax1.plot(z, module_arr, '-b*', label='module')
    ax1.set_xlabel('z($\mu$m)')
    ax1.set_ylabel('|U|', color='b')
    ax1.tick_params('y', colors='b')
    ax2 = ax1.twinx()
    phase_arr[phase_arr == 0] = None
    ax2.plot(z,phase_arr,'r*', label='phase')
    ax2.set_ylabel('$\phi$(U)', color='r')
    ax2.tick_params('y', colors='r')
    plt.savefig(directory_graph)
    return (0)


def treshold(z, phase, p, lim):
    """
    Calculates the phase of the hologram, with the respect of the reference
    wave, at the initial position of the particle.
    Parameters
    ----------
    z: int
        Position of the particle object
    phase: np.array, float 
        Array of the phase of the field propagated
    p: np.array
        Reference wave, that hasn't scattered
    lim: int 
        Center of the hologram to propagate   
        
    Returns
    -------
    diff: :class:`.Image` or :class:`.VectorGrid`
        Matrix of the phase hologram at the plane of the focus (object position)
    """
    p = p[z] * np.ones((lim*2,lim*2))
    diff=phase[:,:,z]
    return (diff)


def midpoint(ptA, ptB):
    """
    Calculates the middle point from two point 
    Parameters
    ----------
    ptA:  int
       Position
    ptB:  int
       Poiìsition    
        
    Returns
    -------
    The middle point: float
    """
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


def object_dimension(directory_obj, pixel_size,lim):
    """
    Calculates the diameters of an object, not circular.
    
    It first performs edge detection, then performs a dilation + erosion to
    close gaps in between object edges.
    Then for each object in the image, it calcaluates the contourns of the
    minimum box (minimum rectangle that circumvent the object) and it sorts
    them from left-to-right (allowing us to extract our reference object).
    It unpacks the ordered bounding box and computes the midpoint between the
    top-left and top-right coordinates, followed by the midpoint between
    bottom-left and bottom-right coordinates.
    Finally it computes the Euclidean distance between the midpoints.
    
    Parameters
    ----------
    directory_obj: str
       Path of the directory of the image of the object reconstructed at the 
       focal point.
    pixel_size: float
        Value of the pixel size (um)
    lim: int
        Value of the half shape of the new matrix. It will be the new center 
    
    Returns
    -------
    orig: :class:`.Image` or :class:`.VectorGrid`
        Original image plus the dimension of the diameter labelled
    dimA: float
        Value of the the first diameter
    dimB: float
        Value of the the second diameter
    ratio: float
        Value of the ratio of the two diameters
    """
    
    image = cv2.imread(directory_obj)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #perform the countour
    gray = cv2.GaussianBlur(gray, (7, 7), 0) 
    edged = cv2.Canny(gray, 50, 100)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)
    # find contours in the edge map
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    # sort the contours from left-to-right
    (cnts, _) = contours.sort_contours(cnts)

    for c in cnts:
        if cv2.contourArea(c) < 5:
            continue
        # compute the rotated bounding box of the contour
        orig = image.copy()
        box = cv2.minAreaRect(c)
        box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
        box = np.array(box, dtype="int")
        # order the points in the contour such that they appear in top-left
        box = perspective.order_points(box)
        # Compute the midpoint
        (tl, tr, br, bl) = box
        (tltrX, tltrY) = midpoint(tl, tr)
        (blbrX, blbrY) = midpoint(bl, br)
        (tlblX, tlblY) = midpoint(tl, bl)
        (trbrX, trbrY) = midpoint(tr, br)
        # draw lines between the midpoints
        cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)), (5, 0, 255), 2)
        cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)), (255, 0, 255), 2)
        # compute the Euclidean distance between the midpoints
        # dA  variable will contain the height distance (pixels)
        # dB  will hold our width distance (pixels).
        dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
        dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
        # compute the size of the object
        dimA = dA * pixel_size
        dimB = dB * pixel_size
        
        diff = dA - dB
        if diff < 0:
            ratio = dB/dA
        else:
            ratio = dA/dB
        
        cv2.putText(orig, "{:.1f}um".format(dimA),
		(int(tltrX - 15), int(tltrY - 15)), cv2.FONT_HERSHEY_SIMPLEX,
		0.65, (100, 100,100), 2)
        cv2.putText(orig, "{:.1f}um".format(dimB),
		(int(trbrX + 10), int(trbrY+ 15)), cv2.FONT_HERSHEY_SIMPLEX,
		0.65, (100, 100,100), 2)
    return(orig, dimA, dimB, ratio)
    

    
def Area(list_radii, dati):
    """
    Calculates the angular medium of an hologram from a data file with three
    columns: data[0] is the x cordinates, data[1] is the y cordinates and 
    data[3] is the respectively hologram intensity.
    It sums the value within two ray and then it average them.
    The function cycles on ray value 
    Parameters
    ----------
    list_radii: np.array, int
        List of "ray" at which the angular medium is calculate
    dati: float by a data file
       The file.data saved for calculate the extinction coefficient 
        
    Returns
    -------
    total_aver: np.array
        Array of the angular medium of the image. 
    """
    
    total_aver = np.array([])
    restrict2=np.array([])
    restrict=np.array([])
    
    for i in list_radii:
        restrict2 = dati[(np.sqrt(dati[:,0]**2+dati[:,1]**2))<i] #dati nei due cerchi
        restrict = restrict2[(np.sqrt(restrict2[:,0]**2+restrict2[:,1]**2))>=(i-1)]
        
        aver = np.sum(restrict[:,2])/len(restrict[:,2])
        total_aver=np.append(total_aver,aver)

    return (total_aver)


def rebin(a, shape, pixcombine):
    """
    Reshape an array (matrix) to a give size using either the sum, mean or median of the
    pixels binned.
    Note that the old array dimensions have to be multiples of the new array
    dimensions.
    Parameters
    ----------
    a: :class:`.Image` or :class:`.VectorGrid`
        Array to reshape (combine pixels)
    shape: (int, int)
        New size of array
    pixcombine: str
        The method to combine the pixels with. Choices are sum, mean and median
        
    Returns
    -------
    reshaped_array: :class:`.Image` or :class:`.VectorGrid`
        Matrix with the new shape binned
    """
    sh = shape[0], a.shape[0]//shape[0], shape[1], a.shape[1]//shape[1]
    if pixcombine == 'sum':
        reshaped_array = a.reshape(sh).sum(-1).sum(1)
    elif pixcombine == 'mean':
        reshaped_array = a.reshape(sh).mean(-1).mean(1)
    elif pixcombine == 'median':
        reshaped_array = a.reshape(sh).median(-1).median(1)

    return reshaped_array
    

def calcolo_hologram(cartella, name, pixel_size, lim):
    """
    Open the image with the correspective path, it calculates the center of the
    hologram and it prints it on the console.
    Then cut the image with a fixed dimension around the center. So the new 
    center of the image is fixed.
    Parameters
    ----------
    cartella: str
        Name of the folder of the image
    name: str
        Number within the name of the image (without type)
    pixel_size: float
        Value of the pixel size (um)
    lim: int
        Value of the half shape of the new matrix. It will be the new center 
        
    Returns
    -------
    data_holo: :class:`.Image` or :class:`.VectorGrid`
        Data reshaped of the hologram
    """
    raw_holo = hp.load_image("../Campioni/Flusso/"+cartella+"/img_correct/img_" + name + ".tiff", spacing = pixel_size)
    
    centro = center_find(raw_holo, centers=1, threshold=0.3, blursize=6.0)
    print(centro)
    data_holo = raw_holo[0, int(centro[0]-lim) : int(centro[0]+lim), int(centro[1]-lim) : int(centro[1]+lim)]  
    
    hp.show(data_holo)
    plt.show()
    
    return(data_holo)
    


def calcolo_hologram_BINNING(cartella, name, pixel_size, lim, binsize, pixcombine):
    """
    Open the image with the correspective path, it calculates the center of the
    hologram and it prints it on the console.
    Then cut the image with a fixed dimension around the center. So the new 
    center of the image is fixed.
    Finally it rebins the image.
    
    Warning: to find the center you have to open the image with holopy function.
    But you can't rebbined DataArray. So you had to open it also with PIL.
    !!!!Maybe you can correct this in a second time!!!!
    Parameters
    ----------
    cartella: str
        Name of the folder of the image
    name: str
        Number within the name of the image (without type)
    pixel_size: float
        Value of the pixel size (um)
    lim: int
        Value of the half shape of the new matrix. It will be the new center
    binsize: int
        Value of the new reshape
    pixcombine: str
        The method to combine the pixels with. Choices are sum, mean and median
        
    Returns
    -------
    data_holo: :class:`.Image` or :class:`.VectorGrid`
        Data reshaped of the hologram
    """
    raw_holo = hp.load_image("../Campioni/Flusso/"+cartella+"/img_correct/img_" + name + ".tiff", spacing = pixel_size)  
    hp.show(raw_holo)
    plt.show()
    
    im = Image.open("../Campioni/Flusso/"+cartella+"/img_correct/img_" + name + ".tiff").convert("L")
    I  = np.asarray(im)
    
    centro = center_find(raw_holo, centers=1, threshold=0.3, blursize=6.0)
    print(centro)
    data_holo = I[int(centro[0]-lim) : int(centro[0]+lim), int(centro[1]-lim) : int(centro[1]+lim)]
    
    data_holo = rebin(data_holo, ((binsize, binsize)), pixcombine)
    lim = lim/2
    
    hp.show(data_holo)
    plt.show()
    
    return(data_holo)
    
def create_file_dat_BINNING(data_holo, lim, name_file):
    """
    This writes on a file.dat the x, y index and the relative intensity
    value of an hologram image in three different columns. 
    Parameters
    ----------
    data_holo: :class:`.Image` or :class:`.VectorGrid`
        Matrix data of the hologram, that can be binned or not
    lim: int
        Value of the half shape of the new matrix. It will be the new center
    name_file: str
        Name of the file dat to write on
        
    Returns
    -------
    dati: .dat
        The file.dat with x, y, and intensity value in three different columns
    """
    leng = lim*2
    dati = open('name_file','w+')
    for i in range(0,leng):
        for j in range(0,leng):
            print ( str(str((i-lim))+" "+str((j-lim))+" "+str(data_holo[i][j])),file=dati)
    dati.close() 
    
    dati = np.loadtxt(open("name_file","rb")) 
    return(dati)
    
    
def create_file_dat(data_holo, lim, name_file):
    """
    This writes on a file.dat the x, y index and the relative intensity
    value of an hologram image in three different columns. 
    Parameters
    ----------
    data_holo: :class:`.Image` or :class:`.VectorGrid`
        Matrix data of the hologram, that can be binned or not
    lim: int
        Value of the half shape of the new matrix. It will be the new center
    name_file: str
        Name of the file dat to write on
        
    Returns
    -------
    dati: .dat
        The file.dat with x, y, and intensity value in three different columns
    """
    leng = lim*2
    dati = open('name_file','w+')
    for i in range(0,leng):
        for j in range(0,leng):
            print ( str(str((i-lim))+" "+str((j-lim))+" "+str(data_holo[i][j].values)),file=dati)
    dati.close() 
    
    dati = np.loadtxt(open("name_file","rb")) 
    return(dati)
        