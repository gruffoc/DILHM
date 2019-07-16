#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 16:25:47 2019

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
from __function import *
from holopy.core.process import bg_correct, subimage, normalize,center_find
import numpy as np
from scipy.signal import argrelextrema
from scipy import optimize
from holopy.scattering import calc_cross_sections
from PIL import Image

"""
Because of the scatterer size, we have a reducted power of the field at the
plane of the detector.
This script calculates the Extinction Coefficient by a Hologram through the 
Optics Thorem.

-It opens an image and calculates the center of it, then it reshapes the image
    over this center. If the image is very noisy, it can be possible to bin the
    image in order to have more signal, but you have to select the relative
    code.
    data_holo represent the hologram matrix in format DataArray with x, y, z
    coordinates
    lim is the new center of the hologram, and leng the total lengh.
    Returns on the console: an image of the hologram with the new shape and the
    center of the image
    
-It creates a file.dat where the x,y and intensity values are splitted in 3
    different columns
    
-It calculates the angular average of the hologram, that is "total_aver" for 
    a list of radii.
    Returns three plot:
    1) A line vertical of the hologram data along the center in function of 
    distance (um)
    2) The angular average in function of distance (um)
    3) The angular average in function of distance (um) - until the first max

-It fits the curve with the following free parameters:
    A amplitude and offset 
    S(0) the scattering value at the 0 angle. It is shown as the difference in
        amplitude (visibility) from an hologram path that have a point-like
        object scatterer. The real part of it is related to the Extinction
        Coefficient by the Optics Theorem
    P is the phase deley respect an hologram path that have a point-like
        object scatterer. 
    sigma is the variance of an exponential decay.
    Returns: the plot of the fit and the value of the Cext

-It calculates the real value of the Cext to confront it.

"""

if __name__ == "__main__":
    
    ##### Path name
    name = "142"
    cartella = "Poly/2um/1"
    name_file = 'esempiostupido.dat'
    
    #### Image setup
    medium_index = 1.33
    pixel_size = 0.236
    illum_wavelen = 0.6328
    lim = 290
    leng = lim*2
    k=np.pi*2/(illum_wavelen/medium_index)
    
    zeta = 350 *0.236
    r = 2
    
    ############################  
    """
    For NO binning here
    """
#    data_holo = calcolo_hologram(cartella, name, pixel_size, lim)
    #############################
    """
    For BINNING here
    """
    binsize = lim
    pixcombine = 'mean'
    data_holo = calcolo_hologram_BINNING(cartella, name, pixel_size, lim, binsize, pixcombine)
    lim =int( lim/2) 
    ###############################
    
    dati = np.loadtxt(open("name_file","rb"))
    line_vert_holo = data_holo[lim:, lim]
    x_range = np.arange(0, lim, 1)* pixel_size
    list_radii = np.arange(0, lim, 0.5)
    total_aver = Area(list_radii, dati)  #media angolare
    primo_maximo = argrelextrema(total_aver, np.greater)[0][0]
        
    plt.figure(0)
    plt.plot(x_range,line_vert_holo, '-*')
    plt.title("Hologram verticla line by the center")
    plt.xlabel("R um")
    plt.show()
    plt.clf()
    
    plt.figure(1)
    plt.plot(list_radii*pixel_size, total_aver, '-*')
    plt.title("Angular average in function of distance (um)")
    plt.xlabel("R um")
    plt.show()
    plt.clf()
    
    plt.figure(2)
    plt.plot(list_radii[:primo_maximo]*pixel_size, total_aver[:primo_maximo], '-*')
    plt.title("Angular average in function of distance (um) - until the first max")
    plt.xlabel("R um")
    plt.show()
    plt.clf()
      
    """
    Fit
    """
    S = -139.61378624
    P = 33.19096515
    A = 1
    sigma = 18
    B= 0
#    zeta = 420

    def func(x,A, S, P, B):
        return np.abs(A)**2+(2*A*np.abs(S)/(k*zeta)*np.cos(k*x**2/(2*zeta)+P))*B*e**(-(x**2)/(2*sigma**2))
    
    params, params_covariance = optimize.curve_fit(func, x_range[1:primo_maximo], total_aver[1:primo_maximo], p0=[A, S, P, B])
    print(params)

    plt.figure(figsize=(6, 4))
    plt.plot(x_range[1:primo_maximo],  total_aver[1:primo_maximo], '.b',label='Data')
    plt.plot(x_range[1:primo_maximo], func(x_range[1:primo_maximo],params[0], params[1], params[2], params[3]),'-r', label='Fitted function')

    Cext=4*np.pi/(k**2)*np.real(params[1]*np.cos(np.pi/2-P))   
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title(r)
    plt.xlabel("R um")
    plt.ylabel("Intensity (a.u)")
    plt.figtext(0.4, 0.8,"Cext= "+ str("{0:.2f}".format(Cext)))


    """
    Real Extinction Coefficient
    """
    distant_sphere = Sphere(r=r, n=1.59)
    x_sec = calc_cross_sections(distant_sphere, medium_index, illum_wavelen)
    print('Cext real=',x_sec[2])



#############################################
#    somma_ver =np.array([0])
#    for i in np.arange(0,10):
#        vert = data_holo[0,lim:lim+100,lim-5+i]
#        somma_ver = somma_ver + vert
#    plt.figure(0)
#    plt.plot(somma_ver/10, "-*")
#    
#    somma_oriz =np.array([0])
#    for i in np.arange(0,10):
#        oriz = data_holo[0,lim-5+i,lim:lim+100]
#        somma_oriz = somma_oriz + oriz
#    plt.figure(1)
#    plt.plot(somma_oriz/10, "-*")
    
    