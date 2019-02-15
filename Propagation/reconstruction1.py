#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 09:31:20 2019

@author: claudiaravasio

    RECONSTRUCTION
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
from numpy.fft import fft2, fftshift, ifftshift, ifft2
import xarray as xr
from holopy.core.io import get_example_data_path, load_average
from holopy.core.process import bg_correct
import holopy as hp
from scipy.misc import fromimage
from PIL import Image as pilimage
from propagation_func import propagate
  
if __name__ == '__main__':
    
    """
    PARAMETERS
    """
    lamda=0.660
    med_wavelength=lamda/1.33
    N=512
    x = np.linspace(0, N, N)
    dx=0.0851  #micron
    
    k=2*np.pi/med_wavelength
#    z=np.linspace(0,100,100)
    
    """
    LOAD IMAGE
    """

    holo_img = Image.open('image01.jpg').convert("L")
    bg_img=Image.open('bg01.jpg').convert("L")
    holo_array  = np.asarray(holo_img)
    bg_array=np.asarray(bg_img)
    
    holo=holo_array/bg_array

#
    z = np.linspace(0, 30, 100)

    phase_arr=np.array([])
    for i in range(0,len(z)):
        rec_vol = propagate(holo, z[i], med_wavelength, dx, N)
        phase=np.angle(rec_vol[286][253])
        p=np.angle(np.e**(-1j*2*np.pi*z[i]/(0.66/1.33)))
        diff=phase-p
        
    
        phase_arr=np.append(phase_arr,diff)
        phase_arr[phase_arr>np.pi]=0
        phase_arr[phase_arr<-np.pi]=0    

    plt.plot(z,phase_arr,'*')
    plt.ylim(-np.pi, np.pi)
    plt.xlabel('z($\mu$m)')
    plt.ylabel('$\phi$(U)')
    plt.show()
    plt.clf() 
  
#    z = np.linspace(16,18,20)
#    #    
#
#    for i in z:
#        print(i)
#        rec_vol = propagate(holo, i, med_wavelength)
#        onda_piana=e**(-1j*i*k)*np.ones((N,N))
#        plt.imshow(np.angle(rec_vol)-np.angle(onda_piana))
#        plt.colorbar()
#        plt.title(i)
#        plt.show()
#        plt.clf()
#    phase=np.array([])
#    for i in z:
#        
#        phase=np.append(phase,np.angle(U[256,256]))
#    
#    plt.plot(z,phase)
#
#    plt.show()
