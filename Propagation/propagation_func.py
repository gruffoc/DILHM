#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 13:48:41 2019

@author: claudiaravasio
Programma che prende in input le immagini (gia` elaborate) e fa la propagazione
del suo campo. 

"""

import sys
import os
import matplotlib
#matplotlib.use('Agg')
from pylab import *
from scipy.ndimage import measurements
from PIL import Image
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from numpy.fft import fft2, fftshift, ifftshift, ifft2
import xarray as xr


"""
(a)Calculating the Fourier transform of H(X,Y)
(b) Simulating the angular spectrum propagator
(c) Multiplying the results of (a) and (b)
(d) Calculating the inverse Fourier transform of (c). The result provides t(x, y).
"""

def G(fx, fy, z, wavelength):
    square_root = np.sqrt(1 - (wavelength**2 * fx**2) - (wavelength**2 * fy**2))
    temp = np.exp(1j * 2 * np.pi * i / wavelength * square_root)
    temp[np.isnan(temp)] = 0 # replace nan's with zeros

    return temp


if __name__ == '__main__':
    lamda=0.660
    dx=0.1  #micron
    N=512
    k=2*np.pi/lamda
    z=np.linspace(0,100,100)
    
    holo_img = Image.open('./M/holo2um.png').convert("L")
    holo_array  = np.asarray(holo_img)
    #plt.imshow(holo_array)
    #plt.colorbar()
    #plt.show()

    ft_holo=ifftshift(fft2(fftshift(holo_array)))

    x = np.linspace(0, N, N)
    f_max = 1/dx         # Spatial sampling frequency
    f_min = f_max/N # Spacing between discrete frequency coordinates
    fx = np.arange(-f_max/2, f_max/2, step = f_min) # Spatial frequency

    FX, FY = np.meshgrid(fx, fx)
    

    phase=np.array([])
    for i in z:
        g=G(FX, FY, z, lamda)
        prop = ft_holo * g
        U=fftshift(ifft2(ifftshift(prop)))/250
        phase=np.append(phase,np.angle(U[256,256]))
    
    plt.plot(z,phase)

    plt.show()
