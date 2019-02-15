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



"""
(a)Calculating the Fourier transform of H(X,Y)
(b) Simulating the angular spectrum propagator
(c) Multiplying the results of (a) and (b)
(d) Calculating the inverse Fourier transform of (c). The result provides t(x, y).
"""
  

def trans_func(d, med_wavelength):
    
    """
    Calculates the optical transfer function to use in reconstruction
    This routine uses the Angular Spectrum propagator.
    Parameters
    ----------
    -d : float or list of floats
       reconstruction distance.  If list or array, this function will
       return an array of transfer functions, one for each distance
    -wavelen : float
       the wavelength in the medium you are propagating through
    -gradient_filter : float (optional)
       Subtract a second transfer function a distance gradient_filter
       from each z
    Returns
    -------
    trans_func : np.ndarray
       The calculated transfer function.  This will be at most as large as
       shape, but may be smaller if the frequencies outside that are zero
    
    """
    
    f_max = 1/dx         # Spatial sampling frequency
    f_min = f_max/N     # Spacing between discrete frequency coordinates
    fx = np.arange(-f_max/2, f_max/2, step = f_min) 
    FX, FY = np.meshgrid(fx, fx)
    
    root = 1 - (med_wavelength * FX)**2 - (med_wavelength * FY)**2
    root *= (root >= 0)

    g = np.exp(-1j*2*np.pi*d/med_wavelength*np.sqrt(root))
    g = g*(root>=0)

    return g

    # set the transfer function to zero where the sqrt is imaginary
    # (this is equivalent to making sure that the largest spatial
    # frequency is 1/wavelength).  (root>=0) returns a boolean matrix
    # that is equal to 1 where the condition is true and 0 where it is
    # false.  Multiplying by this boolean matrix masks the array.
    #Un modo equivalente e`
    #g[np.isnan(g)] = 0 # replace nan's with zeros


def propagate(data, d, med_wavelength):
    
    """
    Propagates a hologram along the optical axis
    Parameters
    ----------
    data : :class:`.Image` or :class:`.VectorGrid`
       Hologram to propagate
    d : float or list of floats
       Distance to propagate, in meters, or desired schema.  A list tells to
       propagate to several distances and return the volume
    gradient_filter : float
       For each distance, compute a second propagation a distance
       gradient_filter away and subtract.  This enhances contrast of
       rapidly varying features
    Returns
    -------
    data : :class:`.Image` or :class:`.Volume`
       The hologram progagated to a distance d from its current location.
    """

    G = trans_func(d, med_wavelength)

    ft_holo=ifftshift(fft2(fftshift(data)))
    
    prop = ft_holo * G
    
    U=fftshift(ifft2(ifftshift(prop)))#/250
    
    return U

    #we may have lost coordinate values to floating point precision during fft/ifft

#    if contains_zero:
#        d = d_old
#        U = xr.concat([data, U], dim='z')


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
#    plt.imshow(holo, 'gray')
    
    

    """
    RECONSTRUCTION
    """

    z = np.linspace(0, 30, 100)

    phase_arr=np.array([])
    for i in range(0,len(z)):
        rec_vol = propagate(holo, z[i], med_wavelength)
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
