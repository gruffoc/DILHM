#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 13:48:41 2019

@author: claudiaravasio
Programma che prende in input le immagini (gia` elaborate) e fa la propagazione
del suo campo. 

(a)Calculating the Fourier transform of H(X,Y)
(b) Simulating the angular spectrum propagator
(c) Multiplying the results of (a) and (b)
(d) Calculating the inverse Fourier transform of (c). The result provides t(x, y).

bg_correct(raw, bg, df=None)
Correct for noisy images by dividing by a background. The calculation used is (raw-df)/(bg-df).
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

  

def trans_func(d, med_wavelength, dx, N):
    
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


def propagate(data, d, med_wavelength, dx, N):
    
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

    G = trans_func(d, med_wavelength, dx, N)

    ft_holo=ifftshift(fft2(fftshift(data)))
    
    prop = ft_holo * G
    
    U=fftshift(ifft2(ifftshift(prop)))#/250
    
    return U

    #we may have lost coordinate values to floating point precision during fft/ifft

#    if contains_zero:
#        d = d_old
#        U = xr.concat([data, U], dim='z')


