#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 10:47:37 2019

@author: claudiaravasio

This program simulate two spere with holopy

"""
import sys
import os
import matplotlib
matplotlib.use('Agg')
from pylab import *
from scipy.ndimage import measurements
import numpy as np
import matplotlib.pyplot as plt

import holopy as hp
from holopy.scattering import calc_holo, Sphere, calc_field, calc_intensity, calc_cross_sections
from holopy.scattering import Spheres
from holopy.scattering import Cylinder

from holopy.core.io import get_example_data_path
imagepath = get_example_data_path('image0002.h5')
exp_img = hp.load(imagepath)

"""
Qui se vuoi fissare due posizioni specifiche
"""
sphere1=Sphere(center=( 20,20, 7), n=1.59, r=0.3)   #all in micro m units
sphere2=Sphere(center=(  25.6,25.6,7), n=1.59, r=0.3)
collection = Spheres([sphere1, sphere2])
medium_index=1.33
illum_wavelength=0.660
illum_polarization=(1,0)
detector=hp.detector_grid(shape=512,spacing=0.1)    #spacing=pix size

holo=calc_holo(detector,collection,medium_index,illum_wavelength,illum_polarization)

hp.show(holo)


"""
for look the trasversal profile of hologram centred in partice position 

plt.title('Hologram')
plt.plot(holo[256])
plt.xlabel("Pixel")
plt.ylabel("Intensity")
plt.title('Holo Intensity')
plt.xlim(100, 400)
"""


"""
Qui se vuoi vedere come varia all-avvicinarsi/allontanarsi di una delle
due sfere
"""
#sphere1=Sphere(center=( 20,20, 7), n=1.59, r=0.3)
#seconda_sfera=[20,22,23,24,24.2,24.35]
#
#for i in range(0,len(seconda_sfera)):
#    print (i)
#    sphere2 = Sphere(center=(25.2,seconda_sfera[i], 4), n = 1.59, r = 0.5)
#    collection = Spheres([sphere1, sphere2])
#    
#    medium_index=1.33
#    illum_wavelength=0.660
#    illum_polarization=(1,0)
#    detector=hp.detector_grid(shape=512,spacing=0.1)
#
#    holox=calc_holo(detector,collection,medium_index,illum_wavelength,illum_polarization)
#    hp.show(holox)
#    
#    """
#    for look the trasversal profile of hologram centred in partice position 
#    
#    plt.plot(holox[252])
#    plt.xlabel("Pixel")
#    plt.ylabel("Intensity")
#    plt.title('Holo Intensity')
#    plt.savefig("./resolution_lateral/holo_intensity"+str(i)+".png",format="png")
#    plt.clf()
#    """
    
#    
