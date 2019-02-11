#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 16:57:40 2019

@author: claudiaravasio

PHASE VS Z in qualsiasi punto dell'immagine new methods singola img
"""

import matplotlib.pyplot as plt
import numpy as np
import holopy as hp
from holopy.core.io import get_example_data_path, load_average
from holopy.core.process import bg_correct
from scipy.ndimage import measurements
from holopy.scattering import Spheres
from holopy.scattering import calc_holo, Sphere

"""
1)LOAD IMAGE
"""

"""
IMMAGINE DI ESEMPIO, HOLOPY SINGLE SPHERE
"""

imagepath = get_example_data_path('image01.jpg')
raw_holo = hp.load_image(imagepath, spacing = 0.0851, medium_index = 1.33, illum_wavelen = 0.66, )
bgpath = get_example_data_path(['bg01.jpg','bg02.jpg','bg03.jpg'])
bg = load_average(bgpath, refimg = raw_holo)
holo = bg_correct(raw_holo, bg)

"""
IMMAGINE DI ESEMPIO, HOLOPY DOUBLE SPHERES
(simulata e salvata)
"""

#medium_index = 1.33
#illum_wavelen = 0.66
#illum_polarization = (1,0)
#detector = hp.detector_grid(shape = 512, spacing = 0.0851)
holo = hp.load_image('outfilename8febb.tif', spacing = 0.0851, medium_index = 1.33, illum_wavelen = 0.66, )


"""
2)PROPAGATION
"""

N=512
z = np.linspace(0,20,300)
rec_vol = hp.propagate(holo, z)
phase=np.angle(rec_vol)

for i in range(0,len(z)):
    p=np.angle(np.e**(-1j*2*np.pi*z[i]/(0.66/1.33)))*np.ones((N,N))
    diff=phase[i]-p
    diff[diff>np.pi]=0
    diff[diff<-np.pi]=0
    if np.isin(diff.any(),diff>2.5):
        hp.show(diff)
        plt.title(z[i])
 





