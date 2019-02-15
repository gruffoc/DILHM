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
from PIL import Image

"""
1)LOAD IMAGE
"""


medium_index=1.29
raw_holo = hp.load_image('IMG_0020.tiff', spacing = 0.265, medium_index = medium_index, illum_wavelen = 0.6328)
bg = hp.load_image('bg.tif',  spacing = 0.265, medium_index = medium_index, illum_wavelen = 0.6328)
holo = raw_holo-bg
#hp.show(holo)


#884,228
#812,800
"""
2)PROPAGATION

Individuare fase giusta da imgshow
"""


z = np.linspace(0,70,20)

rec_vol = hp.propagate(holo, z, illum_wavelen = 0.6328, medium_index = medium_index)
phase=np.angle(rec_vol)

for i in range(0,len(z)):
    p=np.angle(np.e**(-1j*2*np.pi*z[i]/(0.6328/medium_index)))*np.ones((1024,1280))
    diff=phase[i]-p
    diff[diff>np.pi]=0
    diff[diff<-np.pi]=0
    if np.isin(diff.any(),diff>2.5):
        hp.show(diff)
        plt.title(z[i])
# 

