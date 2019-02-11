#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 10:46:03 2019
@author: claudiaravasio

Phase and module on the transversal plane at three different distance from
the focal point. The image analysed is one of the holopy examples.

"""

import matplotlib.pyplot as plt
import numpy as np
import holopy as hp
from holopy.core.io import get_example_data_path, load_average
from holopy.core.process import bg_correct
from scipy.ndimage import measurements
from holopy.scattering import Spheres
from holopy.scattering import calc_holo, Sphere


imagepath = get_example_data_path('image01.jpg')
raw_holo = hp.load_image(imagepath, spacing = 0.0851, medium_index = 1.33, illum_wavelen = 0.66, )
bgpath = get_example_data_path(['bg01.jpg','bg02.jpg','bg03.jpg'])
bg = load_average(bgpath, refimg = raw_holo)
holo = bg_correct(raw_holo, bg)

z=[17.5,20,27]
x=np.linspace(1,512,1000)

p1=np.angle(np.e**(-1j*2*np.pi*z[0]/(0.66/1.33)))
p2=np.angle(np.e**(-1j*2*np.pi*z[1]/(0.66/1.33)))
p3=np.angle(np.e**(-1j*2*np.pi*z[2]/(0.66/1.33)))
 
rec_vol1 = hp.propagate(holo, z[0])
rec_vol2 = hp.propagate(holo, z[1])
rec_vol3 = hp.propagate(holo, z[2])

amp1=np.abs(rec_vol1[:,253,:])
amp2=np.abs(rec_vol2[:,253,:])
amp3=np.abs(rec_vol3[:,253,:])

phase1=np.angle(rec_vol1[:,253,:])
phase2=np.angle(rec_vol2[:,253,:])
phase3=np.angle(rec_vol3[:,253,:])

plt.figure(1)
plt.plot(amp1, label=z[0])
plt.plot(amp2,label=z[1])
plt.plot(amp3,label=z[2])

plt.title("Module")
plt.xlabel('x(pixel)')
plt.ylabel('|U|')
plt.xlim(220,350)
plt.legend()
plt.show()

plt.figure(2)
plt.plot(phase1-p1,label=z[0])
plt.plot(phase2-p2,label=z[1])
plt.plot(phase3-p3,label=z[2])

plt.title("Phase")
plt.xlabel('x(pixel)')
plt.ylabel('|U|')
plt.xlim(220,350)
plt.ylim(-1,3)
plt.legend()
plt.show()






