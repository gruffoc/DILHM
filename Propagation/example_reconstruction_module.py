#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 09:29:39 2019

@author: claudiaravasio

Programma per ottenere la ricostruzione di una particella attraverso lo studio
del modulo. L-immagine esempio e` presa da holopy, la x, y quindi si sanno
a priori.


Simulation of the module
z=17.5, x=286, y=253
diameter: [ 49.14823348]
ray: [ 24.57411674]
"""
import matplotlib.pyplot as plt
import numpy as np
import holopy as hp
from holopy.core.io import get_example_data_path, load_average
from holopy.core.process import bg_correct
from scipy.ndimage import measurements


imagepath = get_example_data_path('image01.jpg')
raw_holo = hp.load_image(imagepath, spacing = 0.0851, medium_index = 1.33, illum_wavelen = 0.66 )
bgpath = get_example_data_path(['bg01.jpg','bg02.jpg','bg03.jpg'])
bg = load_average(bgpath, refimg = raw_holo)
holo = bg_correct(raw_holo, bg)

"""
MODULE VS Z
"""
z = np.linspace(0, 40, 400)

rec_vol = hp.propagate(holo, z)

module_arr=np.array([])
for i in range(0,len(z)):
    
    module=np.abs(rec_vol[i][286][253])
    
    module_arr=np.append(module_arr,module)
    
plt.plot(z,module_arr,'*')
plt.xlabel('z($\mu$m)')
plt.ylabel('|U|')
plt.show()





"""
modulo IMAGE SHOW
"""

#z=np.linspace(17,18,10)
#
#for i in z:    
#    rec_vol = hp.propagate(holo, i)
#    amp=np.abs(rec_vol)
#    hp.show(amp)
#    plt.title(i)
#    plt.xlabel('x(pixel)')
#    plt.ylabel('y(pixel)')
#    
"""
reconstruction
"""
#z=17.5
#rec_vol = hp.propagate(holo, z)
#amp=np.abs(rec_vol)
#
#hp.show(amp)
#
#mask= (amp>0.9).astype(int)
#hp.show(mask)

"""
label sphere
"""
#lw,num=measurements.label(mask)
#
"""
area and diameter
"""
#area=(measurements.sum(mask,lw,range(num+1)))*0.0851*0.0851
#r=(area[1:]/np.pi)**0.5
#d=2*(area[1:]/np.pi)**0.5
#print ('diameter:',d)
#print ('ray:',r)
