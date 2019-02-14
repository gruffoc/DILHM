#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 10:45:48 2019

@author: claudiaravasio

z=17.0568561873, x=284, y=256
diameter: [ 0.66528118]
ray: [ 0.33264059]
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

"""
PHASE VS Z
"""

z = np.linspace(0, 30, 100)
N=512
rec_vol = hp.propagate(holo, z)
phase_arr=np.array([])
for i in range(0,len(z)):
    phase=np.angle(rec_vol[i][286][253])
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





"""
PHASE IMAGE SHOW
"""
#z=np.linspace(16,17.1,10)
#
#for i in z:    
#    rec_vol = hp.propagate(holo, i)
#    p=np.angle(np.e**(-1j*2*np.pi*i/(0.66/1.33)))
#    phase=np.angle(rec_vol)-p
#    phase[phase>np.pi]=0
#    phase[phase<-np.pi]=0
#    hp.show(phase)
#    plt.title(i)
#    plt.xlabel('x(pixel)')
#    plt.ylabel('y(pixel)')


"""
reconstruction
"""
#z=17.38693467
#rec_vol = hp.propagate(holo, z)
#p=np.angle(np.e**(-1j*2*np.pi*z/(0.66/1.33)))
#phase=np.angle(rec_vol)-p
#
#hp.show(phase)
#plt.xlim(100,400)
#plt.ylim(100,400)
#
#mask= (phase>3).astype(int)
#
#hp.show(mask)
#plt.xlim(100,400)
#plt.ylim(100,400)
#"""
#label sphere
#"""
#lw,num=measurements.label(mask)
#hp.show(lw)
#
#"""
#area and diameter
#"""
#area=(measurements.sum(mask,lw,range(num+1)))*0.0851*0.0851
#r=(area[1:]/np.pi)**0.5
#d=2*(area[1:]/np.pi)**0.5
#print ('diameter:',d)
#print ('ray:',r)
