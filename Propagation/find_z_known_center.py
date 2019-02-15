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

Centerx=[36,186,314,270,464,596,642,630,718,816,884,812,1100,1158]
Centery=[134,464,670,950,280,530,196,136,234,396,228,802,454,60]

"""
2)PROPAGATION
Individuare la posizione z sapendo  il centro della immagine
"""

z = np.linspace(20,100,80)

rec_vol = hp.propagate(holo, z, illum_wavelen = 0.6328, medium_index = medium_index)
print ('rec ok')
phase=np.angle(rec_vol)
print ('phase ok')

p=np.angle(np.e**((-1j*2*np.pi*z/(0.6328/medium_index)))) ###fo
print ('p ok')
p_arr=np.array([])
phase_arr=np.array([])
phase_only_arr=np.array([])
for i in range(0,len(z)):
   
#    phase=np.angle(rec_vol[i][Centery[0]][Centerx[0]])
#    p=np.angle(np.e**((-1j*2*np.pi*z[i]/(0.6328/medium_index)))) ###fo
    diff=phase[Centery[11],Centerx[11],i]-p[i]

    phase_arr=np.append(phase_arr,diff)
    phase_only_arr=np.append(phase_only_arr,phase[Centery[11],Centerx[11],i])
#    print(i)
    p_arr=np.append(p_arr,p[i])
    phase_arr[phase_arr>np.pi]=0
    phase_arr[phase_arr<-np.pi]=0   
    
    
    
    

#plt.plot(z,phase_arr,'-*')
#plt.plot(z,phase_only_arr,'-*',label='img')
#plt.plot(z,p_arr,'-*', label='piana')
#plt.legend()
#plt.ylim(-np.pi, np.pi)
#plt.xlabel('z($\mu$m)')
#plt.ylabel('$\phi$(U)')
#plt.show()
#plt.clf()
#
#



#
