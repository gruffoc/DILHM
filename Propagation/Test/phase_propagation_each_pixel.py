#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 11:07:28 2019

@author: claudiaravasio

PHASE VS Z in qualsiasi punto dell'immagine.
Il programma cicla suogni x/y pixel dell'immagine e calcola la fase nel punto
in funzione di z. L'output e` un grafico z vs rad (phase) solo delle curve che
presentano una transizione di fase. Cosi` trovi x,y,z del punto di fuoco.
Questo metodo e` decisamente lento computazionalmente e spero di trovare
qualcosa di meglio a breve. 

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


z = np.linspace(0,20, 200)
rec_vol = hp.propagate(holo, z)

x_arr=np.array([])   
y_arr=np.array([])  
for x in range(278,292,1):   
    print (x)
    for y in range(248,262,1):   
        
        phase_arr=np.array([])
        for i in range(0,len(z)):
            
            phase=np.angle(rec_vol[i][x][y])
            p=np.angle(np.e**(-1j*2*np.pi*z[i]/(0.66/1.33)))
            diff=phase-p
            if diff<-np.pi:
                diff=0
            if diff>np.pi:
                diff=0
            phase_arr=np.append(phase_arr,diff)
        z_max=np.where(phase_arr==np.amax(phase_arr))[0]
        if np.amax(phase_arr)>2:
            print ('max',y,np.amax(phase_arr),z[z_max])
#            label=label=str(x)+','+str(y)
            x_arr=np.append(x_arr,x)
            y_arr=np.append(y_arr,y)
            grapo=plt.plot(z,phase_arr, '*')
            plt.ylim(-np.pi, np.pi)
            plt.xlabel('z($\mu$m)')
            plt.ylabel('$\phi$(U)')
#            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()
plt.clf()

plt.plot(x_arr,y_arr,'o')
plt.show()
plt.clf()
