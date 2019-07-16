#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 09:47:47 2019

@author: claudriel
"""
import sys
import os
import numpy as np
import holopy as hp
import matplotlib.pyplot as plt
import holopy as hp
from holopy.scattering import calc_holo, Sphere
from PIL import Image

from __function import *

"""
This script writes on a file.dat x, y index and the relative intensity value of
an hologram image in three different columns. 
It opens the image, finds the center of the holohgram e resizes the image over
the hologram center with a dim 290x290.
It is possible also do the binning of the image.

Returns on the consoel: the raw image and the center, to control it.
It save the file.dat in the path where you have lunch the script
"""

########### Image setup   
name = "142"
cartella = "Poly/2um/1"  
name_file = "Poly2um_1_142.dat"   
path_file = "File_dat/" + name_file
medium_index = 1.33
pixel_size = 0.236
illum_wavelen = 0.6328
lim = 290
leng = lim*2

############################  
"""
For NO binning here
"""
#data_holo = calcolo_hologram(cartella, name, pixel_size, lim)
#
#
#dati = open(name_file,'w+')
#for i in range(0,leng):
#    for j in range(0,leng):
#        print ( str(str((i-lim))+" "+str((j-lim))+" "+str(data_holo[i][j].values)),file=dati)
#dati.close() 
    
############################
"""
For BINNING here
"""
binsize = lim
pixcombine = 'mean'
data_holo = calcolo_hologram_BINNING(cartella, name, pixel_size, lim, binsize,  pixcombine)
lim =int( lim/2)
leng = lim*2

f1=open(name_file,'w+')
for i in range(0,leng):
    for j in range(0,leng):
       print ( str(str((i-lim))+" "+str((j-lim))+" "+str(data_holo[i][j])),file=f1)
f1.close() 
    
