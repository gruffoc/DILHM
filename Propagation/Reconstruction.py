#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 19 21:45:23 2019

@author: claudriel
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
from PIL import ImageStat
import holopy as hp
from __function import *
from holopy.core.process import bg_correct, subimage, normalize,center_find
import cv2
import time

if __name__ == "__main__":
    
    """
    Parametri da inserire:
       - indice di rifrazione del mezzo 
       - lunghezza d'onda del laser (um)
       - mezza lunghezza della immagine che si vuole tagliare (pixel) 
       - grandezza immagine (standard 1024)
       - directory della immagine raw da analizzare
       - distanze z a cui propagare 
       - directory a cui salvare i grafici e img oggetto
    """
    
    medium_index = 1.33
    pixel_size = 0.236
    illum_wavelen = 0.6328
    lim= 300
    Lx = 1024
    Ly = 1280
    name = "142"
    cartella = "Poly/2um/1"
    raw_holo = hp.load_image("../Campioni/Flusso/"+cartella+"/img_correct/img_" + name + ".tiff", spacing = pixel_size)  
    graph = "../Campioni/Flusso/"+cartella+"/propagation/img_n" + name + ".tiff"
    directory_obj = "../Campioni/Flusso/"+cartella+"/treshold/img_n" + name 
    directory_obj_dimension = "../Campioni/Flusso/"+cartella+"/treshold/img_n" + name + "_dimension.tiff"
    
    z = np.linspace(50,1000, 100)
    
    # plot dell'ologramma e calcolo del centro
    plt.figure(0)  
    hp.show(raw_holo)
    plt.show()
    plt.clf()

    centro = center_find(raw_holo, centers=1, threshold=0.3, blursize=6.0)
    print(centro)

    #controllo sulla grandezza dell'immagine, adatta lim in base se è prossima ai bordi
    if centro[0] < centro[1]:
        if centro[0] - lim < 0:
            lim = int(centro[0])
        if centro[1] + lim > Ly:
            lim = int(Ly - centro[1])
    else:
        if centro[1] - lim < 0:
            lim = int(centro[1])
        if centro[0] + lim > Lx:
            lim = int(Lx - centro[0])
    print (lim)
   
    #propagazione in z in una immagine ristretta da lim
    data_holo = raw_holo[:, int(centro[0]-lim) : int(centro[0]+lim), int(centro[1]-lim) : int(centro[1]+lim)]    
    rec_vol = hp.propagate(data_holo, z, illum_wavelen = illum_wavelen, medium_index = medium_index)
    
    #calcolo posizione di fuoco sia con ampiexxa del campo che con la fase
    modulo = propagation_module(z, rec_vol, lim)
    
    phase = np.angle(rec_vol)
    onda_riferimento = np.angle(np.e**((-1j*2*np.pi*z/(illum_wavelen/medium_index)))) 
    fase = propagation_phase(phase, onda_riferimento, z, lim)
    
    fase = np.nan_to_num(fase)
    max_d, min_d, max_zarray, min_zarray = maximum_minimum(fase, z)
    print("punto massimo (d):", max_d, ", punto minimo (d):", min_d)
    print("fuoco nell'array, punto di max:", max_zarray, ", punto di min:", min_zarray)
    
    fuoco = max_zarray[0]
  #  fuoco = min_zarray[0]
    print(fuoco)
#    fuoco =  int(min(i for i in fase if i > 0))
    plot_twin_propagation(z, modulo, fase, graph)
    
    #Riscalo e rifaccio tutto più centrato attrorno al fuoco#
    #########################################################
    
    z = np.linspace(max_d[0] -50, max_d[0] + 60, 100)
    rec_vol = hp.propagate(data_holo, z, illum_wavelen = illum_wavelen, medium_index = medium_index)
    modulo = propagation_module(z, rec_vol, lim)

    phase = np.angle(rec_vol)
    onda_riferimento = np.angle(np.e**((-1j*2*np.pi*z/(illum_wavelen/medium_index)))) 
    fase = propagation_phase(phase, onda_riferimento, z, lim)
    
    fase = np.nan_to_num(fase)
    pmax_d, min_d, max_zarray, min_zarray = maximum_minimum(fase, z)
    print("punto massimo (d):", max_d, ", punto minimo (d):", min_d)
    print("fuoco nell'array, punto di max:", max_zarray, ", punto di min:", min_zarray)
    
 #   fuoco = max_zarray[0]
    fuoco = min_zarray[0]
    print(fuoco)
#    fuoco =  int(min(i for i in fase if i > 0))
    plot_twin_propagation(z, modulo, fase, graph)
    
    #################################################
    
   
    # plot dell'immagine nel punto di fuoco
    obj = treshold(fuoco, phase, onda_riferimento,lim)
    plt.figure(3)
    plt.figure(figsize=(10,10))
    plt.imshow(obj)
    plt.colorbar()
    plt.savefig(directory_obj)
    
    minimo = np.ones((lim*2,lim*2))*np.abs(np.amin(obj))
    obj = obj + minimo
    obj = obj/np.amax(obj)
    obj = obj*255
    obj = obj.astype(int)
    result = Image.fromarray((obj).astype('uint8'))
    result.save(directory_obj + ".tiff")
    plt.show()
    plt.clf()
    
    #plot delle dimensini dell'oggetto
    time.sleep(1)
    dimension, dimA, dimB, ratio = object_dimension(directory_obj+'.tiff', pixel_size, lim)
    plt.figure(4)
    plt.figure(figsize=(10,10))
    plt.imshow(dimension)
    plt.plot()
    result = Image.fromarray((dimension).astype('uint8')) 
    result.save(directory_obj_dimension)
    
    print("Asse maggiore e minore, e il loro rapporto:", dimA, dimB, ratio)
#    
#    
