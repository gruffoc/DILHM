"""
Created on Thu Feb 07 15:32:37 2019

@author: Claudia

Programma per calibrare la magnificazione dell'obiettivo di raccolta dopo la cella
tramite crosscorrellazione di speckle traslate di 20um ciascuna.
"""

import os
import matplotlib
matplotlib.use('Agg')
from pylab import *
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image
from numpy import *
from numpy.fft import fft2,ifft2, fftshift, ifftshift

def corr_spaz(t1,t2):
    t1=ifftshift(fft2(fftshift(t1)))
    t1=np.conjugate(t1)
    t2=ifftshift(fft2(fftshift(t2)))
    c=np.real(fftshift(ifft2(ifftshift(t1*t2))))
    return c

if __name__ == '__main__':

    """
    Cross correlazione a coppia di immagini
    """

#    dir_img_list=sorted(os.listdir("002/"))
#    pos_array=np.array([])
#    for i in range(1,20):
#
#        im1 = Image.open("002/"+dir_img_list[i]).convert("L")
#        im2 = Image.open("002/"+dir_img_list[i-1]).convert("L")
#    
#        I1  = np.asarray(im1)
#        I2  = np.asarray(im2)
#        
#        I1=I1-np.average(I1)
#        I2=I2-np.average(I2)
#        
#        c=corr_spaz(I1,I2)
#    
##        pos=np.where(c[512]==np.amax(c[512]))[0]
##        pos_array=np.append(pos_array,pos)
#
#        plt.imshow(c)
#        plt.colorbar()
#        plt.savefig("crosst"+str(i)+".png", format="png")
#        plt.clf()


#plt.plot(pos_array,'*')
    """
    Cross correlazione rispetto la prima
    """

    dir_img_list=sorted(os.listdir("002/"))
    pos_array=np.array([])
    for i in range(1,20):

        im1 = Image.open("002/"+dir_img_list[0]).convert("L")
        im2 = Image.open("002/"+dir_img_list[i]).convert("L")
    
        I1  = np.asarray(im1)
        I2  = np.asarray(im2)
        
        I1=I1-np.average(I1)
        I2=I2-np.average(I2)
        
        c=corr_spaz(I1,I2)
    
#        pos=np.where(c[512]==np.amax(c[512]))[0]
#        pos_array=np.append(pos_array,pos)

        plt.imshow(c)
        plt.colorbar()
        plt.savefig("cross0"+str(i)+".png", format="png")
        plt.clf()