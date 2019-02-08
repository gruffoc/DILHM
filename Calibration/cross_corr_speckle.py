"""
Created on Thu Feb 07 15:32:37 2019
@author: Claudia Ravasio

Programma per calibrare la magnificazione dell'obiettivo di raccolta dopo la cella
tramite crosscorrellazione di speckle traslate di 20um ciascuna.

RISULTATI:
1)cross uno a uno:
0.293869798836---->18.17X, con maschera:  0.258638798998--->20.65X

2)cross uno ogni due:
0.270772990019--->19.72X, con maschera -->0.267283423664--->19.98X

3)cross uno ogni tre:
0.266427106821--->20X

media 0.277--->19.3X
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
    x=np.arange(1,1281)
    centro=len(x)/2
    dir_img_list=sorted(os.listdir("calibrazione8febbraio/"))
    pos_array=np.array([])
    for i in range(1,50):
        print i
        im1 = Image.open("calibrazione8febbraio/"+dir_img_list[i]).convert("L")
        im2 = Image.open("calibrazione8febbraio/"+dir_img_list[i-1]).convert("L")
    
        I1  = np.asarray(im1)
        I2  = np.asarray(im2)
        
        I1=I1-np.average(I1)
        I2=I2-np.average(I2)
        
        c=corr_spaz(I1,I2)
        """
        Per plottare i picchi..per ora non so come farlo automaticamente
        a=[515,525,503,513,531,512,495,518,511,519,496,510,527,501,514,506,513,503,505,512,516,512,516,510,516,516,503,517,514,505,513,520,511,512,506,526,501,513,521,512,502,507,504,505,527,500,523,503,530]
        plt.plot(x,c[a[i-1]])
        """
        pos=np.where(c==np.amax(c))[1]
        posi=20./np.abs((pos-centro))
        pos_array=np.append(pos_array,posi)
       
        print i, pos,posi
        
        """
        Per plottare le cross come imshow
        plt.imshow(c)
        plt.colorbar()
        plt.savefig("plot_profile"+str(i)+".png", format="png")
#       plt.clf()

        """
    """
    Per Mettere maschera
    pos_array=pos_array[(pos_array<0.4)]
    pos_array=pos_array[(pos_array>0.134)]
    """
    pix=np.average(pos_array)
    
#    plt.show()
    print pix
    

#    """
#    Cross correlazione rispetto la prima
#    """
#    x=np.arange(1,1281)
#    centro=len(x)/2
#    dir_img_list=sorted(os.listdir("calibrazione8febbraio/"))
#    pos_array=np.array([])
#    for i in range(1,15):
#        print(i)
#        im1 = Image.open("calibrazione8febbraio/"+dir_img_list[0]).convert("L")
#        im2 = Image.open("calibrazione8febbraio/"+dir_img_list[i]).convert("L")
#    
#        I1  = np.asarray(im1)
#        I2  = np.asarray(im2)
#        
#        I1=I1-np.average(I1)
#        I2=I2-np.average(I2)
#        
#        c=corr_spaz(I1,I2)
#        
#        pos=np.where(c==np.amax(c))[0]
#
#        posi=20./np.abs((pos-centro))
#        pos_array=np.append(pos_array,posi)
###       
##        print i, pos
##        plt.imshow(c)
##        plt.colorbar()
##        plt.savefig("mored"+str(i)+".png", format="png")
##        plt.clf()
##○        plt.plot(c[pos_array])
##    pos_array=pos_array[(pos_array<0.4)]
##    pos_array=pos_array[(pos_array>0.134)]
#    dist=np.linspace(1,14,14)
#    pos_array=pos_array*dist
#    pix=np.average(pos_array)
##    
###    plt.show()
#    print pix
#      