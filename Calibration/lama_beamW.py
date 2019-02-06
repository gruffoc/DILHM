"""
Created on Mon Dec 17 16:08:26 2018

@author: Claudia

Questo programma serve a verificare il beam waist del fascio nel punto voluto.
Sperimentalmente ci mettiamo dopo la lente collimante e prendiamo la figura dello spot
del fascio al variare della posizone trasversale di una lama (che quindi vedrà tutto nero
all'inizio perchè lo copre tutto, e viceversa dopo).
Dopodichè intregriamo i valori di intensità e fittiamo con una funzione cumulativa.
La doppia sigma corrisponde al valore di beam waist. 
Se si vuole si può aggiungere il plot della gaussiana con i valori trovati
"""

import sys
import os
import matplotlib
matplotlib.use('Agg')
from pylab import *
from scipy.ndimage import measurements
from PIL import Image
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import curve_fit
from matplotlib import text
import scipy.stats as st
from scipy.special import erf
from scipy import optimize


def intensity(w,lamda,P,a,b, mu1):
    x = np.arange(a,b, res)
    return 2*P/(np.pi*w**2)**0.5*np.exp((-2*(x-mu1)**2 )/(w**2))

if __name__ == '__main__':
    
    length = 300
    res = 0.01
       
    lamda = 632.8e-6
    P=1#mW  
    
    dir_img_list=sorted(os.listdir("lamaG/"))  #directory of the image

    Intensita = np.array([])
    for k in dir_img_list:
        file_path="lamaG"+"/"+k
        im = Image.open(file_path).convert("L")
        I  = np.asarray(im)
        sumI=np.sum(I)
        Intensita = np.append(Intensita,sumI)
    
    Intensita=Intensita/np.amax(Intensita)
    Intensita[np.isnan(Intensita)] = 0
    
    Intensita=1-Intensita
    x_array=np.arange(0,13.6,0.1)
    
    fg = plt.figure(1); fg.clf()
    plt.grid(True)
    plt.xlabel("z(mm)")
    plt.ylabel("Cumulative Probability Density")
    plt.title("Fit to Normal Distribution")
    plt.plot(x_array,Intensita,'ko',label='Data')
    
    mu1,sigma1 = curve_fit(norm.cdf, x_array, Intensita, p0=[0,1])[0]
    
    h = np.arange(0,13.6,res)
    gauss=intensity(2*sigma1,lamda,P,0,13.6, mu1)
    
    plt.plot(x_array, norm.cdf(x_array, mu1, sigma1), "r",label='Fitted function')
    
    
    plt.plot(h,gauss, color="Green",label='Probability Density Function')
    plt.legend(loc='best')


"""
Calcolo dei residui
"""

    #fg2=plt.figure(2)
    #plt.xlabel("z(mm)")
    #plt.ylabel("Residue")
    ##plt.title("Fit to Normal Distribution")
    #residui=Intensita-norm.cdf(x_array,mu1,sigma1)
    #plt.plot(x_array,residui, '-ko')
    #a=range(0,14)
    #x=range(0,14)
    #y=1*np.ones(len(a))

    #media_residui=np.average(residui)
    #plt.plot(x,y,'-r')
    #plt.plot(x,-y,'-r')
    
    
