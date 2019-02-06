"""
Created on Wed Nov 28 16:15:04 2018
@author: Claudia Ravasio

Programma per simulare plot gaussiani 2d con i beam waist ricavati dalla
simulazione "Beam_Waist_Simulation". I beam waist sono quelli calcolati prima della
prima lente, nel punto di fuoco,dopo la seconda lente.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from matplotlib import text
from scipy import signal
from scipy.stats import norm
from scipy.optimize import curve_fit

length = 300
res1 = 0.01
res2 = 0.0001
res3 = 0.01

extent=(-4.5,4.5,-4.5,4.5)

def mag(w,lamda,f,s):
    zr = np.pi*w**2/lamda
    return 1/(((1-s/f)**2+(zr/f)**2)**0.5)

def imageposition(w,lamda,f,s):
    zr = np.pi*w**2/lamda
    return 1/(1/f - 1/(s+zr**2/(s-f)))

#plot gaussiano 2d
def intensity(w,lamda,P,a,b,res):
    x = np.arange(a,b, res)
    return 2*P/(np.pi*w**2)**0.5*np.exp((-2*(x)**2 )/(w**2))
   
lamda = 632.8e-6
wi = 0.405
P=1#mW

lens1 = 110
f1 = 8.6

f2 = 100.0 
lens2 = lens1+f1+f2

m1 = mag(wi,lamda,f1,lens1)
w1 = m1*wi

im1 = imageposition(wi,lamda,f1,lens1)
im2 = -imageposition(wi*m1,lamda,f2,lens2-im1-lens1)

m2 = mag(w1,lamda,f2,lens2-im1-lens1)
w2 = m2*w1

"""
BEAM INIZIALE
"""
h1 = np.arange(-1,1, res1)
gauss=intensity(wi,lamda,P,-1,1,res1)

"""
BEAM NEL FUOCO
"""
h2 = np.arange(-0.01,0.01, res2)
gauss_after_lens=intensity(w1,lamda,P,-0.01,0.01,res2)
"""
BEAM DOPO LA SECONDA LENTE
"""
h3 = np.arange(-10,10, res3)
gauss_after_second_lens=intensity(w2,lamda,P,-10,10,res3)

plt.figure(1)
plt.tile('Beam before the SF')
plt.plot(h1,gauss, color="Red")
plt.xlabel("Contour Radius (mm)")
plt.ylabel("Percent Irradiance")
plt.show()
plt.clf()

plt.figure(2)
plt.tile('Beam in the focal point (pinhole position)')
plt.plot(h2,gauss_after_lens,color="Red")
plt.xlabel("Contour Radius (mm)")
plt.ylabel("Percent Irradiance")
plt.xticks(np.arange(-0.01, 0.015, step=0.005))
plt.show()
plt.clf()

plt.figure(3)
plt.tile('Beam after the collimating lens')
plt.plot(h3,gauss_after_second_lens, color="Red")
plt.xlabel("Contour Radius (mm)")
plt.ylabel("Percent Irradiance")
plt.show()

