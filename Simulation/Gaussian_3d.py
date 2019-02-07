"""
Created on Wed Nov 28 16:15:04 2018
@author: Claudia Ravasio

Programma per simulare plot gaussiani 3d con i beam waist ricavati dalla
simulazione "Beam_Waist_Simulation". I beam waist sono quelli calcolati prima della
prima lente, nel punto di fuoco,dopo la seconda lente.
"""
import numpy as np
import matplotlib.pyplot as plt

# Define initial setuplength and plot resolution
length = 300
res1 = 0.01
res2 = 0.0001
res3 = 0.01

extent1=(-1,1,-1,1)
extent2=(-0.01,0.01,-0.01,0.01)
extent3=(-10,10,-10,10)

def mag(w,lamda,f,s):
    zr = np.pi*w**2/lamda
    return 1/(((1-s/f)**2+(zr/f)**2)**0.5)

def imageposition(w,lamda,f,s):
    zr = np.pi*w**2/lamda
    return 1/(1/f - 1/(s+zr**2/(s-f)))

def intensity_spot(w,lamda,P,extent,res):
    x,y=np.meshgrid(np.arange(extent[0],extent[1],res), np.arange(extent[2], extent[3],res))
    return 2*P/(np.pi*w**2)*np.exp((-2*((x-0)**2+(y-0)**2)) /(w**2))

def int2d(z, cmap, extent): 
    fig= plt.imshow(z,cmap=cmap,extent=extent)
    plt.xlabel("x(mm)")
    plt.ylabel("y(mm)")
    plt.colorbar()
    return fig
    
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
z=intensity_spot(wi,lamda,P,extent1,res1)


"""
BEAM NEL FUOCO
"""
z1=intensity_spot(w1,lamda,P,extent2,res2)


"""
BEAM DOPO LA SECONDA LENTE
"""
z2=intensity_spot(w2,lamda,P,extent3,res3)

plt.figure(1)
plt.title('Beam before the SF')
fig1 = int2d(z, 'gray',extent1)

plt.figure(2)
plt.title('Beam in the focal point (pinhole position)')
fig2= int2d(z1, 'gray',extent2)
plt.xticks(np.arange(-0.01, 0.015, step=0.005))
plt.yticks(np.arange(-0.01, 0.015, step=0.005))

plt.figure(3)
plt.title('Beam after the collimating lens')
fig3= int2d(z2, 'gray',extent3)


