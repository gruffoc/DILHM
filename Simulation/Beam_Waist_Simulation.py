"""
Created on Wed Nov 28 16:15:04 2018
@author: Claudia Ravasio

Questo programma simula il beam waist nel mio set up strumentale: il wi iniziale 
è quello tabulato (calcolato dalla divergenza). 
A seguire ho un filtraggio spaziale composto da due lenti e un pinhole(posto esattamente
nel fuoco delle due lenti). La distanza delle due lenti è pari alle due focali.
L'ultima lente collima il fascio.

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from matplotlib import text
import scipy.stats as st

# Define initial setuplength and plot resolution
length = 300
res = 0.1

# Define plots and range
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.50)

# beam waist function 
def beam(w,lamda,a,b):
    x = np.arange(a,b, res)
    return w*(1+ (lamda*(x-a)/(np.pi*w**2))**2)**0.5
def beamc(w,lamda,a,b):
    x = 55
    return w*(1+ (lamda*(x-a)/(np.pi*w**2))**2)**0.5

# Plot the back trace of the beam
def backbeam(m,w,lamda,a,b):
    zr = np.pi*(w*m)**2/lamda
    x = np.arange(a,b, res)
    return m*w*(1+ ((b-x)/zr)**2)**0.5
# Compute the waist position
def imageposition(w,lamda,f,s):
    zr = np.pi*w**2/lamda
    return 1/(1/f - 1/(s+zr**2/(s-f)))
# Compute the magnification
def mag(w,lamda,f,s):
    zr = np.pi*w**2/lamda
    return 1/(((1-s/f)**2+(zr/f)**2)**0.5)

def intensity(w,lamda,r):
    #x = np.arange(a,b, res)
    return 2*lamda**2(r)/(np.pi*w**2)*np.exp(-2*(r**2 )/(w**2))


lamda = 632.8e-6
rad=0.5e-3 #Rad
wi=lamda/(np.pi*rad)
#wi = 0.405

# initial beam and lens position in mm
lens1 = 110 #mm
f1 = 8.6
m1 = mag(wi,lamda,f1,lens1)  #magnificazione lente 1

f2 = 100.0 
lens2 = lens1+f1+f2

# Compute Initial magnification and beamwaist position
im1 = imageposition(wi,lamda,f1,lens1)
im2 = -imageposition(wi*m1,lamda,f2,lens2-im1-lens1)

w1 = m1*wi

# If you want a specific value of bw in a specific position
c=beamc(w1,lamda,lens1+im1,lens2)

m2 = mag(w1,lamda,f2,lens2-im1-lens1)
w2 = m2*w1

w_focal=backbeam(m1,wi,lamda,lens1,lens1+im1)
w_focal_2=w_focal[len(w_focal)-1]*2

print "0)Beam iniziale:",wi
print "1)Beam dopo la prima lente:",w1
print "3)Beam dopo la seconda lente:",w2
print "4)Magnificazione lente 1 e 2:",m1,m2
print "5)The double beam waist in the focal point is:",w_focal_2*1e3, "um"
print "5)The pinhole (in the focal point) is: 30 um"
    
#1 before obj lens
beam1down ,= plt.plot(np.arange(0,lens1,res),-beam(wi,lamda,0,lens1),color = "Blue")
beam1up ,= plt.plot(np.arange(0,lens1,res),beam(wi,lamda,0.0,lens1),color = "Blue")

# focalized by first obj lens
bbbeam1down ,= plt.plot(np.arange(lens1,lens1+im1,res),
                       -backbeam(m1,wi,lamda,lens1,lens1+im1),color = "Blue")
bbbeam1up ,= plt.plot(np.arange(lens1,lens1+im1,res),
                     backbeam(m1,wi,lamda,lens1,lens1+im1),color = "Blue")

#After focalized
beam2down ,= plt.plot(np.arange(lens1+im1,lens2,res),
                      -beam(w1,lamda,lens1+im1,lens2),color = "Blue")
beam2up ,= plt.plot(np.arange(lens1+im1,lens2,res),
                    beam(w1,lamda,lens1+im1,lens2),color = "Blue")

#After the second lens
beam3down ,= plt.plot(np.arange(lens2,length,res),
                      -beam(w2,lamda,lens2,length),color = "Blue")
beam3up ,= plt.plot(np.arange(lens2,length,res),
                      beam(w2,lamda,lens2,length),color = "Blue")
##
#bbeam2down ,= plt.plot(np.arange(lens2,lens2+im2,res),
#                       -backbeam(m2,w1,lamda,lens2,lens2+im2),color = "Blue")
#
#bbeam2up ,= plt.plot(np.arange(lens2,lens2+im2,res),
#                       backbeam(m2,w1,lamda,lens2,lens2+im2),color = "Blue")

#im1 ,= plt.plot([lens1+im1,lens1+im1], [-w1,w1], color="Red")
#im2 ,= plt.plot([lens2+im2,lens2+im2], [-w2,w2])

#insert in the plot the lens
lens1 ,= plt.plot([lens1,lens1],[-1,1], color="Red")
lens2 ,= plt.plot([lens2,lens2],[-6,6],color="Red")

plt.axis([0, length, -7, 7])

ax.xaxis.set_ticks(np.arange(0, length, 50))
ax.yaxis.set_ticks(np.arange(-6, 8, 2))


"""
tabella dei valori sotto al grafico
"""
#ax.tick_params(labeltop=True, labelright=True)
#

#
# Define wavelength Slider 
#axlamda  = plt.axes([0.25, 0.35, 0.65, 0.03], axisbg=axcolor)
#slamda = Slider(axlamda, 'wavelent', 200, 1200, valinit=632.8)
#
## Define lens position slider
#axpos1  = plt.axes([0.25, 0.25, 0.65, 0.03], axisbg=axcolor)
#axpos2 = plt.axes([0.25, 0.15, 0.65, 0.03], axisbg=axcolor)
#
#spos1 = Slider(axpos1, 'position1', 0.0, length, valinit=pos1ini)
#spos2 = Slider(axpos2, 'position2', 0.0, length, valinit=pos2ini)
#
## Define initial beam wasit slider
#axw0 = plt.axes([0.25, 0.3, 0.65, 0.03], axisbg=axcolor)
#sw0 = Slider(axw0, 'beam waist', 0.0, 2.0, valinit=wini)
#
## Define lens1 focus slider 
#axf1 = plt.axes([0.25, 0.2, 0.65, 0.03], axisbg=axcolor)
#sf1 = Slider(axf1, 'lens 1 focus', 0.0, 300, valinit=f1ini)
#
## Define lens2 foucs slider 
#axf2 = plt.axes([0.25, 0.1, 0.65, 0.03], axisbg=axcolor)
#sf2 = Slider(axf2, 'lens 2 focus', 0.0, 300, valinit=f2ini)


"""
Beam waist in img1+lens1 ingrandito
"""
#beam1d,= plt.plot(np.arange(0,lens1,res),-beam(wi,lamda,0,lens1),color = "Blue")
#beam1u= plt.plot(np.arange(0,lens1,res),beam(wi,lamda,0.0,lens1),color = "Blue")
#
#bbbeam1d ,= plt.plot(np.arange(lens1,lens1+im1,res),
#                      -backbeam(m1,wi,lamda,lens1,lens1+im1),color = "Blue")
#bbbeam1u ,= plt.plot(np.arange(lens1,lens1+im1,res),
#                     backbeam(m1,wi,lamda,lens1,lens1+im1),color = "Blue")
#
#beam2d= plt.plot(np.arange(lens1+im1,lens2,res),
#                      -beam(w1,lamda,lens1+im1,lens2),color = "Blue")
#beam2u,= plt.plot(np.arange(lens1+im1,lens2,res),
#                    beam(w1,lamda,lens1+im1,lens2),color = "Blue")
#
#
#
#im1g= plt.plot([lens1+im1,lens1+im1], [-w1,w1], color="Red")
#
#b=range(109,128)
#rt=np.ones(len(b))*w1
#plt.plot(b,rt, color="Red")
#plt.plot(b,-rt, color="Red")
#
#plt.axis([0, length, -0.5, 0.5])