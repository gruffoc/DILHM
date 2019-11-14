import cv2
import numpy as np
import matplotlib.pylab as plt


N = 1024
im  = cv2.cvtColor(cv2.imread("poli2um.tiff"), cv2.COLOR_BGR2GRAY)[0:N,0:N]
dim = im.shape

onesvec = np.ones(N)
inds  = (np.arange(N)+.5 - N/2.) /(N-1.)

X = np.outer(onesvec, inds)
Y = np.transpose(X)

R  = np.sqrt(X**2. + Y**2.)
R2 = np.sqrt((X-0.25)**2 + (Y+0.25)**2.)

K = np.exp(-(R**2)/0.055)*np.cos(R/0.007)**2

KK = np.cos(R**2 / 0.0015)

signal = np.cos(R**2 / 0.0015) + np.cos(R2**2/0.0015 )



im_f = np.fft.fft2(np.fft.fftshift(im))
ss_f = np.fft.fft2(np.fft.fftshift(signal))
kk_f = np.fft.fft2(np.fft.fftshift(KK))



conv_f = im_f * kk_f

conv2_f = ss_f * kk_f


map_centers = np.real(np.fft.fftshift(np.fft.ifft2(conv_f)))
cent = np.real(np.fft.fftshift((np.fft.ifft2(conv2_f))))

#plt.imshow(np.log10(map_centers/np.max(map_centers)))
#plt.figure(2)
plt.imshow(cent, vmin=0, vmax=400000)
plt.colorbar()
plt.figure(2)
plt.imshow(signal, vmin=0, vmax=0.5)
plt.colorbar()

plt.show()
