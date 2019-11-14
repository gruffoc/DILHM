import cv2
import numpy as np
import matplotlib.pylab as plt

def filter_deconv(N, img, pix_size):
    onesvec = np.ones(N)
    inds  = (np.arange(N)+.5 - N/2.) /(N-1.)
    X = np.outer(onesvec, inds)
    Y = np.transpose(X)
    R  = np.sqrt(X**2. + Y**2.) # * pix_size

    K = np.exp(-(R**2)/0.0109)*(np.cos(R**2/0.00109)) + 2
    #K = 1/(R**2.)

    im_f = np.fft.fft2(np.fft.fftshift(img))
    kk_f = np.fft.fft2(np.fft.fftshift(K))

    conv_f = im_f * kk_f

    map_centers = np.abs(np.fft.fftshift(np.fft.ifft2(conv_f)))**2

    return map_centers

N = 1024
im  = cv2.cvtColor(cv2.imread("poli2um.tiff"), cv2.COLOR_BGR2GRAY)[0:N,0+256:N+256]

centri = filter_deconv(N, im, 0.236) # pix_size is given in um

plt.imshow(np.log10(centri/np.std(centri)))
plt.figure(2)
plt.imshow(im)
plt.show()


# holo[ R < 0.1]
