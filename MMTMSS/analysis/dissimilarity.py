import numpy as np
from scipy import signal


def build_kernel(r):
    XY = np.arange(-r,r+1,1)
    X, Y = np.meshgrid(XY, XY,  indexing='ij')
    kernel = np.zeros(X.shape)
    for lag in [(-.5,-.5), (-.5,.5), (.5,.5), (.5,-.5)]:
        lkernel = (X+lag[0])**2 + (Y+lag[1])**2
        lkernel = (1- (lkernel/(r**2)))**2 * (lkernel<=(r)**2)
        kernel = kernel + lkernel
    kernel = kernel / np.sum(kernel)
    return(kernel[kernel.sum(1)>0][:,kernel.sum(1)>0])


distribution = sc.multivariate_lognormal_cascade(6,
                                       sigma1=.7,
                                       sigma2=.7,
                                       corr=.5)

ratio = distribution[:,:,0] / (distribution[:,:,0] + distribution[:,:,1]) * 100

plt.imshow(ratio, vmin=0, vmax=100, cmap="RdYlBu_r")

kernel=build_kernel(35)
smoothdistribution = np.stack([signal.convolve(distribution[:,:,0],kernel,method='direct', mode='same'),
                               signal.convolve(distribution[:,:,1],kernel,method='direct', mode='same')], axis=-1)


#calculer la somme 
#faire la selection
# faire le calcul


ratio = smooth[:,:,0] / (smooth[:,:,0] + smooth[:,:,1]) * 100




plt.imshow(ratio, vmin=0, vmax=100, cmap="RdYlBu_r")


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from matplotlib.colors import LogNorm

import MMTMSS.models.simple_cascade as sc












