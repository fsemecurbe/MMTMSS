import numpy as np
from scipy import signal
import pandas as pd


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

def dissimilarity(distribution, radius):
    
    kernel = build_kernel(radius)
    smoothdistribution = np.stack([signal.convolve(distribution[:,:,0],kernel,method='direct', mode='same'),
                               signal.convolve(distribution[:,:,1],kernel,method='direct', mode='same')], axis=-1)

    wherenontnull = np.sum(distribution, axis=2)>0

    distributionnonnull = distribution[wherenontnull,:]
    smoothdistribution = smoothdistribution[wherenontnull,:]


    P0local= smoothdistribution[:,0] / (smoothdistribution[:,0] + smoothdistribution[:,1])
    P1local= smoothdistribution[:,1] / (smoothdistribution[:,0] + smoothdistribution[:,1])

    Entropie_locale = np.nansum((- np.log2(P0local) * P0local - np.log2(P1local) * P1local) *  np.sum(distributionnonnull, axis=1)/ np.sum(distributionnonnull) )

    P0 = np.sum(distributionnonnull[:,0])
    P1 = np.sum(distributionnonnull[:,1])
    
    Entropie_globale = -np.log2(P0/(P0+P1))*P0/(P0+P1) - np.log2(P1/(P0+P1))*P1/(P0+P1)   
    return((Entropie_globale - Entropie_locale) /Entropie_globale)

def dissimilarity_analyze(distribution, resolutions):
    res = [dissimilarity(distribution, resolution) for resolution in resolutions]
    return(pd.DataFrame({'resolution' : resolutions, 'dindex' : res}))











