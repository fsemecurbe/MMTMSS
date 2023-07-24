import numpy as np
import pandas as pd

def downsizing_sum(frac, factor):    
    """
    downsizing_sum downsizes a numpy array. It's the first step to perform box
    counting analysis.
    
    Parameters
    ----------
    frac : Numpy Array 
        
    factor : Integer 
        factor of downizing the matrix 

    Returns
    -------
    Nunpy Array

    """    
    if (frac.shape[0] % factor)>0:
        padx= -frac.shape[0] + (frac.shape[0] // factor +1)*factor 
        frac =  np.pad(frac, ((0, padx), (0, 0), (0,0))) 
    if (frac.shape[1] % factor)>0:
        pady= -frac.shape[1] + (frac.shape[1] // factor +1)*factor 
        frac =  np.pad(frac, ((0, 0), (0, pady), (0,0))) 
    
    factors = np.array([factor,factor,1])
    sh = np.column_stack([frac.shape//factors, factors]).ravel()
    return(frac.reshape(sh).sum(tuple(range(1, 2*frac.ndim, 2))))


def dissimilarity_index(frac):
    """
    Perform a simplified version of the dissimilarity index (Reardon et al. 2004).
    The analysis can be computed on downsizing distribution.

    Parameters
    ----------
    frac : Numpy Array
        The Numpy must have two layers. Each layer represents a spatial distribution of population

    Returns
    -------
    None.

    """
    ptot = np.sum(frac, axis=2)
    Prob = frac / ptot[:, :, np.newaxis]
    
    P0 = np.sum(frac[:,:,0]) / np.sum(frac[:,:,0] +  frac[:,:,1])
    P1 = np.sum(frac[:,:,1]) / np.sum(frac[:,:,0] +  frac[:,:,1])
    Entropie_globale = -np.log2(P0)*P0 -np.log2(P1)*P1   
    return((Entropie_globale + np.sum(np.sum(np.log2(Prob) * Prob, axis=2) * ptot) / np.sum(ptot) )/Entropie_globale )

def multifractal_index(frac):
    """
    Compute multiscale index based on multifractal theory.

    Parameters
    ----------
    frac : Numpy arrau
        DESCRIPTION.

    Returns
    -------
    None.

    """
    frac[:,:,0] = frac[:,:,0] / np.sum(frac[:,:,0])
    frac[:,:,1] = frac[:,:,1] / np.sum(frac[:,:,1])
    P = frac[:,:,0] * frac[:,:,1]
    P = P / np.sum(P)
    return((-np.sum(frac[:,:,0]*np.log2(frac[:,:,0])), -np.sum(frac[:,:,1]*np.log2(frac[:,:,1])) ,  -np.sum(P*np.log2(P)),-np.sum(frac[:,:,0]*np.log2(frac[:,:,1]/frac[:,:,0])) ))

