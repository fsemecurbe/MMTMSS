import numpy as np
rng = np.random.default_rng()


def multivariate_cascade(p,q, n):
    """
    Compute a multivariate dyadic cascade based on two probability generators (2*2).
    

    Parameters
    ----------
    p : Numpy Array of dimension 2*2, sum(P)=1
        It represents the generator of the deprived population 
    q : Numpy Array of dimension 2*2, sum(Q)=1
        It represents the generator of the well off population
    n : integer
        Number of steps (scale) of the dyadic cascade.

    Returns
    -------
    An Numpy array (2**n,2**n, 2)

    """
    P = p.copy()
    Q = q.copy()
    for i in range(n):
        P = np.kron(P, p)
        Q = np.kron(Q, q)
    return(np.stack([P,Q], axis=-1))



def multivariate_lognormal_cascade(n, sigma1=1, sigma2=1, corr=0):    
    exp1 = -1/2 * sigma1**2
    exp2 = -1/2 * sigma2**2
    
    PQ = np.exp(rng.multivariate_normal(np.array([exp1,exp2]),  np.array([[sigma1**2,corr*sigma1*sigma2], [corr*sigma1*sigma2, sigma2**2]]), 4))
    
    P = PQ[:,0].reshape(2,2)
    Q = PQ[:,1].reshape(2,2)
    for i in range(n):
        PQ = np.exp(rng.multivariate_normal(np.array([exp1,exp2]),np.array([[sigma1**2,corr*sigma1*sigma2], [corr*sigma1*sigma2, sigma2**2]]), P.shape[0]**2 * 4))
        P = np.kron(P, np.ones((2,2))) 
        P = P * PQ[:,0].reshape(P.shape)
        Q = np.kron(Q, np.ones((2,2))) 
        Q = Q * PQ[:,1].reshape(P.shape)
    
    P = P / np.sum(P)
    Q = Q / np.sum(Q)
    return(np.stack([P,Q], axis=-1))
