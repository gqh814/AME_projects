import numpy as np
from numpy import random
from scipy.stats import norm
import estimation as est

name = 'Probit'

# global flag to make silly checks 
# disable to increase speed 
DOCHECKS = True 

def G(z): 
    return norm.cdf(z)

def q(theta, y, x): 
    return -loglikelihood(theta, y, x)

def loglikelihood(theta, y, x):

    # making sure inputs are as expected 
    if DOCHECKS: 
        assert np.isin(y, [0,1]).all(), f'y must be binary: found non-binary elements.'
        assert y.ndim == 1
        assert x.ndim == 2 
        N,K = x.shape 
        assert y.shape[0] == N
        assert theta.ndim == 1 
        assert theta.size == K 

    Gxb = G(x@theta)
    
    # we cannot take the log of 0.0
    Gxb = np.fmax(Gxb, 1e-8)    # truncate below at 0.00000001 
    Gxb = np.fmin(Gxb, 1.-1e-8) # truncate above at 0.99999999

    ll = (y==1)*np.log(Gxb) + (y==0)*np.log(1-Gxb)
    return ll


def starting_values(y,x): 
    beta = est_ols(y,x)
    print(f'OLS estimate of beta: {beta}')
    return beta * 2.5

def predict(theta, x): 
    # the "prediction" is just Pr(y=1|x)
    yhat = G(x@theta)
    return yhat 

def Ginv(p): 
    '''Inverse cdf, taking arguments in (0;1) and returning numbers in (-inf;inf)
    '''
    return norm.ppf(p)

def sim_data(theta: np.ndarray, N:int) -> tuple: 
    '''sim_data: simulate a dataset of size N with true K-parameter theta

    Args. 
        theta: (K,) vector of true parameters (k=0 will always be a constant)
        N (int): number of observations to simulate 
    
    Returns
        tuple: y,x
            y (float): binary outcome taking values 0.0 and 1.0
            x: (N,K) matrix of explanatory variables
    '''

    # 0. unpack parameters from theta
    # (simple, we are only estimating beta coefficients)
    beta = theta

    K = theta.size 
    assert K>1, f'Not implemented for constant-only'
    
    # 1. simulate x variables, adding a constant 
    oo = np.ones((N,1))
    xx = np.random.normal(size=(N,K-1))
    x = np.hstack([oo, xx]);
    
    # 2. simulate y values
    
    # 2.a draw error terms 
    uniforms = np.random.uniform(size=(N,))
    u = Ginv(uniforms)

    # 2.b compute latent index 
    ystar = x@beta + u
    
    # 2.b compute observed y (as a float)
    y = (ystar>=0).astype(float)

    # 3. return 
    return y, x


def est_ols( y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Estimates y on x by ordinary least squares, returns coefficents

    Args:
        >> y (np.ndarray): Dependent variable (Needs to have shape 2D shape)
        >> x (np.ndarray): Independent variable (Needs to have shape 2D shape)

    Returns:
        np.array: Estimated beta coefficients.
    """
    return np.linalg.inv(x.T@x)@(x.T@y)