import numpy as np 
from scipy.stats import norm
from numpy import linalg as la

name = 'Tobit'

def q(theta, y, x): 
    return -loglikelihood(theta, y, x)

def loglikelihood(theta, y, x): 
    assert y.ndim == 1, f'y should be 1-dimensional'
    assert theta.ndim == 1, f'theta should be 1-dimensional'

    # unpack parameters 
    b = theta[:-1] # first K parameters are betas, the last is sigma 
    mu = b[-1] # the last beta is mu
    b = b[:-1].reshape(-1,1) # the last beta is mu
    sig = np.abs(theta[-1]) # take abs() to ensure positivity (in case the optimizer decides to try negatives)
    
    phi = norm().pdf((y - (x@b).reshape(-1,) - mu)/sig)


    Phi = norm().cdf(((x@b).reshape(-1,)+mu)/sig)
    Phi = np.clip(Phi, 1e-8, 1.-1e-8)

    ll =  (y<0.)*np.log(phi * 1/sig) + (y==0.)*np.log(Phi) # get indicator functions by using (y>0) and (y==0)
    
    # print(phi.shape, Phi.shape, (x@b).reshape(-1,).shape, y.shape)
    return ll


def starting_values(y,x): 
    '''starting_values
    Returns
        theta: K+1 array, where theta[:K] are betas, and theta[-1] is sigma (not squared)
    '''
    print("HOOL")
    N,K = x.shape
    b_ols = la.inv(x.T@x) @ (x.T@y) 
    res = y - x@b_ols
    sig2hat = 1/(N-K) * res.T@res
    sighat = np.sqrt(sig2hat) # our convention is that we estimate sigma, not sigma squared
    theta0 = np.append(b_ols, sighat)
    return theta0 

# def mills_ratio(z): 
#     return norm.pdf(z) / norm.cdf(z)

def predict(theta, x): 
    '''predict(): the expected value of y given x 
    Returns E, E_pos
        E: E(y|x)
    '''
    beta = theta[:-2]
    mu = theta[1]
    sigma = theta[2]
    
    xb = (x@beta)
    print(xb.shape)
    E = (1-norm.cdf((xb+mu)/sigma))*(xb+mu) - sigma*norm.pdf((xb+mu)/sigma)
        

    return E

# def sim_data(theta, N:int): 
#     b = theta[:-1]
#     sig = theta[-1]
#     K=b.size

#     # x will need to contain 1s (a constant term) and randomly generated variables
#     xx = np.random.normal(size=(N,K-1)) 
#     oo = np.ones((N,1))
#     x  = np.hstack((oo, xx))
    
#     # stochastic error term
#     eps = np.random.normal(loc=0, scale=sig, size=(N,))
    
#     y_lat= x@b + eps # the unobserved, latent index (not returned)
#     assert y_lat.ndim==1

#     y = np.fmax(y_lat,0.0) # fmax: elementwise max()

#     return y,x

def q_re(theta, y, x): 
    return -loglikelihood_re(theta, y, x)

def loglikelihood_re(theta, y, x): 
    assert y.ndim == 1, f'y should be 1-dimensional'
    assert theta.ndim == 1, f'theta should be 1-dimensional'

    # unpack parameters 
    b = theta[:-1] # first K parameters are betas, the last is sigma 
    mu = b[-1] # the last beta is mu
    b = b[:-1].reshape(-1,1) # the last beta is mu
    sig = np.abs(theta[-1]) # take abs() to ensure positivity (in case the optimizer decides to try negatives)
    
    phi = norm().pdf((y - (x@b).reshape(-1,) - mu))


    Phi = norm().cdf(((x@b).reshape(-1,)+mu))
    Phi = np.clip(Phi, 1e-8, 1.-1e-8)

    ll =  (y<0.)*np.log(phi * sig) + (y==0.)*np.log(Phi) # get indicator functions by using (y>0) and (y==0)
    
    # print(phi.shape, Phi.shape, (x@b).reshape(-1,).shape, y.shape)
    return ll
