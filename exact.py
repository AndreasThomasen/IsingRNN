import numpy as np
import scipy.integrate as integrate
import scipy.special as sps


def free_energy(beta,n,j):
    
    dd = 5/(4*np.sqrt(n))
    d = np.pi**2/n
    z = 2*beta*j/(1+dd)
    k = 2*np.sinh(z)/((1+d)*np.cosh(z)**2)
    
    fbeta = -np.log(2)/2 - np.log(np.cosh(z)) - integrate.romberg( lambda x: np.log( 1 + np.sqrt(1-k**2*np.cos(x)**2)) ,0,np.pi)/(2*np.pi)
    
    
    return fbeta/beta

def heat_capacity(beta,n,j):
    
    dd = 5/(4*np.sqrt(n))
    d = np.pi**2/n
    z = 2*beta*j/(1+dd)
    k = 2*np.sinh(z)/((1+d)*np.cosh(z)**2)
    
    
    k1 = sps.ellipk(k)
    k2 = sps.ellipe(k)
    
    p = ((1 - np.sinh(z)**2)**2) / (  ( (1+d)**2 )*( np.cosh(z)**4 ) - 4*( np.sinh(z)**2 )  )
    
    a1 = p*(1+d)**2
    a2 = 2*p -1 
    
    
    c = (z**2  /(np.pi * np.tanh(z)**2)  )  *  (a1 * (k1 - k2) - ( 1 - np.tanh(z)**2 )  * (  (np.pi/2)  +  (2 * a2 * np.tanh(z)**2 - 1 ) * k1)  ) 
    
    return c
