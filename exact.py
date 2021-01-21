import numpy as np
import scipy.special as sps


from cmath import acosh, atanh, cos, cosh, exp, log, pi, sinh, sqrt

from scipy.special import logsumexp




def free_energy(beta,l,j):
    
    beta_c = 1 / 2 * log(1 + sqrt(2)).real
    h = beta*j
    hs = atanh(exp(-2 * h))
    def gamma(beta,l,j,r):
        output = acosh(
            cosh(2 * hs) * cosh(2 * h) -
            sinh(2 * hs) * sinh(2 * h) * cos(r * pi / l))
        if r == 2 * l and beta < beta_c:
            output *= -1
        return output
    logz = (-log(2) + 1 / 2 * l**2 * log(2 * sinh(2 * h)) +
            logsumexp([
                sum([
                    log(2 * cosh(l / 2 * gamma(l, j, beta, 2 * r)))
                    for r in range(1, l + 1)
                ]),
                sum([
                    log(2 * sinh(l / 2 * gamma(l, j, beta, 2 * r)))
                    for r in range(1, l + 1)
                ]),
                sum([
                    log(2 * cosh(l / 2 * gamma(l, j, beta, 2 * r - 1)))
                    for r in range(1, l + 1)
                ]),
                sum([
                    log(2 * sinh(l / 2 * gamma(l, j, beta, 2 * r - 1)))
                    for r in range(1, l + 1)
                ]),
            ])).real
    fbeta = -logz/(l**2)
    
    
    return fbeta/beta

def heat_capacity(beta,l,j):
    
    dd = 5/(4*np.sqrt(l**2))
    d = np.pi**2/(l**2)
    z = 2*beta*j/(1+dd)
    k = 2*np.sinh(z)/((1+d)*np.cosh(z)**2)
    
    
    k1 = sps.ellipk(k)
    k2 = sps.ellipe(k)
    
    p = ((1 - np.sinh(z)**2)**2) / (  ( (1+d)**2 )*( np.cosh(z)**4 ) - 4*( np.sinh(z)**2 )  )
    
    a1 = p*(1+d)**2
    a2 = 2*p -1 
    
    
    c = (z**2  /(np.pi * np.tanh(z)**2)  )  *  (a1 * (k1 - k2) - ( 1 - np.tanh(z)**2 )  * (  (np.pi/2)  +  (2 * a2 * np.tanh(z)**2 - 1 ) * k1)  ) 
    
    return c
