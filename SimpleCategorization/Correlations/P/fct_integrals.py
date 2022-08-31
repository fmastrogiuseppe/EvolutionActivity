import matplotlib.pyplot as plt
import numpy as np
import fct_network as net


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
#### Integrals via quadrature

gaussian_norm = (1./np.sqrt(np.pi))
gauss_points, gauss_weights = np.polynomial.hermite.hermgauss(200)
gauss_points = gauss_points*np.sqrt(2)

# 1-Fct integrals

def Psi (gain=net.gain_psi, offset=net.offset_psi):
    integrand = net.Sigmoid(gauss_points, gain, offset)
    return gaussian_norm * np.dot (integrand,gauss_weights)

def PsiSq (gain=net.gain_psi, offset=net.offset_psi):
    integrand = net.Sigmoid(gauss_points, gain, offset)**2
    return gaussian_norm * np.dot (integrand,gauss_weights)

def PsiPrime (gain=net.gain_psi, offset=net.offset_psi):
    integrand = net.SigmoidPrime(gauss_points, gain, offset)
    return gaussian_norm * np.dot (integrand,gauss_weights)

def PsiPrimeSq (gain=net.gain_psi, offset=net.offset_psi):
    integrand = net.SigmoidPrime(gauss_points, gain, offset)**2
    return gaussian_norm * np.dot (integrand,gauss_weights)

def PsiPrimeFourth (gain=net.gain_psi, offset=net.offset_psi):
    integrand = net.SigmoidPrime(gauss_points, gain, offset)**4
    return gaussian_norm * np.dot (integrand,gauss_weights)
