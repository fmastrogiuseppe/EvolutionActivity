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

def PsiThird (gain=net.gain_psi, offset=net.offset_psi):
    integrand = net.Sigmoid(gauss_points, gain, offset)**3
    return gaussian_norm * np.dot (integrand,gauss_weights)

def PsiFourth (gain=net.gain_psi, offset=net.offset_psi):
    integrand = net.Sigmoid(gauss_points, gain, offset)**4
    return gaussian_norm * np.dot (integrand,gauss_weights)

def PsiPrime (gain=net.gain_psi, offset=net.offset_psi):
    integrand = net.SigmoidPrime(gauss_points, gain, offset)
    return gaussian_norm * np.dot (integrand,gauss_weights)

def PsiPrimeSq (gain=net.gain_psi, offset=net.offset_psi):
    integrand = net.SigmoidPrime(gauss_points, gain, offset)**2
    return gaussian_norm * np.dot (integrand,gauss_weights)

def PsiPrimeThird (gain=net.gain_psi, offset=net.offset_psi):
    integrand = net.SigmoidPrime(gauss_points, gain, offset)**3
    return gaussian_norm * np.dot (integrand,gauss_weights)

def PsiPrimeFourth (gain=net.gain_psi, offset=net.offset_psi):
    integrand = net.SigmoidPrime(gauss_points, gain, offset)**4
    return gaussian_norm * np.dot (integrand,gauss_weights)

def PsiSec (gain=net.gain_psi, offset=net.offset_psi):
    integrand = net.SigmoidSec(gauss_points, gain, offset)
    return gaussian_norm * np.dot (integrand,gauss_weights)


# 2-Fct integrals, with 1/2 correlation

def InnerPsiPsi (z, gain, offset):
    integrand = net.Sigmoid(np.sqrt(0.5)*gauss_points + np.sqrt(0.5)*z, gain, offset)
    return gaussian_norm * np.dot (integrand,gauss_weights)

def PsiPsi (gain=net.gain_psi, offset=net.offset_psi):
    integrand = InnerPsiPsi(gauss_points, gain, offset)
    return gaussian_norm * np.dot (integrand**2,gauss_weights)

def InnerPsiPrimePsiPrime (z, gain, offset):
    integrand = net.SigmoidPrime(np.sqrt(0.5)*gauss_points + np.sqrt(0.5)*z, gain, offset)
    return gaussian_norm * np.dot (integrand,gauss_weights)

def PsiPrimePsiPrime (gain=net.gain_psi, offset=net.offset_psi):
    integrand = InnerPsiPrimePsiPrime(gauss_points, gain, offset)
    return gaussian_norm * np.dot (integrand**2,gauss_weights)

def InnerPsiPrimeSqPsiPrimeSq (z, gain, offset):
    integrand = net.SigmoidPrime(np.sqrt(0.5)*gauss_points + np.sqrt(0.5)*z, gain, offset)**2
    return gaussian_norm * np.dot (integrand,gauss_weights)

def PsiPrimeSqPsiPrimeSq (gain=net.gain_psi, offset=net.offset_psi):
    integrand = InnerPsiPrimeSqPsiPrimeSq(gauss_points, gain, offset)
    return gaussian_norm * np.dot (integrand**2,gauss_weights)
    

InnerPsiPsi = np.vectorize(InnerPsiPsi)
InnerPsiPrimePsiPrime = np.vectorize(InnerPsiPrimePsiPrime)
InnerPsiPrimeSqPsiPrimeSq = np.vectorize(InnerPsiPrimeSqPsiPrimeSq)
