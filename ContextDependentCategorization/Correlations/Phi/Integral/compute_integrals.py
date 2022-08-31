import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import sys
from scipy import integrate
import os

sys.path.append('..')

import fct_network as net
import fct_facilities as fac

os.environ["OMP_NUM_THREADS"] = "4" # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "4" # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "4" # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "4" # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "4" # export NUMEXPR_NUM_THREADS=1

bound = 40.
lim = 8


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
### Parameters

gain_values = fac.Retrieve('gain_values.p', '')
offset_values = fac.Retrieve('offset_values.p', '')

# Define local parameters

simID = int(sys.argv[1]) - 1

gainID = simID//len(offset_values)
gain = gain_values[gainID]
offsetID = simID%len(offset_values)
offset = offset_values[offsetID]

print (gain, offset)


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
### Compute integrals

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
#### Int_11p3_12 (former 3100)

def Fct_Int_11p3_12(x1, y1, y2):

    normal = np.exp(- (x1**2 + y1**2 + y2**2 ) /2.) / np.sqrt(2*np.pi)**3

    f1 =  net.SigmoidPrime( ( x1 + y1 ) / np.sqrt(2) , gain, offset)**3
    f2 =  net.SigmoidPrime( ( x1 + y2 ) / np.sqrt(2) , gain, offset)

    return normal * f1 * f2

Int_11p3_12 = integrate.nquad(Fct_Int_11p3_12, [[-bound, bound],[-bound, bound],[-bound, bound]], opts = {'limit':lim}) [0]

print ('Int_11p3_12', Int_11p3_12)
fac.Store(Int_11p3_12, 'Int_11p3_12___'+str(gainID)+str(offsetID)+'.p', 'IntegralsData/')


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
#### Int_11p2_12_13 (former IntA)

def Fct_Int_11p2_12_13(x1, y1, y2, y3):

    normal = np.exp(- (x1**2 + y1**2 + y2**2 + y3**2) /2.) / np.sqrt(2*np.pi)**4

    f1 =  net.SigmoidPrime( ( x1 + y1 ) / np.sqrt(2) ,  gain, offset)**2
    f2 =  net.SigmoidPrime( ( x1 + y2 ) / np.sqrt(2) , gain, offset)
    f3 =  net.SigmoidPrime( ( x1 + y3 ) / np.sqrt(2) , gain, offset)

    return normal * f1 * f2 * f3

Int_11p2_12_13 = integrate.nquad(Fct_Int_11p2_12_13, [[-bound, bound],[-bound, bound],[-bound, bound],[-bound, bound]], opts = {'limit':lim}) [0]

print ('Int_11p2_12_13', Int_11p2_12_13)
fac.Store(Int_11p2_12_13, 'Int_11p2_12_13___'+str(gainID)+str(offsetID)+'.p', 'IntegralsData/')


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
#### Int_11p2_12_22 (former Eint2110)

def Fct_Int_11p2_12_22(x1, x2, y1, y2):

    normal = np.exp(- (x1**2 + x2**2 + y1**2 + y2**2) /2.) / np.sqrt(2*np.pi)**4

    f1 =  net.SigmoidPrime( ( x1 + y1 ) / np.sqrt(2) , gain, offset)**2
    f2 =  net.SigmoidPrime( ( x1 + y2 ) / np.sqrt(2) , gain, offset)
    f3 =  net.SigmoidPrime( ( x2 + y2 ) / np.sqrt(2) , gain, offset)

    return normal * f1 * f2 * f3

Int_11p2_12_22 = integrate.nquad(Fct_Int_11p2_12_22, [[-bound, bound],[-bound, bound],[-bound, bound],[-bound, bound]], opts = {'limit':lim}) [0]

print ('Int_11p2_12_22', Int_11p2_12_22)
fac.Store(Int_11p2_12_22, 'Int_11p2_12_22___'+str(gainID)+str(offsetID)+'.p', 'IntegralsData/')


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
#### Int_11p2_12_21 (former Eint2101)

def Fct_Int_11p2_12_21(x1, x2, y1, y2):

    normal = np.exp(- (x1**2 + x2**2 + y1**2 + y2**2) /2.) / np.sqrt(2*np.pi)**4

    f1 =  net.SigmoidPrime( ( x1 + y1 ) / np.sqrt(2) , gain, offset)**2
    f2 =  net.SigmoidPrime( ( x1 + y2 ) / np.sqrt(2) , gain, offset)
    f3 =  net.SigmoidPrime( ( x2 + y1 ) / np.sqrt(2) , gain, offset)

    return normal * f1 * f2 * f3

Int_11p2_12_21 = integrate.nquad(Fct_Int_11p2_12_21, [[-bound, bound],[-bound, bound],[-bound, bound],[-bound, bound]], opts = {'limit':lim}) [0]

print ('Int_11p2_12_21', Int_11p2_12_21)
fac.Store(Int_11p2_12_21, 'Int_11p2_12_21___'+str(gainID)+str(offsetID)+'.p', 'IntegralsData/')


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
#### Int_11_12_13_14 (former IntC)

def Fct_Int_11_12_13_14(x1, y1, y2, y3, y4):

    normal = np.exp(- (x1**2 + y1**2 + y2**2 + y3**2 + y4**2) /2.) / np.sqrt(2*np.pi)**5

    f1 =  net.SigmoidPrime( ( x1 + y1 ) / np.sqrt(2) , gain, offset)
    f2 =  net.SigmoidPrime( ( x1 + y2 ) / np.sqrt(2) , gain, offset)
    f3 =  net.SigmoidPrime( ( x1 + y3 ) / np.sqrt(2) , gain, offset)
    f4 =  net.SigmoidPrime( ( x1 + y4 ) / np.sqrt(2) , gain, offset)

    return normal * f1 * f2 * f3 * f4

Int_11_12_13_14 = integrate.nquad(Fct_Int_11_12_13_14, [[-bound, bound],[-bound, bound],[-bound, bound],[-bound, bound],[-bound, bound]], opts = {'limit':lim}) [0]

print ('Int_11_12_13_14', Int_11_12_13_14)
fac.Store(Int_11_12_13_14, 'Int_11_12_13_14___'+str(gainID)+str(offsetID)+'.p', 'IntegralsData/')


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
#### Int_11_12_13_23 (former IntD)

def Fct_Int_11_12_13_23(x1, x2, y1, y2, y3):

    normal = np.exp(- (x1**2 +x2**2 + y1**2 + y2**2 + y3**2) /2.) / np.sqrt(2*np.pi)**5

    f1 =  net.SigmoidPrime( ( x1 + y1 ) / np.sqrt(2) , gain, offset)
    f2 =  net.SigmoidPrime( ( x1 + y2 ) / np.sqrt(2) , gain, offset)
    f3 =  net.SigmoidPrime( ( x1 + y3 ) / np.sqrt(2) , gain, offset)
    f4 =  net.SigmoidPrime( ( x2 + y3 ) / np.sqrt(2) , gain, offset)

    return normal * f1 * f2 * f3 * f4

Int_11_12_13_23 = integrate.nquad(Fct_Int_11_12_13_23, [[-bound, bound],[-bound, bound],[-bound, bound],[-bound, bound],[-bound, bound]], opts = {'limit':lim}) [0]

print ('Int_11_12_13_23', Int_11_12_13_23)
fac.Store(Int_11_12_13_23, 'Int_11_12_13_23___'+str(gainID)+str(offsetID)+'.p', 'IntegralsData/')


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
#### Int_11_12_22_23 (former IntE)

def Fct_Int_11_12_22_23(x1, x2, y1, y2, y3):

    normal = np.exp(- (x1**2 +x2**2 + y1**2 + y2**2 + y3**2) /2.) / np.sqrt(2*np.pi)**5

    f1 =  net.SigmoidPrime( ( x1 + y1 ) / np.sqrt(2) , gain, offset)
    f2 =  net.SigmoidPrime( ( x1 + y2 ) / np.sqrt(2) , gain, offset)
    f3 =  net.SigmoidPrime( ( x2 + y2 ) / np.sqrt(2) , gain, offset)
    f4 =  net.SigmoidPrime( ( x2 + y3 ) / np.sqrt(2) , gain, offset)

    return normal * f1 * f2 * f3 * f4

Int_11_12_22_23 = integrate.nquad(Fct_Int_11_12_22_23, [[-bound, bound],[-bound, bound],[-bound, bound],[-bound, bound],[-bound, bound]], opts = {'limit':lim}) [0]

print ('Int_11_12_22_23', Int_11_12_22_23)
fac.Store(Int_11_12_22_23, 'Int_11_12_22_23___'+str(gainID)+str(offsetID)+'.p', 'IntegralsData/')


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
#### Int_11_12_21_22 (former 1111)

def Fct_Int_11_12_21_22(x1, x2, y1, y2):

    normal = np.exp(- (x1**2 +x2**2 + y1**2 + y2**2) /2.) / np.sqrt(2*np.pi)**4

    f1 =  net.SigmoidPrime( ( x1 + y1 ) / np.sqrt(2) , gain, offset)
    f2 =  net.SigmoidPrime( ( x1 + y2 ) / np.sqrt(2) , gain, offset)
    f3 =  net.SigmoidPrime( ( x2 + y1 ) / np.sqrt(2) , gain, offset)
    f4 =  net.SigmoidPrime( ( x2 + y2 ) / np.sqrt(2) , gain, offset)

    return normal * f1 * f2 * f3 * f4

Int_11_12_21_22 = integrate.nquad(Fct_Int_11_12_21_22, [[-bound, bound],[-bound, bound],[-bound, bound],[-bound, bound]], opts = {'limit':lim}) [0]

print ('Int_11_12_21_22', Int_11_12_21_22)
fac.Store(Int_11_12_21_22, 'Int_11_12_21_22___'+str(gainID)+str(offsetID)+'.p', 'IntegralsData/')

