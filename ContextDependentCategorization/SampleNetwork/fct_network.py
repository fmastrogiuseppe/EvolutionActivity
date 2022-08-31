import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import fsolve
from functools import partial

import fct_network as net


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
#### Activation functions

gain_phi = 1.
offset_phi = 0.

gain_psi = 1.
offset_psi = 0.

#

def Sigmoid(x, gain, offset):
    return 1. / (1.+np.exp(-gain*(x-offset)))

def SigmoidPrime(x, gain, offset):
    return gain * np.exp(-gain*(x-offset))/(1.+np.exp(-gain*(x-offset)))**2

#

def TorchSigmoid(x, gain, offset):
    return torch.sigmoid(gain*(x-offset)) 


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
#### Targets 

fh = 0.75
fl = 0.25

#

def SolveComputeI(x, f, gain, offset):
    return f - Sigmoid(x, gain, offset)

def ComputeI(f, gain=gain_phi, offset=offset_phi):
    return fsolve(partial(SolveComputeI, f=f, gain=gain, offset=offset), 0.1)

ih = ComputeI(fh)
il = ComputeI(fl)


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
#### Generate batches

def Batch_ContextCat(mu, nu):

    Q, N = mu.shape
    P = int(Q**2)

    x_ = np.zeros(( P, N ))
    S_ = np.zeros(( P ))
    C_ = np.zeros(( P ))
    c_ = np.zeros(( P ))
    o_ = np.zeros(( P ))

   # Indeces 

    c_[int(0.5*P):] = 1

    for i in range(Q):
        C_[i*Q:i*Q+Q] = i

        for j in range(Q):
            S_[i*Q+j] = j

    # Inputs

    for s in range(P):

        x_[s] = ( mu[int(S_[s]),:] + nu[int(C_[s]),:] ) / np.sqrt(2.)

    # Labels

    labels = net.fh * np.ones(( P, 1 ))
    
    for s in range(P):

        if ( (c_[s] == 0 and S_[s]>=(Q/2.)) or (c_[s] == 1 and S_[s]< Q/2.) ):
            labels[s,0] = net.fl
            o_[s] = 1

    return x_, S_.astype(int), C_.astype(int), c_.astype(int), o_.astype(int), labels


def XdotXMatrix(Q):

    N = 100
    P = int(Q**2)
    x_, S_, C_, c_, o_, labels = net.Batch_ContextCat(np.random.normal(0, 1., (Q, N)), np.random.normal(0, 1., (Q, N)))

    xdotx = np.identity(P)

    for j in range(P):
        for k in range(P):
            if k != j:

                if (C_[j] == C_[k]) or (S_[j] == S_[k]): 
                    xdotx[j,k] = 0.5

    return xdotx

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
#### Torch Model

class BuildModel(torch.nn.Module):

    def __init__(self, N1, N2, N3, \
        gain_psi=gain_psi, offset_psi=offset_psi,\
        gain_phi=gain_phi, offset_phi=offset_phi):

        super(BuildModel, self).__init__()

        self.N1 = N1
        self.N2 = N2
        self.N3 = N3

        self.U = nn.Linear(self.N1, self.N2, bias = False)
        self.W = nn.Linear(self.N2, self.N3, bias = False)

        self.gain_psi = gain_psi
        self.offset_psi = offset_psi

        self.gain_phi = gain_phi
        self.offset_phi = offset_phi
        
    def forward(self, x_):

        k_ = self.U(x_)
        y_ = TorchSigmoid(k_, self.gain_psi, self.offset_psi)
        h_ = self.W(y_)
        z_ = TorchSigmoid(h_, self.gain_phi, self.offset_phi)

        return k_, y_, z_
