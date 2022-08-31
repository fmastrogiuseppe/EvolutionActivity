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
offset_psi = 2.

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

def Batch_SimpleCat(mu):

    P, N = mu.shape

    # Inputs

    x_ = mu

    # Indeces

    S_ = np.arange(P)
    o_ = np.zeros(P)
    o_[int(P/2.):] = 1

    # Labels

    labels = net.fh * np.ones(( P, 1 ))
    labels[o_.astype(bool)] = net.fl
    # labels[:, 0] = np.flipud(labels[:, 1])

    return x_, S_, o_, labels


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
