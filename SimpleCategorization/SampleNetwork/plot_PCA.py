#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####

### Import functions

import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import torch
import torch.nn as nn
from matplotlib import cm
from mpl_toolkits import mplot3d

sys.path.append('')

import fct_facilities as fac
import fct_network as net
import fct_analysis as an

fac.SetPlotParams()


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
### Compute

# Retrieve

w0 = fac.Retrieve('w0.p', 'Data/')
Y_ = fac.Retrieve('Y_.p', 'Data/')
Y_selectivity = fac.Retrieve('Y_selectivity.p', 'Data/')
Y_average = fac.Retrieve('Y_average.p', 'Data/')

P, N = Y_[1,:,:].shape
x_, S_, o_, labels = net.Batch_SimpleCat(np.random.normal(0, 1., (P, N)))


# PCA

X = np.copy(Y_)

X[0,:,:] = Y_[0,:,:] - np.mean(Y_[0,:,:])
CX = np.dot( X[0,:,:].T, X[0,:,:] ) / P
V0 = np.linalg.eig(CX)[1].real 

X[1,:,:] = Y_[1,:,:] - np.mean(Y_[1,:,:])
CX = np.dot( X[1,:,:].T, X[1,:,:] ) / P
V = np.linalg.eig(CX)[1].real 

# Recompute centers after mean subtraction

Y_average[:,:,1] = np.mean(X[:,o_.astype(bool),:], 1)
Y_average[:,:,0] = np.mean(X[:,np.logical_not(o_.astype(bool)),:], 1)


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
### Plots

dashes = [3,3]

#

fac.SetPlotDim(2.6, 2.4)

fig = plt.figure()
ax = plt.axes(projection='3d')

ax.plot(np.dot(X[0,:P/2,:], V0[:,0]), np.dot(X[0,:P/2,:], V0[:,1]), np.dot(X[0,:P/2,:], V0[:,2]), 'o',\
 color = '#FFE5D2', markeredgecolor = '#FF881D', markeredgewidth = 1., markersize=5)
ax.plot(np.dot(X[0,P/2:,:], V0[:,0]), np.dot(X[0,P/2:,:], V0[:,1]), np.dot(X[0,P/2:,:], V0[:,2]), 'o',\
 color = '#D1E3E8', markeredgecolor = '#1E83BA', markeredgewidth = 1., markersize=5)

ax.set_xlim3d(-1.5, 1.5)
ax.set_ylim3d(-1.5, 1.5)
ax.set_zlim3d(-1.5, 1.5)

ax.dist = 12
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')

plt.savefig('3d0.pdf')

plt.show()

#

fig = plt.figure()
ax = plt.axes(projection='3d')

ax.plot(np.dot(X[1,:P/2,:], V[:,0]), np.dot(X[1,:P/2,:], V[:,1]), np.dot(X[1,:P/2,:], V[:,2]), 'o',\
 color = '#FFE5D2', markeredgecolor = '#FF881D', markeredgewidth = 1., markersize=5)
ax.plot(np.dot(X[1,P/2:,:], V[:,0]), np.dot(X[1,P/2:,:], V[:,1]), np.dot(X[1,P/2:,:], V[:,2]), 'o',\
 color = '#D1E3E8', markeredgecolor = '#1E83BA', markeredgewidth = 1., markersize=5)

ax.plot([np.dot(Y_average[1,:,0], V[:,0])], [np.dot(Y_average[1,:,0], V[:,1])], [np.dot(Y_average[1,:,0], V[:,2])], '^',\
 color = '#E2205C', markeredgecolor = '#E2205C', markeredgewidth = .7, markersize=4)
ax.plot([np.dot(Y_average[1,:,1], V[:,0])], [np.dot(Y_average[1,:,1], V[:,1])], [np.dot(Y_average[1,:,1], V[:,2])], '^',\
 color = '#E2205C', markeredgecolor = '#E2205C', markeredgewidth = .7, markersize=4)

ax.plot([np.dot((Y_average[0,:,0]+Y_average[0,:,1])/2, V[:,0])], [np.dot((Y_average[0,:,0]+Y_average[0,:,1])/2, V[:,1])], [np.dot((Y_average[0,:,0]+Y_average[0,:,1])/2, V[:,2])], '^',\
 color = '#FFFFFF', markeredgecolor = '0.6', markeredgewidth = .7, markersize=4)

ax.plot([0,np.dot(3*w0[0,:], V[:,0])], [0,np.dot(3*w0[0,:], V[:,1])], [0,np.dot(3*w0[0,:], V[:,2])], '--', color = '0.6')
ax.plot([0,-np.dot(3*w0[0,:], V[:,0])], [0,-np.dot(3*w0[0,:], V[:,1])], [0,-np.dot(3*w0[0,:], V[:,2])], '--', color = '0.6')

ax.set_xlim3d(-2.5, 2.5)
ax.set_xlim3d(-2.5, 2.5)
ax.set_xlim3d(-2.5, 2.5)

ax.dist = 12
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')

plt.savefig('3d.pdf')

plt.show()


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####

sys.exit(0)
