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
import fct_integrals as integ

fac.SetPlotParams()


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
### Compute

# Retrieve

w0 = fac.Retrieve('w0.p', 'Data/')
Y_ = fac.Retrieve('Y_.p', 'Data/Samples/')
K_ = fac.Retrieve('K_.p', 'Data/Samples/')
Y_selectivity_cat = fac.Retrieve('Y_selectivity_cat.p', 'Data/')
Y_selectivity_ctx = fac.Retrieve('Y_selectivity_ctx.p', 'Data/')
Y_average_cat = fac.Retrieve('Y_average_cat.p', 'Data/')
Y_average_ctx = fac.Retrieve('Y_average_ctx.p', 'Data/')

P, N = Y_[1,:,:].shape
Q = int(np.sqrt(P))

# Define indeces of groups to be plot differently

x_, S_, C_, c_, o_, labels = net.Batch_ContextCat(np.random.normal(0, 1., (Q, N)), np.random.normal(0, 1., (Q, N)))

group1A = np.logical_and( np.logical_not(c_.astype(bool)), np.logical_not(o_.astype(bool)) )
group1B = np.logical_and( np.logical_not(c_.astype(bool)), o_.astype(bool) )
group2A = np.logical_and( c_.astype(bool), np.logical_not(o_.astype(bool)) )
group2B = np.logical_and( c_.astype(bool), o_.astype(bool) )


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
### Plot PCA - K

# Subtract mean because we don't want to visualize the unitary axis

X = np.zeros(( 3, K_.shape[1], K_.shape[2] ))
V = np.zeros(( 3, K_.shape[2], 3 ))

X[0,:,:] = K_[0,:,:] - np.mean(K_[0,:,:]) 
CX = np.dot( X[0,:,:].T, X[0,:,:] ) / P
V[0,:,:] = np.linalg.eig(CX)[1].real[:,0:3]

X[1,:,:] = K_[1,:,:] - np.mean(K_[1,:,:]) 
CX = np.dot( X[1,:,:].T, X[1,:,:] ) / P
V[1,:,:] = np.linalg.eig(CX)[1].real[:,0:3]

X[2,:,:] = (K_[1,:,:] - np.mean(K_[1,:,:])) - (K_[0,:,:] - np.mean(K_[0,:,:])) 
CX = np.dot( X[2,:,:].T, X[2,:,:] ) / P
V[2,:,:] = np.linalg.eig(CX)[1].real[:,0:3]

# Re-define centers after mean subtraction

Y_average_cat = np.zeros(( 3, K_.shape[2], 2 ))
Y_average_ctx = np.zeros(( 3, K_.shape[2], 2 ))

Y_average_cat[:,:,1] = np.mean(X[:,o_.astype(bool),:], 1)
Y_average_cat[:,:,0] = np.mean(X[:,np.logical_not(o_.astype(bool)),:], 1)

Y_average_ctx[:,:,1] = np.mean(X[:,c_.astype(bool),:], 1)
Y_average_ctx[:,:,0] = np.mean(X[:,np.logical_not(c_.astype(bool)),:], 1)

# Plot

fac.SetPlotDim(2.6, 2.4)

fig = plt.figure()
ax = plt.axes(projection='3d')

ax.plot(np.dot(X[0,group1A,:], V[0,:,0]), np.dot(X[0,group1A,:], V[0,:,1]), np.dot(X[0,group1A,:], V[0,:,2]), 'o',\
 color = '#FFE5D2', markeredgecolor = '#FF881D', markeredgewidth = 1., markersize=5)
ax.plot(np.dot(X[0,group1B,:], V[0,:,0]), np.dot(X[0,group1B,:], V[0,:,1]), np.dot(X[0,group1B,:], V[0,:,2]), 'o',\
 color = '#D1E3E8', markeredgecolor = '#1E83BA', markeredgewidth = 1., markersize=5)

ax.plot(np.dot(X[0,group2A,:], V[0,:,0]), np.dot(X[0,group2A,:], V[0,:,1]), np.dot(X[0,group2A,:], V[0,:,2]), 's',\
 color = '#FFE5D2', markeredgecolor = '#FF881D', markeredgewidth = 1., markersize=5)
ax.plot(np.dot(X[0,group2B,:], V[0,:,0]), np.dot(X[0,group2B,:], V[0,:,1]), np.dot(X[0,group2B,:], V[0,:,2]), 's',\
 color = '#D1E3E8', markeredgecolor = '#1E83BA', markeredgewidth = 1., markersize=5)

ax.set_xlim3d(2, 17)
ax.set_ylim3d(-26, 26)
ax.set_zlim3d(-22, 22)

ax.dist = 12
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')

plt.savefig('3d_K0.pdf')

plt.show()

#

fig = plt.figure()
ax = plt.axes(projection='3d')

ax.plot(np.dot(X[1,group1A,:], V[1,:,0]), np.dot(X[1,group1A,:], V[1,:,1]), np.dot(X[1,group1A,:], V[1,:,2]),'o',\
 color = '#FFE5D2', markeredgecolor = '#FF881D', markeredgewidth = 1., markersize=5)
ax.plot(np.dot(X[1,group1B,:], V[1,:,0]), np.dot(X[1,group1B,:], V[1,:,1]), np.dot(X[1,group1B,:], V[1,:,2]),'o',\
 color = '#D1E3E8', markeredgecolor = '#1E83BA', markeredgewidth = 1., markersize=5)

ax.plot(np.dot(X[1,group2A,:], V[1,:,0]), np.dot(X[1,group2A,:], V[1,:,1]), np.dot(X[1,group2A,:], V[1,:,2]),'s',\
 color = '#FFE5D2', markeredgecolor = '#FF881D', markeredgewidth = 1., markersize=5)
ax.plot(np.dot(X[1,group2B,:], V[1,:,0]), np.dot(X[1,group2B,:], V[1,:,1]), np.dot(X[1,group2B,:], V[1,:,2]),'s',\
 color = '#D1E3E8', markeredgecolor = '#1E83BA', markeredgewidth = 1., markersize=5)

ax.plot([np.dot(Y_average_cat[1,:,0], V[1,:,0])], [np.dot(Y_average_cat[1,:,0], V[1,:,1])], [np.dot(Y_average_cat[1,:,0], V[1,:,2])],'^',\
 color = '#E2205C', markeredgecolor = '#E2205C', markeredgewidth = .7, markersize=4)
ax.plot([np.dot(Y_average_cat[1,:,1], V[1,:,0])], [np.dot(Y_average_cat[1,:,1], V[1,:,1])], [np.dot(Y_average_cat[1,:,1], V[1,:,2])],'^',\
 color = '#E2205C', markeredgecolor = '#E2205C', markeredgewidth = .7, markersize=4)

ax.plot([np.dot(Y_average_ctx[1,:,0], V[1,:,0])], [np.dot(Y_average_ctx[1,:,0], V[1,:,1])], [np.dot(Y_average_ctx[1,:,0], V[1,:,2])],'^',\
 color = '#FF99C3', markeredgecolor = '#FF99C3', markeredgewidth = .7, markersize=4)
ax.plot([np.dot(Y_average_ctx[1,:,1], V[1,:,0])], [np.dot(Y_average_ctx[1,:,1], V[1,:,1])], [np.dot(Y_average_ctx[1,:,1], V[1,:,2])],'^',\
 color = '#FF99C3', markeredgecolor = '#FF99C3', markeredgewidth = .7, markersize=4)

ax.plot([np.dot(np.mean(X[0,:,:],0), V[1,:,0])], [np.dot(np.mean(X[0,:,:],0), V[1,:,1])], [np.dot(np.mean(X[0,:,:],0), V[1,:,2])], '^',\
 color = '#FFFFFF', markeredgecolor = '0.6', markeredgewidth = .7, markersize=4)

ax.plot([0,np.dot(130*w0[0,:], V[1,:,0])], [0,np.dot(130*w0[0,:], V[1,:,1])], [0,np.dot(130*w0[0,:], V[1,:,2])], '--', color = '0.6')
ax.plot([0,-np.dot(80*w0[0,:], V[1,:,0])], [0,-np.dot(80*w0[0,:], V[1,:,1])], [0,-np.dot(80*w0[0,:], V[1,:,2])], '--', color = '0.6')

ax.set_xlim3d(-30, 30)
ax.set_ylim3d(-30, 0)
ax.set_zlim3d(-30, 30)

ax.dist = 12
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')

plt.savefig('3d_K.pdf')

plt.show()


#

fig = plt.figure()
ax = plt.axes(projection='3d')

ax.plot(np.dot(X[2,group1A,:], V[2,:,0]), np.dot(X[2,group1A,:], V[2,:,1]), np.dot(X[1,group1A,:], V[2,:,2]),'o',\
 color = '#FFE5D2', markeredgecolor = '#FF881D', markeredgewidth = 1., markersize=5)
ax.plot(np.dot(X[2,group1B,:], V[2,:,0]), np.dot(X[2,group1B,:], V[2,:,1]), np.dot(X[1,group1B,:], V[2,:,2]),'o',\
 color = '#D1E3E8', markeredgecolor = '#1E83BA', markeredgewidth = 1., markersize=5)

ax.plot(np.dot(X[2,group2A,:], V[2,:,0]), np.dot(X[2,group2A,:], V[2,:,1]), np.dot(X[1,group2A,:], V[2,:,2]),'s',\
 color = '#FFE5D2', markeredgecolor = '#FF881D', markeredgewidth = 1., markersize=5)
ax.plot(np.dot(X[2,group2B,:], V[2,:,0]), np.dot(X[2,group2B,:], V[2,:,1]), np.dot(X[1,group2B,:], V[2,:,2]),'s',\
 color = '#D1E3E8', markeredgecolor = '#1E83BA', markeredgewidth = 1., markersize=5)

ax.plot([np.dot(Y_average_cat[2,:,0], V[2,:,0])], [np.dot(Y_average_cat[2,:,0], V[2,:,1])], [np.dot(Y_average_cat[2,:,0], V[2,:,2])],'^',\
 color = '#E2205C', markeredgecolor = '#E2205C', markeredgewidth = .7, markersize=4)
ax.plot([np.dot(Y_average_cat[2,:,1], V[2,:,0])], [np.dot(Y_average_cat[2,:,1], V[2,:,1])], [np.dot(Y_average_cat[2,:,1], V[2,:,2])],'^',\
 color = '#E2205C', markeredgecolor = '#E2205C', markeredgewidth = .7, markersize=4)

ax.plot([np.dot(Y_average_ctx[2,:,0], V[2,:,0])], [np.dot(Y_average_ctx[2,:,0], V[2,:,1])], [np.dot(Y_average_ctx[2,:,0], V[2,:,2])],'^',\
 color = '#FF99C3', markeredgecolor = '#FF99C3', markeredgewidth = .7, markersize=4)
ax.plot([np.dot(Y_average_ctx[2,:,1], V[2,:,0])], [np.dot(Y_average_ctx[2,:,1], V[2,:,1])], [np.dot(Y_average_ctx[2,:,1], V[2,:,2])],'^',\
 color = '#FF99C3', markeredgecolor = '#FF99C3', markeredgewidth = .7, markersize=4)

ax.plot([0,np.dot(400*w0[0,:], V[2,:,0])], [0,np.dot(400*w0[0,:], V[2,:,1])], [0,np.dot(400*w0[0,:], V[2,:,2])], '--', color = '0.6')
ax.plot([0,-np.dot(300*w0[0,:], V[2,:,0])], [0,-np.dot(300*w0[0,:], V[2,:,1])], [0,-np.dot(300*w0[0,:], V[2,:,2])], '--', color = '0.6')

ax.set_xlim3d(-30, 30)
ax.set_ylim3d(-30, 30)
ax.set_zlim3d(-30, 30)

ax.dist = 12
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')

plt.savefig('3d_DeltaK.pdf')

plt.show()


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
### Plot PCA - Y

# Subtract mean because we don't want to visualize the unitary axis

X[0,:,:] = Y_[0,:,:] - np.mean(Y_[0,:,:]) 
CX = np.dot( X[0,:,:].T, X[0,:,:] ) / P
V[0,:,:] = np.linalg.eig(CX)[1].real[:,0:3]

X[1,:,:] = Y_[1,:,:] - np.mean(Y_[1,:,:]) 
CX = np.dot( X[1,:,:].T, X[1,:,:] ) / P
V[1,:,:] = np.linalg.eig(CX)[1].real[:,0:3]

X[2,:,:] = (Y_[1,:,:] - np.mean(Y_[1,:,:])) - (Y_[0,:,:] - np.mean(Y_[0,:,:])) 
CX = np.dot( X[2,:,:].T, X[2,:,:] ) / P
V[2,:,:] = np.linalg.eig(CX)[1].real[:,0:3]

# Re-define centers after mean subtraction

Y_average_cat = np.zeros(( 3, Y_.shape[2], 2 ))
Y_average_ctx = np.zeros(( 3, Y_.shape[2], 2 ))

Y_average_cat[:,:,1] = np.mean(X[:,o_.astype(bool),:], 1)
Y_average_cat[:,:,0] = np.mean(X[:,np.logical_not(o_.astype(bool)),:], 1)

Y_average_ctx[:,:,1] = np.mean(X[:,c_.astype(bool),:], 1)
Y_average_ctx[:,:,0] = np.mean(X[:,np.logical_not(c_.astype(bool)),:], 1)

# Plot

fig = plt.figure()
ax = plt.axes(projection='3d')

ax.plot(np.dot(X[0,group1A,:], V[0,:,0]), np.dot(X[0,group1A,:], V[0,:,1]), np.dot(X[0,group1A,:], V[0,:,2]), 'o',\
 color = '#FFE5D2', markeredgecolor = '#FF881D', markeredgewidth = 1., markersize=5)
ax.plot(np.dot(X[0,group1B,:], V[0,:,0]), np.dot(X[0,group1B,:], V[0,:,1]), np.dot(X[0,group1B,:], V[0,:,2]), 'o',\
 color = '#D1E3E8', markeredgecolor = '#1E83BA', markeredgewidth = 1., markersize=5)

ax.plot(np.dot(X[0,group2A,:], V[0,:,0]), np.dot(X[0,group2A,:], V[0,:,1]), np.dot(X[0,group2A,:], V[0,:,2]), 's',\
 color = '#FFE5D2', markeredgecolor = '#FF881D', markeredgewidth = 1., markersize=5)
ax.plot(np.dot(X[0,group2B,:], V[0,:,0]), np.dot(X[0,group2B,:], V[0,:,1]), np.dot(X[0,group2B,:], V[0,:,2]), 's',\
 color = '#D1E3E8', markeredgecolor = '#1E83BA', markeredgewidth = 1., markersize=5)

ax.set_xlim3d(0, 3.5)
ax.set_ylim3d(-4, 4)
ax.set_zlim3d(-4, 4.5)

ax.dist = 12
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')

plt.savefig('3d_Y0.pdf')

plt.show()

#

fig = plt.figure()
ax = plt.axes(projection='3d')

ax.plot(np.dot(X[1,group1A,:], V[1,:,0]), np.dot(X[1,group1A,:], V[1,:,1]), np.dot(X[1,group1A,:], V[1,:,2]),'o',\
 color = '#FFE5D2', markeredgecolor = '#FF881D', markeredgewidth = 1., markersize=5)
ax.plot(np.dot(X[1,group1B,:], V[1,:,0]), np.dot(X[1,group1B,:], V[1,:,1]), np.dot(X[1,group1B,:], V[1,:,2]),'o',\
 color = '#D1E3E8', markeredgecolor = '#1E83BA', markeredgewidth = 1., markersize=5)

ax.plot(np.dot(X[1,group2A,:], V[1,:,0]), np.dot(X[1,group2A,:], V[1,:,1]), np.dot(X[1,group2A,:], V[1,:,2]),'s',\
 color = '#FFE5D2', markeredgecolor = '#FF881D', markeredgewidth = 1., markersize=5)
ax.plot(np.dot(X[1,group2B,:], V[1,:,0]), np.dot(X[1,group2B,:], V[1,:,1]), np.dot(X[1,group2B,:], V[1,:,2]),'s',\
 color = '#D1E3E8', markeredgecolor = '#1E83BA', markeredgewidth = 1., markersize=5)

ax.plot([np.dot(Y_average_cat[1,:,0], V[1,:,0])], [np.dot(Y_average_cat[1,:,0], V[1,:,1])], [np.dot(Y_average_cat[1,:,0], V[1,:,2])],'^',\
 color = '#E2205C', markeredgecolor = '#E2205C', markeredgewidth = .7, markersize=4)
ax.plot([np.dot(Y_average_cat[1,:,1], V[1,:,0])], [np.dot(Y_average_cat[1,:,1], V[1,:,1])], [np.dot(Y_average_cat[1,:,1], V[1,:,2])],'^',\
 color = '#E2205C', markeredgecolor = '#E2205C', markeredgewidth = .7, markersize=4)

ax.plot([np.dot(Y_average_ctx[1,:,0], V[1,:,0])], [np.dot(Y_average_ctx[1,:,0], V[1,:,1])], [np.dot(Y_average_ctx[1,:,0], V[1,:,2])],'^',\
 color = '#FF99C3', markeredgecolor = '#FF99C3', markeredgewidth = .7, markersize=4)
ax.plot([np.dot(Y_average_ctx[1,:,1], V[1,:,0])], [np.dot(Y_average_ctx[1,:,1], V[1,:,1])], [np.dot(Y_average_ctx[1,:,1], V[1,:,2])],'^',\
 color = '#FF99C3', markeredgecolor = '#FF99C3', markeredgewidth = .7, markersize=4)

ax.plot([np.dot(np.mean(X[0,:,:],0), V[1,:,0])], [np.dot(np.mean(X[0,:,:],0), V[1,:,1])], [np.dot(np.mean(X[0,:,:],0), V[1,:,2])], '^',\
 color = '#FFFFFF', markeredgecolor = '0.6', markeredgewidth = .7, markersize=4)

ax.plot([0,np.dot(12*w0[0,:], V[1,:,0])], [0,np.dot(12*w0[0,:], V[1,:,1])], [0,np.dot(12*w0[0,:], V[1,:,2])], '--', color = '0.6')
ax.plot([0,-np.dot(4*w0[0,:], V[1,:,0])], [0,-np.dot(4*w0[0,:], V[1,:,1])], [0,-np.dot(4*w0[0,:], V[1,:,2])], '--', color = '0.6')

ax.set_xlim3d(-4, 4)
ax.set_ylim3d(-4.5, 0.5)
ax.set_zlim3d(-4, 4)

ax.dist = 12
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')

plt.savefig('3d_Y.pdf')

plt.show()

#

fig = plt.figure()
ax = plt.axes(projection='3d')

ax.plot(np.dot(X[2,group1A,:], V[2,:,0]), np.dot(X[2,group1A,:], V[2,:,1]), np.dot(X[1,group1A,:], V[2,:,2]),'o',\
 color = '#FFE5D2', markeredgecolor = '#FF881D', markeredgewidth = 1., markersize=5)
ax.plot(np.dot(X[2,group1B,:], V[2,:,0]), np.dot(X[2,group1B,:], V[2,:,1]), np.dot(X[1,group1B,:], V[2,:,2]),'o',\
 color = '#D1E3E8', markeredgecolor = '#1E83BA', markeredgewidth = 1., markersize=5)

ax.plot(np.dot(X[2,group2A,:], V[2,:,0]), np.dot(X[2,group2A,:], V[2,:,1]), np.dot(X[1,group2A,:], V[2,:,2]),'s',\
 color = '#FFE5D2', markeredgecolor = '#FF881D', markeredgewidth = 1., markersize=5)
ax.plot(np.dot(X[2,group2B,:], V[2,:,0]), np.dot(X[2,group2B,:], V[2,:,1]), np.dot(X[1,group2B,:], V[2,:,2]),'s',\
 color = '#D1E3E8', markeredgecolor = '#1E83BA', markeredgewidth = 1., markersize=5)

ax.plot([np.dot(Y_average_cat[2,:,0], V[2,:,0])], [np.dot(Y_average_cat[2,:,0], V[2,:,1])], [np.dot(Y_average_cat[2,:,0], V[2,:,2])],'^',\
 color = '#E2205C', markeredgecolor = '#E2205C', markeredgewidth = .7, markersize=4)
ax.plot([np.dot(Y_average_cat[2,:,1], V[2,:,0])], [np.dot(Y_average_cat[2,:,1], V[2,:,1])], [np.dot(Y_average_cat[2,:,1], V[2,:,2])],'^',\
 color = '#E2205C', markeredgecolor = '#E2205C', markeredgewidth = .7, markersize=4)

ax.plot([np.dot(Y_average_ctx[2,:,0], V[2,:,0])], [np.dot(Y_average_ctx[2,:,0], V[2,:,1])], [np.dot(Y_average_ctx[2,:,0], V[2,:,2])],'^',\
 color = '#FF99C3', markeredgecolor = '#FF99C3', markeredgewidth = .7, markersize=4)
ax.plot([np.dot(Y_average_ctx[2,:,1], V[2,:,0])], [np.dot(Y_average_ctx[2,:,1], V[2,:,1])], [np.dot(Y_average_ctx[2,:,1], V[2,:,2])],'^',\
 color = '#FF99C3', markeredgecolor = '#FF99C3', markeredgewidth = .7, markersize=4)

ax.plot([0,np.dot(400*w0[0,:], V[2,:,0])], [0,np.dot(400*w0[0,:], V[2,:,1])], [0,np.dot(400*w0[0,:], V[2,:,2])], '--', color = '0.6')
ax.plot([0,-np.dot(300*w0[0,:], V[2,:,0])], [0,-np.dot(300*w0[0,:], V[2,:,1])], [0,-np.dot(300*w0[0,:], V[2,:,2])], '--', color = '0.6')

ax.set_xlim3d(-3, 3)
ax.set_ylim3d(-3, 3)
ax.set_zlim3d(-3, 3)

ax.dist = 12
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')

plt.savefig('3d_DeltaY.pdf')

plt.show()

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####

sys.exit(0)
