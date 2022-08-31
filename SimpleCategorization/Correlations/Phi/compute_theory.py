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

sys.path.append('')

import fct_facilities as fac
import fct_network as net
import fct_integrals as integ


fac.SetPlotParams()


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
### Set parameters

P = 20
N = 200
eta = 0.4


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
### Theory 

gain_values_th = np.linspace( 1, 4, 80 )
offset_values_th = np.linspace( 0, 2, 80 )

K_bar_th = np.zeros(( 2, 2, len(gain_values_th), len(gain_values_th) ))  # First axis indicates t = 0 or t = T
Y_bar_th = np.zeros(( 2, 2, len(gain_values_th), len(gain_values_th) ))

K_corr_th = np.zeros(( 2, len(gain_values_th), len(offset_values_th) ))
Y_corr_th = np.zeros(( 2, len(gain_values_th), len(offset_values_th) ))

#

doCompute = 0

if doCompute:

	for i, gain in enumerate(gain_values_th):

		for j, offset in enumerate(offset_values_th):

			ih = net.ComputeI(net.fh, gain, offset)
			il = net.ComputeI(net.fl, gain, offset)

			#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
			### t = 0 

			# Almost everyinteging remains 0

			#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
			### t = T 

			# Compute coordinates

			alpha = integ.PsiPrimeSq() + eta * (integ.PsiSq() - integ.Psi()**2)
			beta = eta * P/2. * integ.Psi()**2 

			ch = ( ih + beta/alpha*(ih - il) ) /  (alpha + 2*beta) 
			cl = ( il - beta/alpha*(ih - il) ) / (alpha + 2*beta)

			# Projections on axes

			K_bar_th[1,0,i,j] = ch * integ.PsiPrime()
			K_bar_th[1,1,i,j] = cl * integ.PsiPrime()

			Y_bar_th[1,0,i,j] = ch * integ.PsiPrimeSq()
			Y_bar_th[1,1,i,j] = cl * integ.PsiPrimeSq()

			# Compute dot products

			KnormA = N + integ.PsiPrimeSq() * ch**2
			KnormB = N + integ.PsiPrimeSq() * cl**2 
			KAA = integ.PsiPrime()**2 * ch**2 
			KBB = integ.PsiPrime()**2 * cl**2 
			KAB = integ.PsiPrime()**2 * ( ch * cl )

			YnormA = N * ( integ.PsiSq() - integ.Psi()**2 ) + integ.PsiPrimeFourth() * ch**2 
			YnormB = N * ( integ.PsiSq() - integ.Psi()**2 ) + integ.PsiPrimeFourth() * cl**2 
			YAA = integ.PsiPrimeSq()**2 * ch**2
			YBB = integ.PsiPrimeSq()**2 * cl**2 
			YAB = integ.PsiPrimeSq()**2 * ( ch * cl )

			# Activity measures

			K_corr_th[1,i,j] = KAB / ( np.sqrt(KnormA) * np.sqrt(KnormB) )
			Y_corr_th[1,i,j] = YAB / ( np.sqrt(YnormA) * np.sqrt(YnormB) )


	# Store 

	fac.Store(gain_values_th, 'gain_values_th.p', 'Results/')
	fac.Store(offset_values_th, 'offset_values_th.p', 'Results/')

	fac.Store(K_bar_th, 'K_bar_th.p', 'Results/')
	fac.Store(Y_bar_th, 'Y_bar_th.p', 'Results/')

	fac.Store(K_corr_th, 'K_corr_th.p', 'Results/')
	fac.Store(Y_corr_th, 'Y_corr_th.p', 'Results/')

else:

	# Retrieve

	gain_values_th = fac.Retrieve('gain_values_th.p', 'Results/')
	offset_values_th = fac.Retrieve('offset_values_th.p', 'Results/')

	K_bar_th = fac.Retrieve('K_bar_th.p', 'Results/')
	Y_bar_th = fac.Retrieve('Y_bar_th.p', 'Results/')

	K_corr_th = fac.Retrieve('K_corr_th.p', 'Results/')
	Y_corr_th = fac.Retrieve('Y_corr_th.p', 'Results/')


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
### Plot 

fac.SetPlotDim(2.1, 1.8)
dashes = [3,3]

#

cmap_base =  'bwr' 
vmin, vmax = 0.2, 0.8
cmap = fac.TruncateCmap(cmap_base, vmin, vmax)

#

fg = plt.figure()
ax0 = plt.axes(frameon=True)

cax = plt.imshow(Y_corr_th[1,:,:].T, aspect = 'auto', vmin=-0.1, vmax=0.1, \
	extent = (min(gain_values_th)/4., max(gain_values_th)/4., min(offset_values_th), max(offset_values_th)), origin='lower', interpolation='nearest', cmap = cmap)

cbar = fg.colorbar(cax, ticks=[-0.1, 0, 0.1], orientation='vertical')
cbar.ax.set_xticklabels(['-0.2', '0', '0.2'])

plt.xlabel(r'Gain')
plt.ylabel(r'Threshold')

plt.grid('off')

plt.xticks([0.25, 0.5, 0.75, 1])
plt.yticks([0, 1, 2])

# plt.colorbar()

ax0.spines['top'].set_visible(False)
ax0.spines['right'].set_visible(False)
ax0.yaxis.set_ticks_position('left')
ax0.xaxis.set_ticks_position('bottom')
# plt.locator_params(nbins=4)

plt.savefig('Y_corr.pdf')
plt.show()

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####

sys.exit(0)
