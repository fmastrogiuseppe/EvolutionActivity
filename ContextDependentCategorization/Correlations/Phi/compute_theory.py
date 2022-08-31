#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####

### Import functions

import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import torch
import torch.nn as nn

sys.path.append('')

import fct_facilities as fac
import fct_network as net
import fct_integrals as integ
import fct_analysis as an


fac.SetPlotParams()


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
### Set parameters 

N = 600
eta = 0.2
Q = 8

P = Q**2


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
### Load

gain_values = fac.Retrieve('gain_values.p', 'Integral/') 
offset_values = fac.Retrieve('offset_values.p', 'Integral/') 

gainID = 0
offsetID = 0

gain = gain_values[gainID]
offset = offset_values[offsetID]

#


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
### Theory 

gain_phi_values = np.linspace(1, 4, 40)
offset_phi_values = np.linspace(0, 2, 40)


doCompute = 1

if doCompute:

	K_corr_cat_th = np.zeros(( 2, len(gain_phi_values), len(offset_phi_values) ))
	K_corr_ctx_th = np.zeros(( 2, len(gain_phi_values), len(offset_phi_values) ))
	Y_corr_cat_th = np.zeros(( 2, len(gain_phi_values), len(offset_phi_values) ))
	Y_corr_ctx_th = np.zeros(( 2, len(gain_phi_values), len(offset_phi_values) ))

	#

	Int_11p3_12 = fac.Retrieve('Int_11p3_12___'+str(gainID)+str(offsetID)+'.p', 'Integral/IntegralsData/') # former 3100
	Int_11p2_12_13 = fac.Retrieve('Int_11p2_12_13___'+str(gainID)+str(offsetID)+'.p', 'Integral/IntegralsData/') # former IntA
	Int_11p2_12_22 = fac.Retrieve('Int_11p2_12_22___'+str(gainID)+str(offsetID)+'.p', 'Integral/IntegralsData/') # former Eint2110
	Int_11p2_12_21 = fac.Retrieve('Int_11p2_12_21___'+str(gainID)+str(offsetID)+'.p', 'Integral/IntegralsData/') # former Eint2101
	Int_11_12_13_14 = fac.Retrieve('Int_11_12_13_14___'+str(gainID)+str(offsetID)+'.p', 'Integral/IntegralsData/') # former IntC
	Int_11_12_13_23 = fac.Retrieve('Int_11_12_13_23___'+str(gainID)+str(offsetID)+'.p', 'Integral/IntegralsData/') # former IntD
	Int_11_12_22_23 = fac.Retrieve('Int_11_12_22_23___'+str(gainID)+str(offsetID)+'.p', 'Integral/IntegralsData/') # former IntE
	Int_11_12_21_22 = fac.Retrieve('Int_11_12_21_22___'+str(gainID)+str(offsetID)+'.p', 'Integral/IntegralsData/') # former 1111

	# Compute

	for i, gain_phi in enumerate(gain_phi_values):

		print ('*** ', i)

		for ii, offset_phi in enumerate(offset_phi_values):

			print (ii)

			ih = net.ComputeI(net.fh, gain_phi, offset_phi)
			il = net.ComputeI(net.fl, gain_phi, offset_phi)

			#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
			### Setup useful quantities

			x_, S_, C_, c_, o_, labels = net.Batch_ContextCat(np.random.normal(0, 1., (Q, N)), np.random.normal(0, 1., (Q, N)))

			xdotx = net.XdotXMatrix(Q)

			neighbours = []
			neighbours_plus = []

			for j in range(P):

				n = []

				for k in range(P):
					if k != j:
						if (C_[j] == C_[k]) or (S_[j] == S_[k]): 
							n.append(k)

				neighbours.append(n)
				nn = n[:]
				nn.append(j)
				neighbours_plus.append(nn)

			# PsiPrime integrals

			pp1 = integ.PsiPrimeSq(gain, offset)
			pp2 = integ.PsiPrimePsiPrime(gain, offset)
			pp3 = integ.PsiPrime(gain, offset)**2

			# Compute coordinates

			alpha = integ.PsiPrimeSq(gain, offset) + eta * (integ.PsiSq(gain, offset) - integ.Psi(gain, offset)**2) \
					- 2* ( 0.5*integ.PsiPrimePsiPrime(gain, offset) + eta * (integ.PsiPsi(gain, offset) - integ.Psi(gain, offset)**2) )
			beta = Q * ( 0.5*integ.PsiPrimePsiPrime(gain, offset) + eta * (integ.PsiPsi(gain, offset) - integ.Psi(gain, offset)**2) ) \
					+ eta * P/2. * integ.Psi(gain, offset)**2 

			ch = ( ih + beta/alpha*(ih - il) ) /  (alpha + 2*beta) 
			cl = ( il - beta/alpha*(ih - il) ) /  (alpha + 2*beta) 

			# Activity coordinates vectors

			c = cl * np.ones(P)
			c[np.where(labels[:,0] == net.fh)[0]] = ch

			#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
			#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
			# K

			K_dots_th = np.zeros(( 2, 2, P ))
			K_corrs_th = np.zeros(( 2, 2, P ))

			# Compute dot products between s=cat and s=j

			for cat_idx, cat in enumerate([0,int(Q/2)]): # Loop for computing trials with diff statistics

				for j in range(P):

					# Initial activity

					if j == cat:							K_dots_th[0,cat_idx,j] += 1
					elif neighbours[cat].count(j) == 1:  	K_dots_th[0,cat_idx,j] += 1/2.
					else:									K_dots_th[0,cat_idx,j] += 0

					if j == cat:							K_dots_th[1,cat_idx,j] += 1
					elif neighbours[cat].count(j) == 1:  	K_dots_th[1,cat_idx,j] += 1/2.
					else:									K_dots_th[1,cat_idx,j] += 0

					# Activity changes

					for k in neighbours_plus[j]:
						for kk in neighbours_plus[cat]:

							dot = (xdotx[cat,kk] * xdotx[j,k])*(c[k]*c[kk]) / N

							if kk == k:								dot = dot*pp1
							elif neighbours[kk].count(k) == 1:  	dot = dot*pp2
							else:									dot = dot*pp3

							K_dots_th[1,cat_idx,j] += dot

			# Divide by std

			K_corrs_th[0,:,:] = K_dots_th[0,:,:] / K_dots_th[0,0,0]

			std = np.sqrt(K_dots_th[1,1,int(Q/2)]) * np.ones( P )
			std[np.where(labels[:,0] == net.fh)[0]] = np.sqrt(K_dots_th[1,0,0])

			K_corrs_th[1,0,:] = K_dots_th[1,0,:] / std / np.sqrt(K_dots_th[1,0,0])
			K_corrs_th[1,1,:] = K_dots_th[1,1,:] / std / np.sqrt(K_dots_th[1,1,int(Q/2)])

			#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
			# Activity measures

			# Compute correlation

			K_corr_cat_th[0,i,ii] = an.Corr_th(K_corrs_th[0,:,:], o_)
			K_corr_ctx_th[0,i,ii] = an.Corr_th(K_corrs_th[0,:,:], c_)

			K_corr_cat_th[1,i,ii] = an.Corr_th(K_corrs_th[1,:,:], o_)
			K_corr_ctx_th[1,i,ii] = an.Corr_th(K_corrs_th[1,:,:], c_)

			#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
			#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
			# Y

			Y_dots_th = np.zeros(( 2, 2, P ))
			Y_corrs_th = np.zeros(( 2, 2, P ))


			# Compute dot products between s=cat and s=j

			for cat_idx, cat in enumerate([0,int(Q/2)]): # Loop for computing trials with diff statistics

				for j in range(P):

					# Initial activity

					if j == cat:								Y_dots_th[0,cat_idx,j] += (integ.PsiSq(gain, offset)-integ.Psi(gain, offset)**2)
					elif neighbours[cat].count(j) == 1:  		Y_dots_th[0,cat_idx,j] += (integ.PsiPsi(gain, offset)-integ.Psi(gain, offset)**2)
					else:										Y_dots_th[0,cat_idx,j] += 0

					if j == cat:								Y_dots_th[1,cat_idx,j] += (integ.PsiSq(gain, offset)-integ.Psi(gain, offset)**2)
					elif neighbours[cat].count(j) == 1:  		Y_dots_th[1,cat_idx,j] += (integ.PsiPsi(gain, offset)-integ.Psi(gain, offset)**2)
					else:										Y_dots_th[1,cat_idx,j] += 0

					# Activity changes

					for k in neighbours_plus[cat]:
						for kk in neighbours_plus[j]:

							dot_ = (xdotx[cat,k] * xdotx[j,kk])*(c[k]*c[kk]) / N
							dot = 0


							# PART A: k = cat, kk = j

							if ( (k == cat) and (kk==j) ):

								if j == cat:							dot = dot_*integ.PsiPrimeFourth(gain, offset) 
								elif neighbours[cat].count(j) == 1:  	dot = dot_*integ.PsiPrimeSqPsiPrimeSq(gain, offset) 
								else:									dot = dot_*integ.PsiPrimeSq(gain, offset)**2 


							# PART B: k = cat, kk neq j, and viceversa

							elif ( (k == cat) and (kk != j) ):

								if j == cat: 																dot = dot_*Int_11p3_12

								elif neighbours[cat].count(j) == 1:

									if kk == cat:															dot = dot_*Int_11p3_12
									else:
										if ( (S_[cat] == S_[j] == S_[kk]) or (C_[cat] == C_[j] == C_[kk])): dot = dot_*Int_11p2_12_13
										else:																dot = dot_*Int_11p2_12_22

								else:
									if ( (C_[kk] == C_[cat]) or (S_[kk] == S_[cat]) ): 						dot = dot_*Int_11p2_12_22
									else: 																	dot = dot_*pp1*pp2


							elif ( (k != cat) and (kk == j) ):

								if j == cat: 																dot = dot_*Int_11p3_12

								elif neighbours[cat].count(j) == 1:

									if k == j:																dot = dot_*Int_11p3_12
									else:
										if ( (S_[j] == S_[cat] == S_[k]) or (C_[j] == C_[cat] == C_[k])):   dot = dot_*Int_11p2_12_13
										else:																dot = dot_*Int_11p2_12_22

								else:
									if ( (C_[k] == C_[j]) or (S_[k] == S_[j]) ): 							dot = dot_*Int_11p2_12_22
									else: 																	dot = dot_*pp1*pp2

					# PART C: k neq cat, kk neq

							else:

								if j == cat:

									if k == kk: 																dot = dot_*integ.PsiPrimeSqPsiPrimeSq(gain, offset) 
									else:
										if ( (S_[k] == S_[kk] == S_[cat]) or (C_[k] == C_[kk] == C_[cat]) ) :	dot = dot_*Int_11p2_12_13
										else: 																	dot = dot_*Int_11p2_12_21

								#

								elif neighbours[cat].count(j) == 1:

									if k == kk: 																dot = dot_*Int_11p2_12_13

									elif (k == j and kk == cat): 												dot = dot_*integ.PsiPrimeSqPsiPrimeSq(gain, offset) 

									elif k == j:
										if ( (S_[kk] == S_[j] == S_[cat]) or (C_[kk] == C_[j] == C_[cat]) ):	dot = dot_*Int_11p2_12_13
										else: 																	dot = dot_*Int_11p2_12_21

									elif kk == cat:
										if ( (S_[k] == S_[j] == S_[cat]) or (C_[k] == C_[j] == C_[cat]) ):		dot = dot_*Int_11p2_12_13
										else: 																	dot = dot_*Int_11p2_12_21

									else:

										if C_[cat] == C_[j]: 
											if C_[k] == C_[kk] == C_[cat]:										dot = dot_*Int_11_12_13_14
											elif ( (C_[k] == C_[cat]) or (C_[kk] == C_[cat]) ):					dot = dot_*Int_11_12_13_23
											else:
												if C_[kk] == C_[k]:												dot = dot_*Int_11_12_21_22
												else:															dot = dot_*Int_11_12_22_23

										else: 
											if S_[k] == S_[kk] == S_[cat]:										dot = dot_*Int_11_12_13_14
											elif ( (S_[k] == S_[cat]) or (S_[kk] == S_[cat]) ):					dot = dot_*Int_11_12_13_23
											else:
												if S_[kk] == S_[k]:												dot = dot_*Int_11_12_21_22
												else:															dot = dot_*Int_11_12_22_23

								#

								else:

									if k == kk: 																		dot = dot_*Int_11p2_12_21

									else:

										if ( (C_[cat] == C_[k]) and (C_[j] == C_[kk])):
											if ( (S_[cat] == S_[kk]) and (S_[j] == S_[k])):  							dot = dot_*Int_11_12_21_22
											elif ( (S_[cat] == S_[kk]) or (S_[j] == S_[k]) or (S_[kk] == S_[k]) ):		dot = dot_*Int_11_12_22_23
											else:																		dot = dot_*pp2**2

										elif ( (S_[cat] == S_[k]) and (S_[j] == S_[kk])):
											if ( (C_[cat] == C_[kk]) and (C_[j] == C_[k])):  							dot = dot_*Int_11_12_21_22
											elif ( (C_[cat] == C_[kk]) or (C_[j] == C_[k]) or (C_[kk] == C_[k]) ):		dot = dot_*Int_11_12_22_23
											else:																		dot = dot_*pp2**2

										else:
											if ( (C_[k] == C_[kk]) or (S_[k] == S_[kk]) ):								dot = dot_*Int_11_12_13_23
											else:																		dot = dot_*pp2**2


							Y_dots_th[1,cat_idx,j] += dot
							# if dot == 0: print 'Combinatorics prob'


			# Divide by std

			Y_corrs_th[0,:,:] = Y_dots_th[0,:,:] / Y_dots_th[0,0,0]

			std = np.sqrt(Y_dots_th[1,1,int(Q/2)]) * np.ones( P )
			std[np.where(labels[:,0] == net.fh)[0]] = np.sqrt(Y_dots_th[1,0,0])

			Y_corrs_th[1,0,:] = Y_dots_th[1,0,:] / std / np.sqrt(Y_dots_th[1,0,0])
			Y_corrs_th[1,1,:] = Y_dots_th[1,1,:] / std / np.sqrt(Y_dots_th[1,1,int(Q/2)])

			#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
			# Activity measures

			# Compute correlation

			Y_corr_cat_th[0,i,ii] = an.Corr_th(Y_corrs_th[0,:,:], o_)
			Y_corr_ctx_th[0,i,ii] = an.Corr_th(Y_corrs_th[0,:,:], c_)

			Y_corr_cat_th[1,i,ii] = an.Corr_th(Y_corrs_th[1,:,:], o_)
			Y_corr_ctx_th[1,i,ii] = an.Corr_th(Y_corrs_th[1,:,:], c_)

	# Store 

	fac.Store(K_corr_cat_th, 'K_corr_cat_th.p', 'Results/')
	fac.Store(K_corr_ctx_th, 'K_corr_ctx_th.p', 'Results/')

	fac.Store(Y_corr_cat_th, 'Y_corr_cat_th.p', 'Results/')
	fac.Store(Y_corr_ctx_th, 'Y_corr_ctx_th.p', 'Results/')

else:

	# Retrieve

	K_corr_cat_th = fac.Retrieve('K_corr_cat_th.p', 'Results/')
	K_corr_ctx_th = fac.Retrieve('K_corr_ctx_th.p', 'Results/')

	Y_corr_cat_th = fac.Retrieve('Y_corr_cat_th.p', 'Results/')
	Y_corr_ctx_th = fac.Retrieve('Y_corr_ctx_th.p', 'Results/')


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
### Plots

Y_corr_ctx_th = Y_corr_ctx_th[1,:,:] - Y_corr_ctx_th[0,:,:]

#

fac.SetPlotDim(2.1, 1.8)
dashes = [3,3]

cmap_base =  'bwr' 
vmin, vmax = 0.23, 0.77
cmap = fac.TruncateCmap(cmap_base, vmin, vmax)

#

fg = plt.figure()
ax0 = plt.axes(frameon=True)

cax = plt.imshow(Y_corr_ctx_th.T, vmin = -0.1, vmax = 0.1, aspect = 'auto', \
	extent = (min(gain_phi_values)/4., max(gain_phi_values)/4., min(offset_phi_values), max(offset_phi_values)), origin='lower', interpolation='nearest', cmap = cmap)

cbar = fg.colorbar(cax, ticks=[-0.1, 0, 0.1], orientation='vertical')
cbar.ax.set_xticklabels(['-0.1', '0', '0.1'])

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

plt.savefig('Y_corr_avg.pdf')
plt.show()

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####

sys.exit(0)
