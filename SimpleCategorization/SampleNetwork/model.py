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
import fct_analysis as an

fac.SetPlotParams()


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
### Parameters

# Circuit

N = 200   # Number of neurons
P = 20    # Number of stimuli

# Training

eta = 0    # Ratio eta_w/eta_u
eta_u = 0.1 
eta_w = eta * eta_u

Nepochs = int(10000)
Nskip = int(Nepochs/20.)
Nsample = N


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
### Circuit

doTraining = 0


if doTraining:

	# Generate model

	model = net.BuildModel(N, N, 1)

	nn.init.normal_(model.U.weight, 0, 1./np.sqrt(model.N1))
	nn.init.normal_(model.W.weight, 0, 1./np.sqrt(model.N2))
	w0 = np.copy(model.W.weight.data.numpy())

	# Input generating vectors

	mu = np.random.normal(0, 1., (P, N))

	# Training setting

	criterion = nn.MSELoss()
	optimizer = torch.optim.SGD([ {'params': model.U.parameters(), 'lr': eta_u},
	                			  {'params': model.W.parameters(), 'lr': eta_w} ])

	x_, S_, o_, labels = net.Batch_SimpleCat(mu)

	x_ = torch.from_numpy(x_).float()				# Convert to tensor
	labels = torch.from_numpy(labels).float()


	# Measures

	loss_evolution = np.zeros(int(Nepochs/Nskip))

	Y_ = np.zeros(( 2, P, Nsample ))

	Y_average = np.zeros(( 2, N, 2 ))
	Y_selectivity = np.zeros(( 2, N ))


	#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
	### Train

	model.train()

	for epoch in range(Nepochs):

		#### GD

		k_, y_, z_ = model(x_)			# Forward pass
		loss = criterion(z_, labels)	# Compute loss

		optimizer.zero_grad()			# Zero the gradients

		loss.backward()					# Backward pass
		optimizer.step()				# Update the parameters


		#### Analysis

		if epoch == 0:

			# Take initial measures

			Y_[0,:,:] = y_.detach().numpy()[:,0:Nsample]

			Y_selectivity[0,:] = an.SI(y_.detach().numpy(), o_, average = 0)

			Y_average[0,:,0] = np.mean(y_.detach().numpy()[0:int(P/2.),:], 0)
			Y_average[0,:,1] = np.mean(y_.detach().numpy()[int(P/2.):,:], 0)


		if epoch % Nskip == 0:

			print('epoch: ', epoch,' loss: ', loss.item())

			# Save loss

			loss_evolution[int(epoch/Nskip)] = loss.item()


	# Take final measures

	Y_[1,:,:] = y_.detach().numpy()[:,0:Nsample]

	Y_selectivity[1,:] = an.SI(y_.detach().numpy(), o_, average = 0)

	Y_average[1,:,0] = np.mean(y_.detach().numpy()[0:int(P/2.),:], 0)
	Y_average[1,:,1] = np.mean(y_.detach().numpy()[int(P/2.):,:], 0)


	# Store

	fac.Store(loss_evolution, 'loss_evolution.p', 'Data/')
	fac.Store(w0, 'w0.p', 'Data/')
	fac.Store(x_.detach().numpy(), 'x_.p', 'Data/')

	fac.Store(Y_, 'Y_.p', 'Data/')
	fac.Store(Y_selectivity, 'Y_selectivity.p', 'Data/')
	fac.Store(Y_average, 'Y_average.p', 'Data/')

else:

	# Retrieve

	loss_evolution = fac.Retrieve('loss_evolution.p', 'Data/')
	w0 = fac.Retrieve('w0.p', 'Data/')

	Y_ = fac.Retrieve('Y_.p', 'Data/')
	Y_selectivity = fac.Retrieve('Y_selectivity.p', 'Data/')
	Y_average = fac.Retrieve('Y_average.p', 'Data/')


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
### Plots

dashes = [3,3]

#

fac.SetPlotDim(2.5, 1.8)

fg = plt.figure()
ax0 = plt.axes(frameon=True)

t = Nskip*np.arange(0, int(Nepochs/Nskip))

plt.plot( t, loss_evolution, 'k' )

plt.xlabel(r'Epoch')
plt.ylabel(r'MSE loss')

plt.xlim(0, Nepochs)

ax0.spines['top'].set_visible(False)
ax0.spines['right'].set_visible(False)
ax0.yaxis.set_ticks_position('left')
ax0.xaxis.set_ticks_position('bottom')
plt.locator_params(nbins=4)

plt.savefig('loss.pdf')
plt.show()

#

fac.SetPlotDim(1.85, 1.5)

fg = plt.figure()
ax0 = plt.axes(frameon=True)

cmap_base =  'bwr' 
vmin, vmax = 0.2, 0.8
cmap = fac.TruncateCmap(cmap_base, vmin, vmax)

mu = np.random.normal(0, 1., (P, N)) # Temporary trick
x_, S_, o_, labels = net.Batch_SimpleCat(mu)

cax = plt.pcolor(np.corrcoef(x_), vmin=-0.6, vmax=0.6, cmap = cmap)

plt.axvline(x=10, color = '0', ls = '-', linewidth = 0.7)
plt.axhline(y=10, color = '0', ls = '-', linewidth = 0.7)

cbar = fg.colorbar(cax, ticks=[-0.6, 0, 0.6], orientation='vertical')
cbar.ax.set_xticklabels(['-0.5', '0', '0.5'])

# plt.xlabel('Stimuli')
# plt.ylabel('Stimuli')

plt.grid('off')

plt.xticks([1, 20])
plt.yticks([1, 20])

plt.gca().invert_yaxis()

ax0.spines['top'].set_visible(False)
ax0.spines['right'].set_visible(False)
ax0.yaxis.set_ticks_position('left')
ax0.xaxis.set_ticks_position('bottom')
# plt.locator_params(nbins=4)

plt.savefig('x_corr_matrix.pdf')

plt.show()

#

fg = plt.figure()
ax0 = plt.axes(frameon=True)

cax = plt.pcolor(np.corrcoef(Y_[0,:,:]), vmin=-0.6, vmax=0.6, cmap = cmap)

plt.axvline(x=10, color = '0', ls = '-', linewidth = 0.7)
plt.axhline(y=10, color = '0', ls = '-', linewidth = 0.7)

cbar = fg.colorbar(cax, ticks=[-0.6, 0, 0.6], orientation='vertical')
cbar.ax.set_xticklabels(['-0.5', '0', '0.5'])

plt.grid('off')

plt.xticks([1, 20])
plt.yticks([1, 20])

plt.gca().invert_yaxis()

ax0.spines['top'].set_visible(False)
ax0.spines['right'].set_visible(False)
ax0.yaxis.set_ticks_position('left')
ax0.xaxis.set_ticks_position('bottom')
# plt.locator_params(nbins=4)

plt.savefig('y0_corr_matrix.pdf')

plt.show()

#

fg = plt.figure()
ax0 = plt.axes(frameon=True)

cax = plt.pcolor(np.corrcoef(Y_[1,:,:]), vmin=-0.6, vmax=0.6, cmap = cmap)

plt.axvline(x=10, color = '0', ls = '-', linewidth = 0.7)
plt.axhline(y=10, color = '0', ls = '-', linewidth = 0.7)

cbar = fg.colorbar(cax, ticks=[-0.6, 0, 0.6], orientation='vertical')
cbar.ax.set_xticklabels(['-0.5', '0', '0.5'])

plt.grid('off')

plt.xticks([1, 20])
plt.yticks([1, 20])

plt.gca().invert_yaxis()

ax0.spines['top'].set_visible(False)
ax0.spines['right'].set_visible(False)
ax0.yaxis.set_ticks_position('left')
ax0.xaxis.set_ticks_position('bottom')
# plt.locator_params(nbins=4)

plt.savefig('y_corr_matrix.pdf')

plt.show()

#

fac.SetPlotDim(0.9, 1.35)

fg = plt.figure()
ax0 = plt.axes(frameon=True)

av_corr_0 = np.average(np.corrcoef(Y_[0,:,:])[0:int(P/2),int(P/2):])

plt.axhline(y=0, color = '0', ls = '-', linewidth = 0.7)
# plt.axhline(y=av_corr_0, color = '#FFD012', ls = '-', linewidth = 0.7)

plt.plot(0, av_corr_0, 'o', markersize = 4, color = '0.4')

plt.grid('off')

plt.xlim(-1, 1)
plt.ylim(-0.4, 0.4)

plt.xticks([])
plt.yticks([-0.4, 0, 0.4])

ax0.spines['bottom'].set_visible(False)
ax0.spines['top'].set_visible(False)
ax0.spines['right'].set_visible(False)
ax0.yaxis.set_ticks_position('left')
ax0.xaxis.set_ticks_position('bottom')

plt.savefig('y0_corr_summary.pdf')

plt.show()


#

fg = plt.figure()
ax0 = plt.axes(frameon=True)

av_corr = np.average(np.corrcoef(Y_[1,:,:])[0:int(P/2),int(P/2):])

plt.axhline(y=0, color = '0', ls = '-', linewidth = 0.7)
# plt.axhline(y=av_corr_0, color = '0', ls = '-', linewidth = 0.7)
plt.axhline(y=av_corr, color = '0.4', ls = '--', linewidth = 0.7)

plt.plot(0, av_corr, 'o', markersize = 4, color = '0.4')

plt.grid('off')

plt.xlim(-1, 1)
plt.ylim(-0.4, 0.4)

plt.xticks([])
plt.yticks([-0.4, 0, 0.4])

ax0.spines['bottom'].set_visible(False)
ax0.spines['top'].set_visible(False)
ax0.spines['right'].set_visible(False)
ax0.yaxis.set_ticks_position('left')
ax0.xaxis.set_ticks_position('bottom')

plt.savefig('y_corr_summary.pdf')

plt.show()

#

fac.SetPlotDim(2., 1.65)

fg = plt.figure()
ax0 = plt.axes(frameon=True)

# plt.axvline(x = np.mean(Y_selectivity[0,:]), ls = '-', color = '0', linewidth = 0.7)

plt.hist(Y_selectivity[0,:], color = '0.85', normed = True, bins = 10)

plt.axvline(x = np.mean(Y_selectivity[0,:]), ls = '--', color = '0.4', linewidth = 0.7)
plt.plot(np.mean(Y_selectivity[0,:]), 0.4, '^', markersize = 4, color = '0.4', label = 'pop. average')

plt.xlabel(r'Cat selectivity index')
plt.ylabel(r'Neurons count')

plt.xlim(-0.1, 0.8)
plt.xticks([ 0., 0.4, 0.8])

plt.ylim(0, 12)
plt.yticks([0, 6, 12])

plt.legend(frameon = False)

ax0.spines['top'].set_visible(False)
ax0.spines['right'].set_visible(False)
ax0.yaxis.set_ticks_position('left')
ax0.xaxis.set_ticks_position('bottom')
# plt.locator_params(nbins=4)

plt.savefig('SI_0.pdf')
plt.show()

#

fg = plt.figure()
ax0 = plt.axes(frameon=True)

plt.axvline(x = np.mean(Y_selectivity[0,:]), ls = '-', color = '0', linewidth = 0.7)

plt.hist(Y_selectivity[1,:], color = '0.85', normed = True, bins = 20)

plt.axvline(x = np.mean(Y_selectivity[1,:]), ls = '--', color = '0.4', linewidth = 0.7)
plt.plot(np.mean(Y_selectivity[1,:]), 0.4, '^', markersize = 4, color = '0.4')

plt.xlabel(r'Cat selectivity index')
plt.ylabel(r'Neurons count')

plt.xlim(-0.1, 0.8)
plt.xticks([ 0., 0.4, 0.8])

plt.ylim(0, 12)
plt.yticks([0, 6, 12])

ax0.spines['top'].set_visible(False)
ax0.spines['right'].set_visible(False)
ax0.yaxis.set_ticks_position('left')
ax0.xaxis.set_ticks_position('bottom')
# plt.locator_params(nbins=4)

plt.savefig('SI_T.pdf')
plt.show()


#

fac.SetPlotDim(2., 1.8)

fg = plt.figure()
ax0 = plt.axes(frameon=True)

Nmax = N

plt.plot([0, 1.2], [0, 1.2], color = '0.8', linewidth = 0.7)

log_fit = np.polyfit( Y_average[0,:,0] , Y_average[0,:,1], 1)
x_range = np.linspace(0,1.2,100)
corr_coeff = np.corrcoef(Y_average[0,:,0], Y_average[0,:,1])

plt.plot( Y_average[0,:Nmax,0], Y_average[0,:Nmax,1], 'o', color = '0.8', markeredgecolor = '1', markeredgewidth = '0.2' )
line, = plt.plot(x_range, log_fit[1] + x_range*log_fit[0], '-', color='0.4', label=r'$'+str(round(corr_coeff[0,1],2))+'$' )

plt.xlabel('Average activity $\mathsf{A}$')
plt.ylabel('Average activity $\mathsf{B}$')

plt.xlim(0,0.8)
plt.ylim(0,0.8)

plt.xticks([0, 0.4, 0.8])
plt.yticks([0, 0.4, 0.8])

ax0.spines['top'].set_visible(False)
ax0.spines['right'].set_visible(False)
ax0.yaxis.set_ticks_position('left')
ax0.xaxis.set_ticks_position('bottom')
plt.locator_params(nbins=3)

plt.legend(frameon=False, loc = 2)

plt.savefig('Y_average_0.pdf')

plt.show()

#

fg = plt.figure()
ax0 = plt.axes(frameon=True)

plt.plot([0, 1.2], [0, 1.2], color = '0.8', linewidth = 0.7)

log_fit = np.polyfit( Y_average[1,:,0] , Y_average[1,:,1], 1)
corr_coeff = np.corrcoef(Y_average[1,:,0], Y_average[1,:,1])

plt.plot( Y_average[1,:Nmax,0], Y_average[1,:Nmax,1], 'o', color = '0.8', markeredgecolor = '1', markeredgewidth = '0.2' )
line, = plt.plot(x_range, log_fit[1] + x_range*log_fit[0], '-', color='0.4', label=r'$'+str(round(corr_coeff[0,1],2))+'$' )

plt.xlabel('Average activity $\mathsf{A}$')
plt.ylabel('Average activity $\mathsf{B}$')

plt.xlim(0,0.8)
plt.ylim(0,0.8)

plt.xticks([0, 0.4, 0.8])
plt.yticks([0, 0.4, 0.8])

ax0.spines['top'].set_visible(False)
ax0.spines['right'].set_visible(False)
ax0.yaxis.set_ticks_position('left')
ax0.xaxis.set_ticks_position('bottom')
plt.locator_params(nbins=3)

plt.legend(frameon=False, loc = 2)

plt.savefig('Y_average_T.pdf')

plt.show()

#

fg = plt.figure()
ax0 = plt.axes(frameon=True)

log_fit = np.polyfit( np.fabs(w0[0,:]) , Y_selectivity[0,:], 1)
x_range = np.linspace(-1,1,100)
corr_coeff = np.corrcoef(np.fabs(w0[0,:]), Y_selectivity[0,:])

plt.plot( np.fabs(w0[0,:]), Y_selectivity[0,:], 'o', color = '#AEEAEC', markeredgecolor = '1', markeredgewidth = '0.2', markersize = 2.8 )
line, = plt.plot(x_range, log_fit[1] + x_range*log_fit[0], '-', color='#00CED1', label=r'$'+str(round(corr_coeff[0,1],2))+'$' )

plt.xlabel('Initial readout connectivity')
plt.ylabel('Cat selectivity index')

plt.xlim(0, 0.3)
plt.xticks([0, 0.1, 0.2, 0.3])

plt.ylim(-0.08, 0.8)
plt.yticks([0, 0.4, 0.8])

ax0.spines['top'].set_visible(False)
ax0.spines['right'].set_visible(False)
ax0.yaxis.set_ticks_position('left')
ax0.xaxis.set_ticks_position('bottom')
plt.locator_params(nbins=4)
plt.legend(frameon=False, loc=1)

# plt.colorbar()

plt.savefig('SI_w0_0.pdf')

plt.show()

#


fg = plt.figure()
ax0 = plt.axes(frameon=True)

log_fit = np.polyfit( np.fabs(w0[0,:]) , Y_selectivity[1,:], 1)
corr_coeff = np.corrcoef(np.fabs(w0[0,:]), Y_selectivity[1,:])

plt.plot( np.fabs(w0[0,:]), Y_selectivity[1,:], 'o', color = '#AEEAEC', markeredgecolor = '1', markeredgewidth = '0.2', markersize = 2.8 )
line, = plt.plot(x_range, log_fit[1] + x_range*log_fit[0], '-', color='#00CED1', label=r'$'+str(round(corr_coeff[0,1],2))+'$' )

plt.xlabel('Initial readout connectivity')
plt.ylabel('Cat selectivity index')

plt.xlim(0, 0.3)
plt.xticks([0, 0.1, 0.2, 0.3])

plt.ylim(-0.08, 0.8)
plt.yticks([0, 0.4, 0.8])

ax0.spines['top'].set_visible(False)
ax0.spines['right'].set_visible(False)
ax0.yaxis.set_ticks_position('left')
ax0.xaxis.set_ticks_position('bottom')
plt.locator_params(nbins=4)
plt.legend(frameon=False, loc=1)

# plt.colorbar()

plt.savefig('SI_w0_T.pdf')

plt.show()

#

fac.SetPlotDim(1.1, 1.1)

nrn_idx = np.where(Y_selectivity[1,:]>0.5)[0]

colorA = '#E2205C'
colorB = '#1E83BA'

colorA = '#FF881D'
colorB = '#1E83BA'

'''
for jj, j in enumerate(nrn_idx):

	fg = plt.figure()
	ax0 = plt.axes(frameon=True)

	plt.hist(Y_[0,:int(P/2.),j], color = '1', normed = True, bins = 5, alpha = 1, edgecolor=colorA, linewidth=1.)
	plt.hist(Y_[0,int(P/2.):,j], color = '1', normed = True, bins = 5, alpha = 1, edgecolor=colorB, linewidth=1.)

	plt.plot(np.mean(Y_[0,:,:]), 0.4, '^', markersize = 4, color = '#009999')

	ax0.spines['top'].set_visible(False)
	ax0.spines['right'].set_visible(False)
	ax0.spines['left'].set_visible(False)
	ax0.yaxis.set_ticks_position('left')
	ax0.xaxis.set_ticks_position('bottom')
	# plt.locator_params(nbins=4)

	plt.xlim(0, 1)
	plt.xticks([0, 1])

	plt.yticks([])

	plt.xlabel('Activity')
	# plt.ylabel('Frequency')

	plt.savefig('Samples/Y_'+str(j)+'_0.pdf')
	plt.show()

	#

	fg = plt.figure()
	ax0 = plt.axes(frameon=True)

	plt.hist(Y_[1,:int(P/2.),j], color = '1', normed = True, bins = 8, alpha = 1, edgecolor=colorA, linewidth=1.)
	plt.hist(Y_[1,int(P/2.):,j], color = '1', normed = True, bins = 8, alpha = 1, edgecolor=colorB, linewidth=1.)

	plt.plot(np.mean(Y_[1,:,:]), 0.4, '^', markersize = 4, color = '#009999')

	ax0.spines['top'].set_visible(False)
	ax0.spines['right'].set_visible(False)
	ax0.spines['left'].set_visible(False)
	ax0.yaxis.set_ticks_position('left')
	ax0.xaxis.set_ticks_position('bottom')
	# plt.locator_params(nbins=4)

	plt.xlim(0, 1)
	plt.xticks([0, 1])

	plt.yticks([])

	plt.xlabel('Activity')
	# plt.ylabel('Frequency')

	plt.savefig('Samples/Y_'+str(j)+'_T.pdf')
	plt.show()
'''
#

j = 66

fg = plt.figure()
ax0 = plt.axes(frameon=True)

plt.hist(Y_[0,:int(P/2.),j], bins=np.linspace(0, 1, 15, endpoint = False), normed = True, alpha = 0.2, facecolor=colorA, edgecolor='None')
plt.hist(Y_[0,:int(P/2.),j], bins=np.linspace(0, 1, 15, endpoint = False), normed = True, alpha = 1, edgecolor=colorA, linewidth=0.7, facecolor='None', histtype=u'step')

plt.hist(Y_[0,int(P/2.):,j], bins=np.linspace(0, 1, 15, endpoint = False), normed = True, alpha = 0.2, facecolor=colorB, edgecolor='None')
plt.hist(Y_[0,int(P/2.):,j], bins=np.linspace(0, 1, 15, endpoint = False), normed = True, alpha = 1, edgecolor=colorB, linewidth=0.7, facecolor='None', histtype=u'step')

# plt.hist(Y_[0,:int(P/2.),j], bins=np.linspace(0, 1, 15, endpoint = False), normed = True, alpha = 1, edgecolor=colorA, linewidth=0.7, facecolor='None', histtype=u'step')
# plt.hist(Y_[0,int(P/2.):,j], bins=np.linspace(0, 1, 15, endpoint = False), normed = True, alpha = 1, edgecolor=colorB, linewidth=0.7, facecolor='None', histtype=u'step')

plt.axvline(x = np.mean(Y_[0,:,:]), ls = '--', color = '0.4', linewidth = 0.7)
# plt.plot(np.mean(Y_[0,:,:]), 0.4, '^', markersize = 4, color = '0.4')

ax0.spines['top'].set_visible(False)
ax0.spines['right'].set_visible(False)
# ax0.spines['left'].set_visible(False)
ax0.yaxis.set_ticks_position('left')
ax0.xaxis.set_ticks_position('bottom')
# plt.locator_params(nbins=4)

plt.xlim(-0.05, 1)
plt.xticks([0, 1])

plt.ylim(0, 10)
plt.yticks([0, 10])

plt.xlabel('Activity')
# plt.ylabel('Frequency')

plt.savefig('Samples/Y_a_0.pdf')
plt.show()

#

fg = plt.figure()
ax0 = plt.axes(frameon=True)

plt.hist(Y_[1,:int(P/2.),j], bins=np.linspace(0, 1, 15, endpoint = False), normed = True, alpha = 0.2, facecolor=colorA, edgecolor='None')
plt.hist(Y_[1,:int(P/2.),j], bins=np.linspace(0, 1, 15, endpoint = False), normed = True, alpha = 1, edgecolor=colorA, linewidth=0.7, facecolor='None', histtype=u'step')

plt.hist(Y_[1,int(P/2.):,j], bins=np.linspace(0, 1, 15, endpoint = False), normed = True, alpha = 0.2, facecolor=colorB, edgecolor='None')
plt.hist(Y_[1,int(P/2.):,j], bins=np.linspace(0, 1, 15, endpoint = False), normed = True, alpha = 1, edgecolor=colorB, linewidth=0.7, facecolor='None', histtype=u'step')

# plt.hist(Y_[1,:int(P/2.),j], bins=np.linspace(0, 1, 15, endpoint = False), normed = True, alpha = 1, edgecolor=colorA, linewidth=0.7, facecolor='None', histtype=u'step')
# plt.hist(Y_[1,int(P/2.):,j], bins=np.linspace(0, 1, 15, endpoint = False), normed = True, alpha = 1, edgecolor=colorB, linewidth=0.7, facecolor='None', histtype=u'step')

plt.axvline(x = np.mean(Y_[1,:,:]), ls = '--', color = '0.4', linewidth = 0.7)
# plt.plot(np.mean(Y_[1,:,:]), 0.4, '^', markersize = 4, color = '0.4')

ax0.spines['top'].set_visible(False)
ax0.spines['right'].set_visible(False)
# ax0.spines['left'].set_visible(False)
ax0.yaxis.set_ticks_position('left')
ax0.xaxis.set_ticks_position('bottom')
# plt.locator_params(nbins=4)

plt.xlim(-0.05, 1)
plt.xticks([0, 1])

plt.ylim(0, 12)
plt.yticks([0, 12])

plt.xlabel('Activity')
# plt.ylabel('Frequency')

plt.savefig('Samples/Y_a_T.pdf')
plt.show()

#


j = 49

fg = plt.figure()
ax0 = plt.axes(frameon=True)

plt.hist(Y_[0,:int(P/2.),j], bins=np.linspace(0, 1, 15, endpoint = False), normed = True, alpha = 0.2, facecolor=colorA, edgecolor='None')
plt.hist(Y_[0,:int(P/2.),j], bins=np.linspace(0, 1, 15, endpoint = False), normed = True, alpha = 1, edgecolor=colorA, linewidth=0.7, facecolor='None', histtype=u'step')

plt.hist(Y_[0,int(P/2.):,j], bins=np.linspace(0, 1, 15, endpoint = False), normed = True, alpha = 0.2, facecolor=colorB, edgecolor='None')
plt.hist(Y_[0,int(P/2.):,j], bins=np.linspace(0, 1, 15, endpoint = False), normed = True, alpha = 1, edgecolor=colorB, linewidth=0.7, facecolor='None', histtype=u'step')

# plt.hist(Y_[0,:int(P/2.),j], bins=np.linspace(0, 1, 15, endpoint = False), normed = True, alpha = 1, edgecolor=colorA, linewidth=0.7, facecolor='None', histtype=u'step')
# plt.hist(Y_[0,int(P/2.):,j], bins=np.linspace(0, 1, 15, endpoint = False), normed = True, alpha = 1, edgecolor=colorB, linewidth=0.7, facecolor='None', histtype=u'step')

plt.axvline(x = np.mean(Y_[0,:,:]), ls = '--', color = '0.4', linewidth = 0.7)
# plt.plot(np.mean(Y_[0,:,:]), 0.4, '^', markersize = 4, color = '0.4')

ax0.spines['top'].set_visible(False)
ax0.spines['right'].set_visible(False)
# ax0.spines['left'].set_visible(False)
ax0.yaxis.set_ticks_position('left')
ax0.xaxis.set_ticks_position('bottom')
# plt.locator_params(nbins=4)

plt.xlim(-0.05, 1)
plt.xticks([0, 1])

plt.ylim(0, 10)
plt.yticks([0, 10])

plt.xlabel('Activity')
# plt.ylabel('Frequency')

plt.savefig('Samples/Y_b_0.pdf')
plt.show()

#

fg = plt.figure()
ax0 = plt.axes(frameon=True)

plt.hist(Y_[1,:int(P/2.),j], bins=np.linspace(0, 1, 15, endpoint = False), normed = True, alpha = 0.2, facecolor=colorA, edgecolor='None')
plt.hist(Y_[1,:int(P/2.),j], bins=np.linspace(0, 1, 15, endpoint = False), normed = True, alpha = 1, edgecolor=colorA, linewidth=0.7, facecolor='None', histtype=u'step')

plt.hist(Y_[1,int(P/2.):,j], bins=np.linspace(0, 1, 15, endpoint = False), normed = True, alpha = 0.2, facecolor=colorB, edgecolor='None')
plt.hist(Y_[1,int(P/2.):,j], bins=np.linspace(0, 1, 15, endpoint = False), normed = True, alpha = 1, edgecolor=colorB, linewidth=0.7, facecolor='None', histtype=u'step')

# plt.hist(Y_[1,:int(P/2.),j], bins=np.linspace(0, 1, 15, endpoint = False), normed = True, alpha = 1, edgecolor=colorA, linewidth=0.7, facecolor='None', histtype=u'step')
# plt.hist(Y_[1,int(P/2.):,j], bins=np.linspace(0, 1, 15, endpoint = False), normed = True, alpha = 1, edgecolor=colorB, linewidth=0.7, facecolor='None', histtype=u'step')

plt.axvline(x = np.mean(Y_[1,:,:]), ls = '--', color = '0.4', linewidth = 0.7)
# plt.plot(np.mean(Y_[1,:,:]), 0.4, '^', markersize = 4, color = '0.4')

ax0.spines['top'].set_visible(False)
ax0.spines['right'].set_visible(False)
# ax0.spines['left'].set_visible(False)
ax0.yaxis.set_ticks_position('left')
ax0.xaxis.set_ticks_position('bottom')
# plt.locator_params(nbins=4)

plt.xlim(-0.05, 1)
plt.xticks([0, 1])

plt.ylim(0, 12)
plt.yticks([0, 12])

plt.xlabel('Activity')
# plt.ylabel('Frequency')

plt.savefig('Samples/Y_b_T.pdf')
plt.show()

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####

sys.exit(0)
