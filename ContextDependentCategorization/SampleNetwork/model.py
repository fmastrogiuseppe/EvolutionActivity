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
import fct_analysis as an
import fct_integrals as integ

fac.SetPlotParams()


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
### Parameters

# Circuit

N = 600   # Number of neurons
Q = 8   # Number of stimuli 
P = Q**2

# Training

eta = 0.    # Ratio eta_w/eta_u
eta_u = 0.5
eta_w = eta * eta_u

Nepochs = int(40000)
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
	u0 = np.copy(model.U.weight.data.numpy())

	# Input generating vectors

	mu = np.random.normal(0, 1., (Q, N))
	nu = np.random.normal(0, 1., (Q, N))

	# Training setting

	criterion = nn.MSELoss()
	optimizer = torch.optim.SGD([ {'params': model.U.parameters(), 'lr': eta_u},
	                			  {'params': model.W.parameters(), 'lr': eta_w} ])

	x_, S_, C_, c_, o_, labels = net.Batch_ContextCat(mu, nu)

	x_ = torch.from_numpy(x_).float()				# Convert to tensor
	labels = torch.from_numpy(labels).float()


	# Measures

	loss_evolution = np.zeros(int(Nepochs/Nskip))

	Y_ = np.zeros(( 2, P, Nsample ))
	K_ = np.zeros(( 2, P, Nsample ))

	Y_average_cat = np.zeros(( 2, N, 2 ))
	Y_average_ctx = np.zeros(( 2, N, 2 ))

	Y_selectivity_cat = np.zeros(( 2, N ))
	Y_selectivity_ctx = np.zeros(( 2, N ))


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


		if epoch == 0:

			# Take initial measures

			Y_[0,:,:] = y_.detach().numpy()[:,0:Nsample]
			K_[0,:,:] = k_.detach().numpy()[:,0:Nsample]

			Y_selectivity_cat[0,:] = an.SI(y_.detach().numpy(), o_, average = 0)
			Y_selectivity_ctx[0,:] = an.SI(y_.detach().numpy(), c_, average = 0, exclude = 1, indeces_eclude = C_)

			Y_average_cat[0,:,1] = np.mean(y_.detach().numpy()[o_.astype(bool),:], 0)
			Y_average_cat[0,:,0] = np.mean(y_.detach().numpy()[np.logical_not(o_.astype(bool)),:], 0)
			
			Y_average_ctx[0,:,1] = np.mean(y_.detach().numpy()[c_.astype(bool),:], 0)
			Y_average_ctx[0,:,0] = np.mean(y_.detach().numpy()[np.logical_not(c_.astype(bool)),:], 0)

		#### Analysis

		print epoch, loss.item()
			
		if loss.item() <1e-5: break

		if epoch % Nskip == 0:

			print('epoch: ', epoch,' loss: ', loss.item())

			# Save loss

			loss_evolution[int(epoch/Nskip)] = loss.item()

	model.eval()

	# Take final measures

	Y_[1,:,:] = y_.detach().numpy()[:,0:Nsample]
	K_[1,:,:] = k_.detach().numpy()[:,0:Nsample]

	Y_selectivity_cat[1,:] = an.SI(y_.detach().numpy(), o_, average = 0)
	Y_selectivity_ctx[1,:] = an.SI(y_.detach().numpy(), c_, average = 0, exclude = 1, indeces_eclude = C_)

	Y_average_cat[1,:,1] = np.mean(y_.detach().numpy()[o_.astype(bool),:], 0)
	Y_average_cat[1,:,0] = np.mean(y_.detach().numpy()[np.logical_not(o_.astype(bool)),:], 0)

	Y_average_ctx[1,:,1] = np.mean(y_.detach().numpy()[c_.astype(bool),:], 0)
	Y_average_ctx[1,:,0] = np.mean(y_.detach().numpy()[np.logical_not(c_.astype(bool)),:], 0)

	# Store

	fac.Store(loss_evolution, 'loss_evolution.p', 'Data/')
	fac.Store(w0, 'w0.p', 'Data/')

	fac.Store(Y_, 'Y_.p', 'Data/Samples/')
	fac.Store(K_, 'K_.p', 'Data/Samples/')
	fac.Store(Y_selectivity_cat, 'Y_selectivity_cat.p', 'Data/')
	fac.Store(Y_selectivity_ctx, 'Y_selectivity_ctx.p', 'Data/')
	fac.Store(Y_average_cat, 'Y_average_cat.p', 'Data/')
	fac.Store(Y_average_ctx, 'Y_average_ctx.p', 'Data/')

else:

	# Retrieve

	loss_evolution = fac.Retrieve('loss_evolution.p', 'Data/')
	w0 = fac.Retrieve('w0.p', 'Data/')

	Y_ = fac.Retrieve('Y_.p', 'Data/Samples/')
	K_ = fac.Retrieve('K_.p', 'Data/Samples/')
	Y_selectivity_cat = fac.Retrieve('Y_selectivity_cat.p', 'Data/')
	Y_selectivity_ctx = fac.Retrieve('Y_selectivity_ctx.p', 'Data/')
	Y_average_cat = fac.Retrieve('Y_average_cat.p', 'Data/')
	Y_average_ctx = fac.Retrieve('Y_average_ctx.p', 'Data/')


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
### Plots


dashes = [3,3]

#

fac.SetPlotDim(2.1, 1.7)

fg = plt.figure()
ax0 = plt.axes(frameon=True)

cmap_base =  'bwr' 
vmin, vmax = 0.2, 0.8
cmap = fac.TruncateCmap(cmap_base, vmin, vmax)

mu = np.random.normal(0, 1., (Q, N)) # Temporary trick
nu = np.random.normal(0, 1., (Q, N)) # Temporary trick
x_, S_, C_, c_, o_, labels = net.Batch_ContextCat(mu, nu)

cax = plt.pcolor(np.corrcoef(x_), cmap = cmap, vmin=-0.8, vmax=0.8)

plt.axvline(x=P/2., color = '0', ls = '-', linewidth = 0.7)
plt.axhline(y=P/2., color = '0', ls = '-', linewidth = 0.7)

cbar = fg.colorbar(cax, ticks=[-0.8, 0, 0.8], orientation='vertical')
cbar.ax.set_xticklabels(['-0.8', '0', '0.8'])

plt.xlabel('Trials')
plt.ylabel('Trials')

plt.grid('off')

plt.xticks([])
plt.yticks([])

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

cax = plt.pcolor(np.corrcoef(Y_[0,:,:]), cmap = cmap, vmin=-0.8, vmax=0.8)

plt.axvline(x=P/2., color = '0', ls = '-', linewidth = 0.7)
plt.axhline(y=P/2., color = '0', ls = '-', linewidth = 0.7)

cbar = fg.colorbar(cax, ticks=[-0.8, 0, 0.8], orientation='vertical')
cbar.ax.set_xticklabels(['-0.5', '0', '0.5'])

plt.xlabel('Trials')
plt.ylabel('Trials')

plt.grid('off')

plt.xticks([])
plt.yticks([])

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

cax = plt.pcolor(np.corrcoef(Y_[0,:,:])[:Q,4*Q:4*Q+Q], cmap = cmap, vmin=-0.8, vmax=0.8)

cbar = fg.colorbar(cax, ticks=[-0.8, 0, 0.8], orientation='vertical')
cbar.ax.set_xticklabels(['-0.8', '0', '0.8'])

plt.grid('off')

plt.xticks([])
plt.yticks([])

plt.gca().invert_yaxis()

ax0.spines['top'].set_visible(False)
ax0.spines['right'].set_visible(False)
ax0.yaxis.set_ticks_position('left')
ax0.xaxis.set_ticks_position('bottom')
# plt.locator_params(nbins=4)

plt.savefig('y0_corr_zoom.pdf')

plt.show()

#

fg = plt.figure()
ax0 = plt.axes(frameon=True)

cax = plt.pcolor(np.corrcoef(Y_[1,:,:]), cmap = cmap, vmin=-0.8, vmax=0.8)

plt.axvline(x=P/2., color = '0', ls = '-', linewidth = 0.7)
plt.axhline(y=P/2., color = '0', ls = '-', linewidth = 0.7)

cbar = fg.colorbar(cax, ticks=[-0.8, 0, 0.8], orientation='vertical')
cbar.ax.set_xticklabels(['-0.5', '0', '0.5'])

plt.xlabel('Trials')
plt.ylabel('Trials')

plt.grid('off')

plt.xticks([])
plt.yticks([])

plt.gca().invert_yaxis()

ax0.spines['top'].set_visible(False)
ax0.spines['right'].set_visible(False)
ax0.yaxis.set_ticks_position('left')
ax0.xaxis.set_ticks_position('bottom')
# plt.locator_params(nbins=4)

plt.savefig('y_corr_matrix.pdf')

plt.show()

#

fg = plt.figure()
ax0 = plt.axes(frameon=True)

cax = plt.pcolor(np.corrcoef(Y_[1,:,:])[:Q,4*Q:4*Q+Q], cmap = cmap, vmin=-0.8, vmax=0.8)

cbar = fg.colorbar(cax, ticks=[-0.8, 0, 0.8], orientation='vertical')
cbar.ax.set_xticklabels(['-0.8', '0', '0.8'])

plt.grid('off')

plt.xticks([])
plt.yticks([])

plt.gca().invert_yaxis()

ax0.spines['top'].set_visible(False)
ax0.spines['right'].set_visible(False)
ax0.yaxis.set_ticks_position('left')
ax0.xaxis.set_ticks_position('bottom')
# plt.locator_params(nbins=4)

plt.savefig('y_corr_zoom.pdf')

plt.show()

#

fac.SetPlotDim(0.9, 1.35)

fg = plt.figure()
ax0 = plt.axes(frameon=True)

av_corr_0 = np.average(np.corrcoef(Y_[0,:,:])[0:int(P/2),int(P/2):])

# plt.axhline(y=0, color = '0', ls = '-', linewidth = 0.7)
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

# plt.axhline(y=0, color = '0', ls = '-', linewidth = 0.7)
plt.axhline(y=av_corr_0, color = '0', ls = '-', linewidth = 0.7)
# plt.axhline(y=av_corr, color = '0.4', ls = '--', linewidth = 0.7)

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

# plt.axvline(x = np.mean(Y_selectivity_cat[0,:]), ls = '-', color = '0', linewidth = 0.7)
# plt.axvline(x = 0, ls = '-', color = '0.', linewidth = 0.7)

plt.hist(Y_selectivity_cat[0,:], color = '0.85', normed = True, bins = 4)

plt.axvline(x = np.mean(Y_selectivity_cat[0,:]), ls = '--', color = '0.4', linewidth = 0.7)
plt.plot(np.mean(Y_selectivity_cat[0,:]), 0.4, '^', markersize = 4, color = '0.4', label = 'pop. average')

plt.xlabel(r'Cat selectivity index')
plt.ylabel(r'Neurons count')

# plt.xlim(-0.02, 0.04)
# plt.xticks([ 0., 0.02, 0.04])

plt.xlim(-0.03, 0.2)
plt.xticks([ 0., 0.1, 0.2])

plt.ylim(0, 16)
plt.yticks([0, 8, 16])

plt.legend(frameon = False)

ax0.spines['top'].set_visible(False)
ax0.spines['right'].set_visible(False)
ax0.yaxis.set_ticks_position('left')
ax0.xaxis.set_ticks_position('bottom')
# plt.locator_params(nbins=4)

plt.savefig('SI_cat_0.pdf')
plt.show()

#

fg = plt.figure()
ax0 = plt.axes(frameon=True)

plt.axvline(x = np.mean(Y_selectivity_cat[0,:]), ls = '-', color = '0', linewidth = 0.7)
# plt.axvline(x = 0, ls = '-', color = '0.', linewidth = 0.7)

plt.hist(Y_selectivity_cat[1,:], color = '0.85', normed = True, bins = 20)

plt.axvline(x = np.mean(Y_selectivity_cat[1,:]), ls = '--', color = '0.4', linewidth = 0.7)
plt.plot(np.mean(Y_selectivity_cat[1,:]), 0.4, '^', markersize = 4, color = '0.4')

plt.xlabel(r'Cat selectivity index')
plt.ylabel(r'Neurons count')

plt.xlim(-0.03, 0.2)
plt.xticks([ 0., 0.1, 0.2])

plt.ylim(0, 16)
plt.yticks([0, 8, 16])

ax0.spines['top'].set_visible(False)
ax0.spines['right'].set_visible(False)
ax0.yaxis.set_ticks_position('left')
ax0.xaxis.set_ticks_position('bottom')
# plt.locator_params(nbins=4)

plt.savefig('SI_cat_T.pdf')
plt.show()

#

fg = plt.figure()
ax0 = plt.axes(frameon=True)

# plt.axvline(x = np.mean(Y_selectivity_ctx[0,:]), ls = '-', color = '0', linewidth = 0.7)
# plt.axvline(x = 0, ls = '-', color = '0.', linewidth = 0.7)

plt.hist(Y_selectivity_ctx[0,:], color = '0.85', normed = True, bins = 20)

plt.axvline(x = np.mean(Y_selectivity_ctx[0,:]), ls = '--', color = '0.4', linewidth = 0.7)
plt.plot(np.mean(Y_selectivity_ctx[0,:]), 0.25, '^', markersize = 4, color = '0.4', label = 'pop. average')

plt.xlabel(r'Ctx selectivity index')
plt.ylabel(r'Neurons count')

plt.xlim(-0.2, 0.6)
plt.xticks([ -0.2, 0., 0.2, 0.4, 0.6])

plt.ylim(0, 8)
plt.yticks([0, 4, 8])

# plt.legend(frameon = False)

ax0.spines['top'].set_visible(False)
ax0.spines['right'].set_visible(False)
ax0.yaxis.set_ticks_position('left')
ax0.xaxis.set_ticks_position('bottom')
# plt.locator_params(nbins=4)

plt.savefig('SI_ctx_0.pdf')
plt.show()

#

fg = plt.figure()
ax0 = plt.axes(frameon=True)

plt.axvline(x = np.mean(Y_selectivity_ctx[0,:]), ls = '-', color = '0', linewidth = 0.7)
# plt.axvline(x = 0, ls = '-', color = '0.', linewidth = 0.7)

plt.hist(Y_selectivity_ctx[1,:], color = '0.85', normed = True, bins = 20)

plt.axvline(x = np.mean(Y_selectivity_ctx[1,:]), ls = '--', color = '0.4', linewidth = 0.7)
plt.plot(np.mean(Y_selectivity_ctx[1,:]), 0.25, '^', markersize = 4, color = '0.4')

plt.xlabel(r'Ctx selectivity index')
plt.ylabel(r'Neurons count')

plt.xlim(-0.2, 0.6)
plt.xticks([ -0.2, 0., 0.2, 0.4, 0.6])

plt.ylim(0, 8)
plt.yticks([0, 4, 8])

ax0.spines['top'].set_visible(False)
ax0.spines['right'].set_visible(False)
ax0.yaxis.set_ticks_position('left')
ax0.xaxis.set_ticks_position('bottom')
# plt.locator_params(nbins=4)

plt.savefig('SI_ctx_T.pdf')
plt.show()

#

# nrn_idx_cat = np.flipud(np.argsort(Y_selectivity_cat[1,:]-Y_selectivity_cat[0,:]))[0:70]
# nrn_idx_ctx = np.flipud(np.argsort(Y_selectivity_ctx[1,:]-Y_selectivity_ctx[0,:]))[0:70]


fac.SetPlotDim(2., 1.8)

fg = plt.figure()
ax0 = plt.axes(frameon=True)

Nmax = N

plt.plot([0, 1.2], [0, 1.2], color = '0.85', linewidth = 0.7)

log_fit = np.polyfit( Y_average_cat[0,:,0] , Y_average_cat[0,:,1], 1)
x_range = np.linspace(0,1.2,100)
corr_coeff = np.corrcoef(Y_average_cat[0,:,0], Y_average_cat[0,:,1])

plt.plot( Y_average_cat[0,:Nmax,0], Y_average_cat[0,:Nmax,1], 'o', color = '0.8', markeredgecolor = '1', markeredgewidth = '0.2' )
# plt.plot( Y_average_cat[0,nrn_idx_cat,0], Y_average_cat[0,nrn_idx_cat,1], 'o', color = '0.5', markeredgecolor = '1', markeredgewidth = '0.2' )
line, = plt.plot(x_range, log_fit[1] + x_range*log_fit[0], '-', color='0.4', label=r'$'+str(round(corr_coeff[0,1],2))+'$' )

plt.xlabel('Average activity $\mathsf{A}$')
plt.ylabel('Average activity $\mathsf{B}$')

plt.xlim(0,1)
plt.ylim(0,1)

plt.xticks([0, 0.5, 1])
plt.yticks([0, 0.5, 1])

ax0.spines['top'].set_visible(False)
ax0.spines['right'].set_visible(False)
ax0.yaxis.set_ticks_position('left')
ax0.xaxis.set_ticks_position('bottom')
plt.locator_params(nbins=3)

plt.legend(frameon=False, loc = 2)

plt.savefig('Y_average_cat_0.pdf')

plt.show()

#

fg = plt.figure()
ax0 = plt.axes(frameon=True)

plt.plot([0, 1.2], [0, 1.2], color = '0.8', linewidth = 0.7)

log_fit = np.polyfit( Y_average_cat[1,:,0] , Y_average_cat[1,:,1], 1)
corr_coeff = np.corrcoef(Y_average_cat[1,:,0], Y_average_cat[1,:,1])

plt.plot( Y_average_cat[1,:Nmax,0], Y_average_cat[1,:Nmax,1], 'o', color = '0.8', markeredgecolor = '1', markeredgewidth = '0.2' )
# plt.plot( Y_average_cat[1,nrn_idx_cat,0], Y_average_cat[1,nrn_idx_cat,1], 'o', color = '0.5', markeredgecolor = '1', markeredgewidth = '0.2' )
line, = plt.plot(x_range, log_fit[1] + x_range*log_fit[0], '-', color='0.4', label=r'$'+str(round(corr_coeff[0,1],2))+'$' )

plt.xlabel('Average activity $\mathsf{A}$')
plt.ylabel('Average activity $\mathsf{B}$')

plt.xlim(0,1)
plt.ylim(0,1)

plt.xticks([0, 0.5, 1])
plt.yticks([0, 0.5, 1])

ax0.spines['top'].set_visible(False)
ax0.spines['right'].set_visible(False)
ax0.yaxis.set_ticks_position('left')
ax0.xaxis.set_ticks_position('bottom')
plt.locator_params(nbins=3)

plt.legend(frameon=False, loc = 2)

plt.savefig('Y_average_cat_T.pdf')

plt.show()

#

fg = plt.figure()
ax0 = plt.axes(frameon=True)

Nmax = N

plt.plot([0, 1.2], [0, 1.2], color = '0.85', linewidth = 0.7)

log_fit = np.polyfit( Y_average_ctx[0,:,0] , Y_average_ctx[0,:,1], 1)
x_range = np.linspace(0,1.2,100)
corr_coeff = np.corrcoef(Y_average_ctx[0,:,0], Y_average_ctx[0,:,1])

plt.plot( Y_average_ctx[0,:Nmax,0], Y_average_ctx[0,:Nmax,1], 'o', color = '0.8', markeredgecolor = '1', markeredgewidth = '0.2' )
# plt.plot( Y_average_ctx[0,nrn_idx_ctx,0], Y_average_ctx[0,nrn_idx_ctx,1], 'o', color = '0.5', markeredgecolor = '1', markeredgewidth = '0.2' )
line, = plt.plot(x_range, log_fit[1] + x_range*log_fit[0], '-', color='0.4', label=r'$'+str(round(corr_coeff[0,1],2))+'$' )

plt.xlabel('Average activity $\mathsf{1}$')
plt.ylabel('Average activity $\mathsf{2}$')

plt.xlim(0,1.2)
plt.ylim(0,1.2)

plt.xticks([0, 0.6, 1.2])
plt.yticks([0, 0.6, 1.2])

ax0.spines['top'].set_visible(False)
ax0.spines['right'].set_visible(False)
ax0.yaxis.set_ticks_position('left')
ax0.xaxis.set_ticks_position('bottom')
plt.locator_params(nbins=3)

plt.legend(frameon=False, loc = 2)

plt.savefig('Y_average_ctx_0.pdf')

plt.show()

#

fg = plt.figure()
ax0 = plt.axes(frameon=True)

plt.plot([0, 1.2], [0, 1.2], color = '0.8', linewidth = 0.7)

log_fit = np.polyfit( Y_average_ctx[1,:,0] , Y_average_ctx[1,:,1], 1)
corr_coeff = np.corrcoef(Y_average_ctx[1,:,0], Y_average_ctx[1,:,1])

plt.plot( Y_average_ctx[1,:Nmax,0], Y_average_ctx[1,:Nmax,1], 'o', color = '0.8', markeredgecolor = '1', markeredgewidth = '0.2' )
# plt.plot( Y_average_ctx[1,nrn_idx_ctx,0], Y_average_ctx[1,nrn_idx_ctx,1], 'o', color = '0.5', markeredgecolor = '1', markeredgewidth = '0.2' )
line, = plt.plot(x_range, log_fit[1] + x_range*log_fit[0], '-', color='0.4', label=r'$'+str(round(corr_coeff[0,1],2))+'$' )

plt.xlabel('Average activity $\mathsf{1}$')
plt.ylabel('Average activity $\mathsf{2}$')

plt.xlim(0,1.2)
plt.ylim(0,1.2)

plt.xticks([0, 0.6, 1.2])
plt.yticks([0, 0.6, 1.2])

ax0.spines['top'].set_visible(False)
ax0.spines['right'].set_visible(False)
ax0.yaxis.set_ticks_position('left')
ax0.xaxis.set_ticks_position('bottom')
plt.locator_params(nbins=3)

plt.legend(frameon=False, loc = 2)

plt.savefig('Y_average_ctx_T.pdf')

plt.show()

#

fg = plt.figure()
ax0 = plt.axes(frameon=True)

x_range = np.linspace(-1,1,100)
log_fit = np.polyfit( np.fabs(w0[0,:]) , Y_selectivity_cat[1,:]-Y_selectivity_cat[0,:], 1)
corr_coeff = np.corrcoef(np.fabs(w0[0,:]), Y_selectivity_cat[1,:]-Y_selectivity_cat[0,:])

plt.plot( np.fabs(w0[0,:]), Y_selectivity_cat[1,:]-Y_selectivity_cat[0,:], 'o', color = '#AEEAEC', markeredgecolor = '1', markeredgewidth = '0.2', markersize = 2.8 )
line, = plt.plot(x_range, log_fit[1] + x_range*log_fit[0], '-', color='#00CED1', label=r'$'+str(round(corr_coeff[0,1],2))+'$' )

plt.xlabel('Initial readout connectivity')
plt.ylabel('Cat selectivity change')

plt.xlim(0, 0.1)
plt.xticks([0, 0.05, 0.1])

plt.ylim(-.015, 0.3)
plt.yticks([0, 0.1, 0.2, 0.3])

ax0.spines['top'].set_visible(False)
ax0.spines['right'].set_visible(False)
ax0.yaxis.set_ticks_position('left')
ax0.xaxis.set_ticks_position('bottom')
plt.locator_params(nbins=4)
plt.legend(frameon=False, loc=2)

# plt.colorbar()

plt.savefig('SI_cat_w0_T.pdf')

plt.show()

#

fg = plt.figure()
ax0 = plt.axes(frameon=True)

log_fit = np.polyfit( np.fabs(w0[0,:]) , Y_selectivity_ctx[1,:]-Y_selectivity_ctx[0,:], 1)
corr_coeff = np.corrcoef(np.fabs(w0[0,:]), Y_selectivity_ctx[1,:]-Y_selectivity_ctx[0,:])

plt.plot( np.fabs(w0[0,:]), Y_selectivity_ctx[1,:]-Y_selectivity_ctx[0,:], 'o', color = '#AEEAEC', markeredgecolor = '1', markeredgewidth = '0.2', markersize = 2.8 )
line, = plt.plot(x_range, log_fit[1] + x_range*log_fit[0], '-', color='#00CED1', label=r'$'+str(round(corr_coeff[0,1],2))+'$' )

plt.xlabel('Initial readout connectivity')
plt.ylabel('Ctx selectivity change')

plt.xlim(0, 0.1)
plt.xticks([0, 0.05, 0.1])

plt.ylim(-0.4, 0.8)
plt.yticks([-0.4, 0, 0.4, 0.8])

ax0.spines['top'].set_visible(False)
ax0.spines['right'].set_visible(False)
ax0.yaxis.set_ticks_position('left')
ax0.xaxis.set_ticks_position('bottom')
plt.locator_params(nbins=4)
plt.legend(frameon=False, loc=2)

# plt.colorbar()

plt.savefig('SI_ctx_w0_T.pdf')

plt.show()

#

fac.SetPlotDim(8, 1.5)

fg = plt.figure()
ax0 = plt.axes(frameon=True)

plt.plot( Y_selectivity_cat[1,:]-Y_selectivity_cat[0,:], Y_selectivity_ctx[1,:]-Y_selectivity_ctx[0,:], 'o', color = '0.8', markeredgecolor = '1', markeredgewidth = '0.2', markersize = 2.8 )
# plt.axvline(x=percentile_cat, color = '#E2205C')
# plt.axhline(y=percentile_ctx, color = '#FF99C3')

plt.xlabel('Cat selectivity change')
plt.ylabel('Ctx selectivity change')

plt.xlim(0, 0.2)
plt.xticks([0, 0.2])

plt.ylim(0, 0.6)
plt.yticks([0, 0.6])

ax0.spines['top'].set_visible(False)
ax0.spines['right'].set_visible(False)
ax0.yaxis.set_ticks_position('left')
ax0.xaxis.set_ticks_position('bottom')
# plt.locator_params(nbins=4)
plt.legend(frameon=False, loc=1)

# plt.colorbar()

plt.savefig('SI_ctx_cat.pdf')

plt.show()

#

fac.SetPlotDim(2, 1.8)

fg = plt.figure()
ax0 = plt.axes(frameon=True)

plt.plot( Y_selectivity_cat[1,:]-Y_selectivity_cat[0,:], Y_selectivity_ctx[1,:]-Y_selectivity_ctx[0,:], 'o', color = '0.8', markeredgecolor = '1', markeredgewidth = '0.2', markersize = 2.8 )

plt.xlabel('Cat selectivity change')
plt.ylabel('Ctx selectivity change')

plt.xscale('log')
plt.xlim(1e-6, 1)
plt.xticks([1e-6, 1e-3, 1])

plt.ylim(-0.3, 0.6)
plt.yticks([-0.3, 0, 0.3, 0.6])

ax0.spines['top'].set_visible(False)
ax0.spines['right'].set_visible(False)
ax0.yaxis.set_ticks_position('left')
ax0.xaxis.set_ticks_position('bottom')
# plt.locator_params(nbins=4)
plt.legend(frameon=False, loc=1)

# plt.colorbar()

plt.savefig('SI_ctx_cat_log.pdf')

plt.show()


# Construct axes dcat and dctx

x_, S_, C_, c_, o_, labels = net.Batch_ContextCat(np.random.normal(0, 1., (Q, N)), np.random.normal(0, 1., (Q, N)))

DeltaY_average_ctx = np.zeros(( N, 2 ))

DeltaY_average_ctx[:,1] = np.mean((Y_[1,:,:] - Y_[0,:,:])[c_.astype(bool),:], 0)
DeltaY_average_ctx[:,0] = np.mean((Y_[1,:,:] - Y_[0,:,:])[np.logical_not(c_.astype(bool)),:], 0)

DeltaY_average_cat = np.zeros(( N, 2 ))

DeltaY_average_cat[:,1] = np.mean((Y_[1,:,:] - Y_[0,:,:])[o_.astype(bool),:], 0)
DeltaY_average_cat[:,0] = np.mean((Y_[1,:,:] - Y_[0,:,:])[np.logical_not(o_.astype(bool)),:], 0)

#

fg = plt.figure()
ax0 = plt.axes(frameon=True)

log_fit = np.polyfit(np.fabs(DeltaY_average_cat[:,0]-DeltaY_average_cat[:,1]), Y_selectivity_cat[1,:]-Y_selectivity_cat[0,:], 1)
x_range = np.linspace(-1,1,100)
corr_coeff = np.corrcoef(np.fabs(DeltaY_average_cat[:,0]-DeltaY_average_cat[:,1]), Y_selectivity_cat[1,:]-Y_selectivity_cat[0,:])

plt.plot(np.fabs(DeltaY_average_cat[:,0]-DeltaY_average_cat[:,1]), Y_selectivity_cat[1,:]-Y_selectivity_cat[0,:], 'o',\
	color = '#C7E4FF', markeredgecolor = '1', markeredgewidth = '0.2' )
line, = plt.plot(x_range, log_fit[1] + x_range*log_fit[0], '-', color='#65ADFF', label=r'$'+str(round(corr_coeff[0,1],2))+'$' )

plt.xlabel(r'Cat direction ${d}_i^{\textrm{cat}}$')
plt.ylabel('Cat selectivity change')

plt.xlim(-0.02, 0.4)
plt.xticks([0, 0.2, 0.4])

plt.ylim(-0.02, 0.4)
plt.yticks([0, 0.2, 0.4])

plt.legend(loc=2, frameon=False)

ax0.spines['top'].set_visible(False)
ax0.spines['right'].set_visible(False)
ax0.yaxis.set_ticks_position('left')
ax0.xaxis.set_ticks_position('bottom')
plt.locator_params(nbins=4)

plt.savefig('SI_axis_cat.pdf')
plt.show()

#

fg = plt.figure()
ax0 = plt.axes(frameon=True)

log_fit = np.polyfit(np.fabs(DeltaY_average_ctx[:,0]-DeltaY_average_ctx[:,1]), Y_selectivity_ctx[1,:]-Y_selectivity_ctx[0,:], 1)
x_range = np.linspace(-1,1,100)
corr_coeff = np.corrcoef(np.fabs(DeltaY_average_ctx[:,0]-DeltaY_average_ctx[:,1]), Y_selectivity_ctx[1,:]-Y_selectivity_ctx[0,:])

plt.plot(np.fabs(DeltaY_average_ctx[:,0]-DeltaY_average_ctx[:,1]), Y_selectivity_ctx[1,:]-Y_selectivity_ctx[0,:], 'o',\
	color = '#C7E4FF', markeredgecolor = '1' )
line, = plt.plot(x_range, log_fit[1] + x_range*log_fit[0], '-', color='#65ADFF', label=r'$'+str(round(corr_coeff[0,1],2))+'$' )

plt.xlabel(r'Ctx direction ${d}_i^{\textrm{ctx}}$')
plt.ylabel('Ctx selectivity change')

plt.xlim(-0.02, 0.6)
plt.xticks([0, 0.3, 0.6])

plt.ylim(-0.3, 0.6)
plt.yticks([-0.3, 0, 0.3, 0.6])

plt.legend(loc=4, frameon=False)

ax0.spines['top'].set_visible(False)
ax0.spines['right'].set_visible(False)
ax0.yaxis.set_ticks_position('left')
ax0.xaxis.set_ticks_position('bottom')
plt.locator_params(nbins=4)

plt.savefig('SI_axis_ctx.pdf')
plt.show()


# Now try to predict axes from theory 

axis1 = np.zeros(N)
axis2 = np.zeros(N)

for i in range(N):

	axis1[i] =  ( np.mean( net.SigmoidPrime(K_[0,:,i], net.gain_psi, net.offset_psi)[S_<Q/2.] ) - \
				np.mean( net.SigmoidPrime(K_[0,:,i], net.gain_psi, net.offset_psi)[S_>=Q/2.] ) )

	axis2[i] = ( np.mean( net.SigmoidPrime(K_[0,:,i], net.gain_psi, net.offset_psi)[C_<Q/2.] ) - \
				np.mean( net.SigmoidPrime(K_[0,:,i], net.gain_psi, net.offset_psi)[C_>=Q/2.] ) )

#

pct = 85
percentile_cat = np.percentile(Y_selectivity_cat[1,:], pct)
percentile_ctx = np.percentile(Y_selectivity_ctx[1,:], pct)

where_cat = np.where((Y_selectivity_cat[1,:]-Y_selectivity_cat[0,:])>percentile_cat)[0]
where_ctx = np.where((Y_selectivity_ctx[1,:]-Y_selectivity_ctx[0,:])>percentile_ctx)[0]

#

fac.SetPlotDim(2, 1.8)

fg = plt.figure()
ax0 = plt.axes(frameon=True)

where = np.setdiff1d(where_cat, where_ctx)
plt.plot(0, np.mean(np.fabs((axis1))[where] - np.fabs((axis2))[where]), 'o-', markersize = 4, color = '#FFA11A')

where = np.intersect1d(where_cat, where_ctx)
plt.plot(1, np.mean(np.fabs((axis1))[where] - np.fabs((axis2))[where]), 'o-', markersize = 4, color = '#FF571D')

plt.ylim(-0.01, 0.02)
plt.yticks([-0.01, 0 , 0.01, 0.02])

plt.xlim(-0.5, 1.5)
plt.xticks([0,1], ['cat', 'cat and ctx'])

plt.xlabel('Neurons selective to')
plt.ylabel('Gain measure')

ax0.spines['top'].set_visible(False)
ax0.spines['right'].set_visible(False)
ax0.yaxis.set_ticks_position('left')
ax0.xaxis.set_ticks_position('bottom')
# plt.locator_params(nbins=4)
plt.legend(frameon=False, loc=2)

plt.savefig('bars_G.pdf')
plt.show()

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####

sys.exit(0)
