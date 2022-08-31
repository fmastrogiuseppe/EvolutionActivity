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
from sklearn import svm
from sklearn.model_selection import GridSearchCV

sys.path.append('')

import fct_facilities as fac
import fct_network as net
import fct_analysis as an
import fct_integrals as integ

fac.SetPlotParams()


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
### Parameters

# Retrieve

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


def Score(data):

	param_grid = {'C': [1, 20, 50, 100]}
	model = svm.SVC(kernel='linear')
	grid = GridSearchCV(model, param_grid)

	grid.fit(data, o_)
	model = grid.best_estimator_

	return model.score(data, o_)


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
### Compute and plot

score = [ Score(K_[0,:,:]), Score(K_[1,:,:]), Score(Y_[0,:,:]), Score(Y_[1,:,:]) ]

#

fac.SetPlotDim(2.0, 1.8)
plt.rcParams['xtick.labelsize'] = 9.5

fg = plt.figure()
ax0 = plt.axes(frameon=True)

plt.bar(range(len(score)), score,  color = '#CCE5FF')

plt.axhline(y=0.5, ls='--', color ='#000000', label = 'chance')

plt.xlabel('Dataset')
plt.ylabel('Training accuracy')

plt.xticks(np.arange(len(score)), [r'$\bm{k}_0$', r'$\bm{k}$', r'$\bm{y}_0$', r'$\bm{y}$'])
plt.yticks([ 0, 0.7, 1.4 ])

plt.xlim(-1, len(score))
plt.ylim(0, 1.4)

plt.legend(frameon=False)

# ax0.spines['bottom'].set_visible(False)
ax0.spines['top'].set_visible(False)
ax0.spines['right'].set_visible(False)
ax0.yaxis.set_ticks_position('left')
ax0.xaxis.set_ticks_position('bottom')
# plt.locator_params(nbins=4)

# plt.colorbar()

plt.grid('off')

plt.savefig('score.pdf')

plt.show()


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####

sys.exit(0)
