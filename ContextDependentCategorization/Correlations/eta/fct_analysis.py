import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

import fct_network as net


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
### Selectivity


def SI(X, indeces, indeces_eclude = 0, average = 1, fact_average = 0, exclude = 0):

	P = X.shape[0]

	# Compute between and within averages

	BC = np.zeros(X.shape[1])
	WC = np.zeros(X.shape[1])

	for sp in range(P):

		s_B = np.where(indeces != indeces[sp])[0]

		for i, s in enumerate(s_B):
			BC += (X[sp,:] - X[s,:])**2 / (len(s_B)*P)

		s_W = np.where(indeces == indeces[sp])[0] 
		s_W = s_W[s_W != sp]   # Exclude s=s'

		if exclude: 
			s_W = np.where( np.logical_and( indeces == indeces[sp], \
					indeces_eclude != indeces_eclude[sp] ) )[0]

		for i, s in enumerate(s_W):
			WC += (X[sp,:] - X[s,:])**2 / (len(s_W)*P)

	# Compute selectivity index for all neurons

	SI = (BC-WC) / (BC+WC)

	if average == 0:
		return SI

	else:
		if fact_average: return np.mean((BC - WC)) / np.mean((BC + WC))  # Equivalent to clustering
		else: return np.mean(SI)


def SI_th(X, indeces, indeces_exclude = 0, exclude = 0):

	P = X.shape[1]
	Q = int(np.sqrt(P))

	BC = 0
	WC = 0

	for cat_idx, cat in enumerate([0,int(Q/2)]): 

		s_B = np.where(indeces != indeces[cat])[0]
		BC += np.mean(X[cat_idx,s_B]) / 2.

		s_W = np.where(indeces == indeces[cat])[0] 
		
		if exclude: 
			s_W = np.where( np.logical_and( indeces == indeces[0], \
					indeces_exclude != indeces_exclude[0] ) )[0]

		s_W = s_W[s_W != cat]   # Exclude s=s'
		WC += np.mean(X[cat_idx,s_W]) / 2.

	SI = (WC - BC) / (X[0,0] + X[1,int(Q/2)] - (BC + WC))

	return SI


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
### Correlation


def CorrelationMatrix(X):

	return np.corrcoef(X)


def Corr(C, indeces):

	P = C.shape[0]
	corr = 0

	# Takes averages of the correlation matrix

	for sp in range(P):

		s_B = np.where(indeces != indeces[sp])[0] 

		for i, s in enumerate(s_B):
			corr += C[sp,s] / (len(s_B)*P)

	return corr


def CorrAvg(X, indeces):

	X_A = np.mean(X[indeces.astype(bool),:], 0)
	X_B = np.mean(X[np.logical_not(indeces.astype(bool)),:], 0)

	return np.corrcoef(X_A, X_B)[0,1]


def Corr_th(C, indeces):

	P = C.shape[1]
	Q = int(np.sqrt(P))

	BC = 0

	for cat_idx, cat in enumerate([0,int(Q/2)]): 

		s_B = np.where(indeces != indeces[cat])[0] 
		BC += np.mean(C[cat_idx,s_B]) / 2.

	return BC


def CorrAvg_th(C, indeces):

	P = C.shape[1]
	Q = int(np.sqrt(P))

	# Average over s = 0

	s_B = np.where(indeces != indeces[0])[0] 
	BC = np.mean(C[0,s_B]) / 2.

	s_W = np.where(indeces == indeces[0])[0] 
	WC1 = np.mean(C[0,s_W])

	# Average over s = Q/2

	s_B = np.where(indeces != indeces[int(Q/2.)])[0] 
	BC += np.mean(C[1,s_B]) / 2.

	s_W = np.where(indeces == indeces[int(Q/2.)])[0] 
	WC2 = np.mean(C[1,s_W])

	# if context
	if indeces[int(Q/2)] == 0: return BC / ( (WC1+WC2)/ 2. )
	
	# if valence
	else: return BC / ( np.sqrt(WC1)*np.sqrt(WC2) )


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
### Dimensionality

def PR(X):

	# Center

	X = X  - np.outer(np.ones(X.shape[0]), np.mean(X,0))

	# Compute covariance matrix

	C = np.dot( X.T, X ) / X.shape[0]
	lambdas = np.linalg.eigvals(C).real

	# Compute PR

	D = np.sum(lambdas)**2 / np.sum(lambdas**2)

	return D

