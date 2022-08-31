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

