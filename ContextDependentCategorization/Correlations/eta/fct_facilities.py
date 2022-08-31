import matplotlib.pyplot as plt
import numpy as np
import pickle
from matplotlib import rc
import matplotlib as mpl


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
#### Plot

def SetPlotParams():

	plt.style.use('ggplot')
	rc('text', usetex=True)
	plt.rcParams['text.latex.preamble']=[r"\usepackage{bm}"]

	fig_width = 2. # width in inches
	fig_height = 1.8  # height in inches
	fig_size =  [fig_width,fig_height]
	plt.rcParams['figure.figsize'] = fig_size
	plt.rcParams['figure.autolayout'] = True

	plt.rcParams['lines.linewidth'] = 1.
	plt.rcParams['lines.markeredgewidth'] = 0.3
	plt.rcParams['lines.markersize'] = 2.5
	plt.rcParams['font.size'] = 10
	plt.rcParams['legend.fontsize'] = 8
	plt.rcParams['axes.facecolor'] = '1'
	plt.rcParams['axes.edgecolor'] = '0'
	plt.rcParams['axes.linewidth'] = '0.7'

	plt.rcParams['axes.labelcolor'] = '0'
	plt.rcParams['axes.labelsize'] = 9.5
	plt.rcParams['axes.titlesize'] = 9.5
	plt.rcParams['xtick.labelsize'] = 8
	plt.rcParams['ytick.labelsize'] = 8
	plt.rcParams['xtick.color'] = '0'
	plt.rcParams['ytick.color'] = '0'
	plt.rcParams['xtick.major.size'] = 2
	plt.rcParams['ytick.major.size'] = 2

	plt.rcParams['font.sans-serif'] = 'Arial'

def SetPlotDim(x,y):

	fig_width = x # width in inches
	fig_height = y # height in inches
	fig_size =  [fig_width,fig_height]
	plt.rcParams['figure.figsize'] = fig_size
	plt.rcParams['figure.autolayout'] = True


def TruncateCmap(cmap, minval=0.0, maxval=1.0, n=100):
	'''
	https://stackoverflow.com/a/18926541
	'''
	if isinstance(cmap, str):
		cmap = plt.get_cmap(cmap)
		new_cmap = mpl.colors.LinearSegmentedColormap.from_list(
		'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
		cmap(np.linspace(minval, maxval, n)))
	return new_cmap


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
#### Data managing

def Store (obj, name, path):

	f = open ( path+name, 'wb')
	pickle.dump(obj,f, protocol=2)
	f.close()

def Retrieve (name, path):

	f = open( path+name, 'rb')
	obj = pickle.load(f)
	f.close()

	return obj
