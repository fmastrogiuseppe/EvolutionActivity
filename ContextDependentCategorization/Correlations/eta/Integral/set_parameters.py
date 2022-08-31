import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import sys
from scipy import integrate
import os

sys.path.append('..')

import fct_facilities as fac


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
### Set parameters

gain_values = np.array([ 2.5 ])
offset_values = np.array([ 2 ])

# You need to run...

print ('You need to run... ', len(gain_values)*len(offset_values))

# Store

fac.Store(gain_values, 'gain_values.p', '')
fac.Store(offset_values, 'offset_values.p', '')