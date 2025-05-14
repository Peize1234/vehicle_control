import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import itertools
from pathlib import Path
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
np.seterr(all='raise')

a = np.array([ 0.99530294, -0.09680943])
b = np.array([ 0.9952943,  -0.0968985 ])

print(np.sum(a * b))
print(np.arccos(np.clip(np.sum(a * b), -1.0, 1.0)))