import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import itertools
from pathlib import Path
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
np.seterr(all='raise')

a = {1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7}

print(len(a))