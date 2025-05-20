import numpy as np
from collections import deque
import sys

a = [np.zeros(100, dtype=int)]
print(sys.getsizeof(np.zeros(1000, dtype=int)))

print(sys.getsizeof(a))