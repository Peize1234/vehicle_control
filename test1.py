import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import itertools
from pathlib import Path
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# file_path = r"D:\PythonCode\RL_exp\PPO_logs\AutoDrive-v2\PPO_AutoDrive-v2_log_131.csv"
#
# data = pd.read_csv(file_path).values
# data = data[:, 2]
#
# for i, d in enumerate(data):
#     data[i] = d[1:-1].split()
#
# data = np.vstack(data).astype(np.float32)
#
# avg_reward = np.mean(data, axis=1)
#
# plt.plot(avg_reward)
# plt.xlabel("Episode")
# plt.ylabel("Average Reward")
# plt.show()

A = np.array([[0, 0, 0, 0, 0, 0, 0],
              [0, 0, 1, 1, 0, 0, 0],
              [1, 0, 0, 0, 0, 0, 0],
              [1, 0, 1, 0, 1, 1, 0],
              [0, 0, 0, 1, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0],
              [0, 0, 1, 0, 0, 1, 0]], dtype=bool)
A += np.eye(A.shape[0], dtype=bool)

old = A
for i in range(A.shape[0] - 1):
    new = A @ (A + np.eye(A.shape[0], dtype=bool))
    if np.all(old == new):
        print(i)
        print(new.astype(int))

    old = new
