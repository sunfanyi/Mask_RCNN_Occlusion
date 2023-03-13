# -*- coding: utf-8 -*-
# @File    : try.py
# @Time    : 06/03/2023
# @Author  : Fanyi Sun
# @Github  : https://github.com/sunfanyi
# @Software: PyCharm

import numpy as np

a = np.array([[3, 4], [6, 1]])
b = np.array([[1, 3], [2, 5], [5, 2]])


diff = a[:, np.newaxis, :] - b[np.newaxis, :, :]
distances = np.sqrt(np.sum(diff**2, axis=-1))
min_dist = np.min(distances, axis=1)