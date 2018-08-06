# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 14:33:10 2018

@author: jiangjiechu
"""

import numpy as np
import matplotlib.pyplot as plt

soa = np.array([[1, 0, 3, 2], [1, 2, 1, 1], [2, 0, 9, 9]])
X, Y, U, V = zip(*soa)
plt.figure()
ax = plt.gca()
ax.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1)
ax.set_xlim([-1, 10])
ax.set_ylim([-1, 10])
plt.draw()
plt.show()