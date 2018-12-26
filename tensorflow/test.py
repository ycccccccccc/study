import collections
import numpy as np
import random
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 8))
x, y = 5,5
plt.scatter(x, y)
plt.annotate('qqq',
             xy=(x, y),
             xytext=(0, 0),
             textcoords='offset points',
             ha='center',
             va='center')
plt.show()