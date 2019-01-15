import numpy as np
import matplotlib.pyplot as plt
from math import *

num = 100000
x_range = 250

def norm(m,n):
    dot = np.random.normal(m,n,num)
    xx = [int(x) for x in dot]
    return xx
a = np.zeros([11,100000])
for i in range(10):
    a[i] = norm(10+i*2,1+i/5.0)

for i in range(num):
    a[10,i] = sum(a[0:10,i])

count = np.zeros([11,x_range])
for con in range(11):
    for i in a[con]:
        count[con,int(i+20)] += 1

x = range(x_range)
x = [i - 20 for i in x]

for i in range(11):
    y = [y/num for y in count[i]]
    plt.plot(x,y)
plt.show()
