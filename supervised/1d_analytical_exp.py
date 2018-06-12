import random
from math import ceil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

x = [-11.,-10., -5.0, 2.0, 10.0, 11.0]
y = [0.,1.,1.,1.,1.,1.]
alpha = 0.5
mixup_labels = True

lst = {}

for i in range(0,800000):
    r1 = random.randint(0,len(x)-1)
    r2 = random.randint(0,len(x)-1)
    lamb = float(np.random.beta(alpha,alpha))

    x1, y1 = x[r1], y[r1]
    x2, y2 = x[r2], y[r2]

    x_new = x1*lamb + x2*(1-lamb)
    if mixup_labels:
        y_new = y1*lamb + y2*(1-lamb)
    else:
        y_new = y1

    try:
        lst[round(x_new,1)].append(y_new)
    except:
        lst[round(x_new,1)] = [y_new]

margin = None
elst = []
vlst = []

for e in sorted(lst.keys()):
    v = sum(lst[e])/len(lst[e])
    print e,v

    if abs(0.5 - v) < 0.01:
        margin = e

    elst.append(e)
    vlst.append(v)

plt.scatter(x,y)
print "preplot"
plt.plot(elst,vlst)
print "plotted"
plt.savefig('margins.png')


