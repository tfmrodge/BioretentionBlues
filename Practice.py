# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 14:14:25 2018

@author: Tim Rodgers
"""

import numpy as np
import pylab
from scipy.optimize import curve_fit

def sigmoid(x, x0, k):
     y = 1 / (1 + np.exp(-k*(x-x0)))
     return y

xdata = np.array([0,0.0229999999999961,0.141999999999996,0.283999999999992,0.424999999999983,0.567000000000007,0.709000000000003,0.849999999999994,0.99199999999999,1.13399999999999,1.27599999999998])
ydata = np.array([0,0.0831035223026541,1.0627710716992,4.65730750197914,10.4155082554683,17.5850932537215,20.9764530863231,22.196163673915,22.4347725324083,22.5260271318681,22.5387871318681])
ydata1 = ydata/ydata[-1]

popt, pcov = curve_fit(sigmoid, xdata, ydata1)
print(popt)

x = np.linspace(-1, 2, 50)
y = sigmoid(x, *popt)
y1 = y * ydata[-1]

pylab.plot(xdata, ydata, 'o', label='data')
pylab.plot(x,y1, label='fit')
pylab.ylim(0, 25)
pylab.legend(loc='best')
pylab.show()
