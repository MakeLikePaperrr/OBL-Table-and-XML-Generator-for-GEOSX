from __future__ import division, print_function, absolute_import
import numpy as np


def gdem(X, X1, X2, X3):
    dX2 = X - X3
    dX1 = X - X2
    dX = X - X1
    b01 = dX.dot(dX1)
    b02 = dX.dot(dX2)
    b12 = dX1.dot(dX2)
    b11 = dX1.dot(dX1)
    b22 = dX2.dot(dX2)
    den = b11*b22-b12**2
    mu1 = (b02*b12 - b01*b22)/den
    mu2 = (b01*b12 - b02*b11)/den
    dacc = (dX - mu2*dX1)/(1+mu1+mu2)
    return dacc

__all__ = ['gdem' ]