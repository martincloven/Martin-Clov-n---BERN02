#By: Martin Clovén , 
import numpy as np


def pred(x, beta_0, beta_1):
    val = np.e**(beta_0 + beta_1 * x)
    return val


def main(x, y):
    for i in range(len(x)):
        preds[i] = pred(x, beta_0, beta_1)
    return preds



x = []
y = []
x_ask = []