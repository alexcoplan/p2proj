import numpy as np

def digit_label(n):
  return np.array([0.0]*n + [1.0] + [0.0]*(9-n))


