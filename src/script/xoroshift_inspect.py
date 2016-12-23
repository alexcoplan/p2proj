# run play/rand.out | python xoroshift_inspect.py

import matplotlib.pyplot as plt
import fileinput

xs = []
ys = []

for line in fileinput.input():
  xs.append(int(line) // 512)
  ys.append(int(line) % 512)
  
plt.plot(xs, ys, 'ro')
plt.axis([0, 512, 0, 512])
plt.show()
