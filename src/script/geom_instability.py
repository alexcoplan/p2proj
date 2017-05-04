import matplotlib.pyplot as plt # type: ignore
import numpy as np # type: ignore

x_vals = np.linspace(40.0,40.6, num=800)
naive  = 1e8 * ((1e-8 ** x_vals) ** (1.0/x_vals))
better = 1e8 * np.exp((x_vals * np.log(1e-8)) / x_vals)

naive_line, = plt.plot(x_vals, naive)
better_line, = plt.plot(x_vals, better)

plt.title("Numerical Instability in Naïve Geometric Implementation")
plt.xlabel("Combinations (x)")
plt.ylabel("Result (y)")
plt.legend([naive_line, better_line], ['Geometric','LogGeometric'])
plt.show()
