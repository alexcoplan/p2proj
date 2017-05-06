import numpy as np # type: ignore
from scipy.interpolate import interp2d # type: ignore
import matplotlib.pyplot as plt # type: ignore
from matplotlib import cm # type: ignore
from mpl_toolkits.mplot3d import Axes3D # type: ignore

x_coords = np.linspace(0.0,10.0, num=80)
y_coords = np.linspace(0.0,10.0, num=80)

x,y = np.meshgrid(x_coords, y_coords)

sig_sharpness = 1.0
sig_shift = 32.0

z = 1.0 / (1.0 + np.exp(- sig_sharpness * (x ** 1.5 + y ** 1.5 - sig_shift))) \
  + 1.0 / (0.8 + 0.05*(x*2 + y*2))

# generate smooth noise
xn = np.linspace(0.0,10.0, num=15)
yn = np.linspace(0.0,10.0, num=15)
Xn,Yn = np.meshgrid(xn,yn)
z_noise = 0.03 * np.random.randn(*Xn.shape)
f_noise = interp2d(xn, yn, z_noise, kind='cubic')

z += f_noise(x_coords,y_coords)

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(x,y,z, cmap=cm.viridis)
ax.set_zlabel('Loss')
ax.set_xlabel('x')
ax.set_ylabel('y')

plt.show()
