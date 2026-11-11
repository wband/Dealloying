import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as fft
from scipy.special import erf
import sys

l_int = 2.0
amplitude_factor = float(4)
erfL = float(2)
def interface(z):
    # Scale and clamp z into [-0.5, 0.5]
    z_clamped = np.clip(z / l_int, -0.5, 0.5)

    # Compute continuous interface profile
    # Clamp bounds to [1e-3, 1 - 1e-3]
    return 0.5 * (1.0 + np.sin(np.pi * z_clamped))

# Domain parameters
nx = 5
ny = 513
Lx = 2.5
Ly = 80.0
dx = Lx / nx

y0 =  (7.0 / 8.0) * Ly
y = np.linspace(0, Ly, ny)

# 2. Vertical displacement from surface to grid points (shape: (nx, ny))
# dist_y > 0 means grid point is BELOW the interface
dist_y = y0*np.ones(nx)[:,None] - y[None, :]

# 5. Evaluate phase-field interface
field = interface(dist_y)

# File I/O
#x1_val = float(sys.argv[1]) if len(sys.argv) > 1 else 0.2
x1_inf = 0.2
x2_inf = 0.01

x1 = field * np.maximum(erf((y0-y[None,:]) / erfL) * (x1_inf - x2_inf), 0.0) + x2_inf

plt.imshow(x1)
plt.colorbar()
plt.show()

field_clipped = np.clip(field, 1e-3, 1.0 - 1e-3)
with open("dealloy_n_p1D.in", "wb") as f:
    f.write(field_clipped.tobytes(order="F"))

with open("dealloy_x1_p1D.in", "wb") as f:
    f.write(x1.tobytes(order="F"))

x2 = np.full_like(field, 0.01)
with open("dealloy_x2_p1D.in", "wb") as f:
    f.write(x2.tobytes(order="F"))
