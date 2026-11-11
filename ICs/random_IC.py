import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as fft
from scipy.special import erf
import sys

l_int = 2.0
#amplitude_factor = float(sys.argv[1]) if len(sys.argv) > 2 else 1.0
#erfL = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
#seedv = int(sys.argv[3]) if len(sys.argv) > 3 else 1990
amplitude_factor = float(4)
erfL = float(2)
seedv = int(1990)
def interface(z):
    # Scale and clamp z into [-0.5, 0.5]
    z_clamped = np.clip(z / l_int, -0.5, 0.5)

    # Compute continuous interface profile
    # Clamp bounds to [1e-3, 1 - 1e-3]
    return 0.5 * (1.0 + np.sin(np.pi * z_clamped))

# Domain parameters
nx = 257
ny = 257
Lx = 80.0
Ly = 80.0
dx = Lx / nx

# Generate symmetric initial signal
rng = np.random.default_rng(seed=seedv)
h0half = rng.normal(loc=0.0, scale=1.0, size=nx) * amplitude_factor
h0 = np.concatenate((h0half[:-1], np.flipud(h0half)))
N = len(h0)  # N = 1025

# Fourier Transform
hf = fft.fft(h0)
freqs = fft.fftfreq(N, d=dx)

# Frequency Bandpass Filter using fftfreq (replacing manual loop/mid calculation)
# Note: Physical wave numbers k = 2 * pi * freqs
# Equivalent to filtering based on grid index inverse wave numbers (2 < |index_diff| < 20)
q_grid = np.abs(freqs * Lx)  # Index-equivalent grid wavenumber space
bandpass = (q_grid > 2.0) & (q_grid < 8.0)

hf_filtered = np.where(bandpass, hf, 0.0)

# Inverse FFT to get smoothed signal
hnew = np.real(fft.ifft(hf_filtered))

# Compute spatial derivative via Fourier differentiation: d/dx -> i * k
k = 2.0 * np.pi * freqs
dhnew_dx = np.real(fft.ifft(1j * k * hf_filtered))

# Extract the relevant interface profile slice
ys = (7.0 / 8.0) * Ly + hnew[:nx]
y = np.linspace(0, Ly, ny)

# 1. Surface vertical position y0 and derivative dhnew_dx (both shape: (nx,))
y0 =  (7.0 / 8.0) * Ly
ys = (7.0 / 8.0) * Ly + hnew[:nx]
dh_dx = dhnew_dx[:nx]

# 2. Vertical displacement from surface to grid points (shape: (nx, ny))
# dist_y > 0 means grid point is BELOW the interface
dist_y = ys[:, None] - y[None, :]

# 3. Local normal scale factor: 1 / sqrt(1 + (dh/dx)^2)  (shape: (nx, 1))
norm_scale = 1.0 / np.sqrt(1.0 + dh_dx[:, None] ** 2)

# 4. First-order normal distance (valid in narrow band near interface)
dist = dist_y * norm_scale

# 5. Evaluate phase-field interface
field = interface(dist)

# File I/O
#x1_val = float(sys.argv[1]) if len(sys.argv) > 1 else 0.2
x1_inf = 0.2
x2_inf = 0.01

x1 = field * np.maximum(erf((y0-y[None,:]) / erfL) * (x1_inf - x2_inf), 0.0) + x2_inf

plt.imshow(x1)
plt.colorbar()
plt.show()

field_clipped = np.clip(field, 1e-3, 1.0 - 1e-3)
with open("dealloy_n.in", "wb") as f:
    f.write(field_clipped.tobytes(order="F"))

with open("dealloy_x1.in", "wb") as f:
    f.write(x1.tobytes(order="F"))

x2 = np.full_like(field, 0.01)
with open("dealloy_x2.in", "wb") as f:
    f.write(x2.tobytes(order="F"))
