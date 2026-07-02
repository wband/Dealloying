import numpy as np
#import matplotlib.pyplot as plt
import scipy.fft as fft
import sys
from scipy.ndimage import distance_transform_edt

l_int = 1.5
def interface(z):
    phi = 0.5 * (1.0 + np.sin(np.pi * max(-0.5, min(0.5, z / 1.1 / l_int))))
    return max(min(phi, 1.0 - 1e-3), 1e-3)

nx = 513
ny = 513
nxhalf = 513
rng = np.random.default_rng(seed=1990)
h0half = rng.normal(loc=0.0, scale=8.0, size=(nxhalf))
h0 = np.concatenate((h0half[:-1],np.flipud(h0half)))
#plt.plot(h0)
hf = fft.fftshift(fft.fft(h0))
mid = nxhalf
hf1 = np.copy(hf)
for i in range(0,nxhalf*2-1):
    invq = np.abs(i-mid)
    if (2.0 < invq < 20.0):
        hf1[i] = hf[i]
    else:
        hf1[i] = 0

hnew = fft.ifft(fft.ifftshift(hf1))
Lx = 84.0
Ly = 84.0
y0 = 7.0/8.0*Ly + np.real(hnew)[:nxhalf]
y = np.linspace(0,Ly, ny)

binfield = np.ones((nx,ny))
for i in range(nx):
    binfield[i, y > y0[i]] = 0.0

dist = distance_transform_edt(binfield) - distance_transform_edt(1-binfield)
distbig = np.zeros((nx+2, ny+2))
distbig[1:-1,1:-1] = dist

def diffusion_iter(arr):
    arr[0,:] = arr[1,:]
    arr[-1,:] = arr[-2,:]
    arr[:,0] = arr[:,1]
    arr[:,-1] = arr[:,-2]
    arr[1:-1,1:-1] = arr[1:-1,1:-1] + 0.1*(-4.0*arr[1:-1,1:-1]+arr[:-2,1:-1]+arr[2:,1:-1]+arr[1:-1,:-2]+arr[1:-1,2:])

for i in range(0,10):
   diffusion_iter(distbig)

dist = distbig[1:-1,1:-1]

field = np.ones((nx,ny))
for i in range(nx):
    for j in range(ny): 
        field[i,j] = interface(dist[i,j]*Lx/float(nx))

x1_val=0.2
if len(sys.argv) > 1:
    x1_val=float(sys.argv[1])

f = open("dealloy_n.in",'wb')
f.write(field.tobytes(order='F'))
f.close()

x1 = np.ones_like(field)*x1_val
f = open("dealloy_x1.in",'wb')
f.write(x1.tobytes(order='F'))
f.close()

x2 = np.ones_like(field)*0.01
f = open("dealloy_x2.in",'wb')
f.write(x2.tobytes(order='F'))
f.close()
