# Copyright 2017-2019 Aaron C. Prunty
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#        
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#==============================================================================
import os
import sys
import numpy as np
from vezda.data_utils import get_user_windows, fft_and_window
from vezda.math_utils import nextPow2
from vezda.sampling_utils import sampleSpace
from vezda.plot_utils import setFigure
from vezda.LinearOperators import asConvolutionalOperator
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import textwrap
sys.path.append(os.getcwd())
import pulseFun

rinterval, tinterval, tstep, dt, sinterval = get_user_windows()

datadir = np.load('datadir.npz')
recordingTimes = np.load(str(datadir['recordingTimes']))
recordingTimes = recordingTimes[tinterval]

Nt = len(recordingTimes)
T = recordingTimes[-1] - recordingTimes[0]
convolutionTimes = np.linspace(-T, T, 2 * Nt - 1)

N = nextPow2(2 * Nt)
freqs = np.fft.rfftfreq(N, tstep * dt)
Nf = len(freqs)

if 'sources' in datadir:
    sourcePoints = np.load(str(datadir['sources']))
    sourcePoints = sourcePoints[sinterval, :]
else:
    sys.exit(textwrap.dedent(
            '''
            '''))
    
Ns = sourcePoints.shape[0]

try:
    samplingGrid = np.load('samplingGrid.npz')
except FileNotFoundError:
    samplingGrid = None
        
if samplingGrid is None:
    sys.exit(textwrap.dedent(
            '''
            A sampling grid needs to be set up before test functions can
            be computed.
            Enter:
                
                vzgrid --help
                
            from the command-line for more information on how to set up a
            sampling grid.
            '''))
            
x = samplingGrid['x']
y = samplingGrid['y']
Nx, Ny = len(x), len(y)
tau = samplingGrid['tau']
if 'z' in samplingGrid:
    z = samplingGrid['z']
    Nz = len(z)
    Nsp = Nx * Ny * Nz
    samplingPoints = np.vstack(np.meshgrid(x, y, z, indexing='ij')).reshape(3, Nsp).T
else:
    Nsp = Nx * Ny
    samplingPoints = np.vstack(np.meshgrid(x, y, indexing='ij')).reshape(2, Nsp).T
    
# Load the focusing solution phi
phi = np.load('solutionNFE.npz')['X']

image = np.load('imageNFE.npz')['Image']
image = image.reshape(-1)

index = np.argmax(image)
focusingPoint = samplingPoints[index, :]
phi = phi[:, index]

velocity = 2
pulse = lambda t : pulseFun.pulse(t)

# Incident field
U_i = sampleSpace(samplingPoints, convolutionTimes, sourcePoints, velocity, pulse)
U_i = fft_and_window(U_i, tstep * dt, double_length=False)
N = asConvolutionalOperator(U_i)

# Focusing field
V_phi = N.dot(phi)
M = len(V_phi)
Nm = int(M / Nsp)
V_phi = V_phi.reshape((Nsp, Nm))
V_phi = np.fft.irfft(V_phi, axis=1)

def update_plot(i, data, scat):
    scat.set_array(data[:, i])
    return scat,

color_data = V_phi
x = samplingPoints[:, 0]
y = samplingPoints[:, 1]

fig, ax = setFigure(num_axes=1, mode='light', ax1_dim=2)
ax.plot(focusingPoint[0], focusingPoint[1], 'r*', markersize=12)
scat = ax.scatter(x, y, c=V_phi[:, 0], s=100, cmap='gray')

ani = animation.FuncAnimation(fig, update_plot, frames=range(Nm),
                              fargs=(color_data, scat))

plt.show()

#par = rsf.Par()
#outputFile = rsf.Output()

# Put axes in output
#outputFile.put('n1', Ns)
#outputFile.put('o1', 0)
#outputFile.put('d1', 1)

#outputFile.put('n2', Nt)
#outputFile.put('o2', 0.0)
#outputFile.put('d2', tstep * dt)

#outputFile.write(phi[:, :, 0])
#outputFile.close()

