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
import textwrap
import numpy as np
from vezda.math_utils import nextPow2
from vezda.plot_utils import default_params
from scipy.linalg import norm
from scipy.signal import tukey
from tqdm import trange
from pathlib import Path
import pickle

sys.path.append(os.getcwd())
import pulseFun

def solve(V, Sp, Uh, alpha, domain):
    
    #==============================================================================
    # Load the receiver coordinates and recording times from the data directory
    datadir = np.load('datadir.npz')
    recordingTimes = np.load(str(datadir['recordingTimes']))
    receiverPoints = np.load(str(datadir['receivers']))
    if 'sources' in datadir:
        sourcePoints = np.load(str(datadir['sources']))
    else:
        sourcePoints = None
    if Path('noisyData.npz').exists():
        userResponded = False
        print(textwrap.dedent(
              '''
              Detected that band-limited noise has been added to the data array.
              Would you like to solve the Lippmann-Schwinger equation with the 
              noisy data? ([y]/n)
              
              Enter 'q/quit' exit the program.
              '''))
        while userResponded == False:
            answer = input('Action: ')
            if answer == '' or answer == 'y' or answer == 'yes':
                print('Proceeding with solution of Lippmann-Schwinger equation with noisy data...')
                # read in the noisy data array
                recordedData = np.load('noisyData.npz')['noisyData']
                userResponded = True
            elif answer == 'n' or answer == 'no':
                print('Proceeding with solution of Lippmann-Schwinger equation with noise-free data...')
                # read in the recorded data array
                recordedData  = np.load(str(datadir['recordedData']))
                userResponded = True
            elif answer == 'q' or answer == 'quit':
                sys.exit('Exiting program.\n')
            else:
                print('Invalid response. Please enter \'y/yes\', \'n\no\', or \'q/quit\'.')
                
    else:
        # read in the recorded data array
        recordedData  = np.load(str(datadir['recordedData']))
    
    # Compute length of time step.
    # This parameter is used for FFT shifting and time windowing
    dt = recordingTimes[1] - recordingTimes[0]
    
    if Path('plotParams.pkl').exists():
        plotParams = pickle.load(open('plotParams.pkl', 'rb'))
    else:
        plotParams = default_params()
    
    # Load the windowing parameters for the receiver and time axes of
    # the 3D data array
    if Path('window.npz').exists():
        print('Detected user-specified window:\n')
        windowDict = np.load('window.npz')
        
        # Receiver window parameters
        rstart = windowDict['rstart']
        rstop = windowDict['rstop']
        rstep = windowDict['rstep']
        
        # For display/printing purposes, count receivers with one-based
        # indexing. This amounts to incrementing the rstart parameter by 1
        print('window @ receivers : start =', rstart + 1)
        print('window @ receivers : stop =', rstop)
        print('window @ receivers : step =', rstep, '\n')
        
        # Time window parameters (with units of time)
        tstart = windowDict['tstart']
        tstop = windowDict['tstop']
        tstep = windowDict['tstep']
        tu = plotParams['tu']
        if tu != '':
            print('window @ time : start = %0.2f %s' %(tstart, tu))
            print('window @ time : stop = %0.2f %s' %(tstop, tu))
        else:
            print('window @ time : start =', tstart)
            print('window @ time : stop =', tstop)
        print('window @ time : step =', tstep, '\n')
        
        # Convert window parameters to corresponding array indices
        tstart = int(round(tstart / dt))
        tstop = int(round(tstop / dt))
        
        # Source window parameters
        slabel = windowDict['slabel']
        sstart = windowDict['sstart']
        sstop = windowDict['sstop']
        sstep = windowDict['sstep']
                
        # For display/printing purposes, count recordings/sources with one-based
        # indexing. This amounts to incrementing the sstart parameter by 1
        print('window @ %s : start = %s' %(slabel, sstart + 1))
        print('window @ %s : stop = %s' %(slabel, sstop))
        print('window @ %s : step = %s\n' %(slabel, sstep))                
                
        print('Applying window to data volume...')
        
    else:
        # Set default window parameters if user did
        # not specify window parameters.
        
        # Time window parameters (integers corresponding to array indices)
        tstart = 0
        tstop = len(recordingTimes)
        tstep = 1
        
        # Receiver window parameters
        rstart = 0
        rstop = receiverPoints.shape[0]
        rstep = 1
        
        # Source window parameters
        sstart = 0
        sstop = recordedData.shape[2]
        sstep = 1
        
    # Slice the recording times according to the time window parameters
    # to create a time window array
    tinterval = np.arange(tstart, tstop, tstep)
    recordingTimes = recordingTimes[tinterval]
    
    # Slice the receiverPoints array according to the receiver window parametes
    rinterval = np.arange(rstart, rstop, rstep)
    receiverPoints = receiverPoints[rinterval, :]
    
    sinterval = np.arange(sstart, sstop, sstep)
    if sourcePoints is not None:
        # Slice the sourcePoints array according to the source window parametes
        sourcePoints = sourcePoints[sinterval, :]

    recordedData = recordedData[rinterval, :, :]
    recordedData = recordedData[:, tinterval, :]
    recordedData = recordedData[:, :, sinterval]
    
    Nr, Nt, Ns = recordedData.shape
    
    # Apply tapered cosine (Tukey) window to time signals.
    # This ensures the fast fourier transform (FFT) used in
    # the definition of the matrix-vector product below is
    # acting on a function that is continuous at its edges.
                
    peakFreq = pulseFun.peakFreq
    # Np : Number of samples in the dominant period T = 1 / peakFreq
    Np = int(round(1 / (tstep * dt * peakFreq)))
    # alpha is set to taper over 6 of the dominant period of the
    # pulse function (3 periods from each end of the signal)
    percent_taper = 6 * Np / Nt
    print('Tapering time signals with Tukey window: %d'
          %(int(round(percent_taper * 100))) + '%')
    TukeyWindow = tukey(Nt, percent_taper)
    recordedData *= TukeyWindow[None, :, None]
    
    #==============================================================================
    if domain == 'freq':
        # Transform data into the frequency domain and bandpass for efficient solution
        # to Lippmann-Schwinger equation
        print('Transforming data to the frequency domain...')
    
        N = nextPow2(2 * Nt)
        recordedData = np.fft.rfft(recordedData, n=N, axis=1)
    
        if plotParams['fmax'] is None:
            freqs = np.fft.rfftfreq(N, tstep * dt)
            plotParams['fmax'] = np.max(freqs)
            pickle.dump(plotParams, open('plotParams.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)
    
        # Apply the frequency window
        fmin = plotParams['fmin']
        fmax = plotParams['fmax']
        fu = plotParams['fu']   # frequency units (e.g., Hz)
    
        if fu != '':
            print('Applying frequency window: [%0.2f %s, %0.2f %s]' %(fmin, fu, fmax, fu))
        else:
            print('Applying frequency window: [%0.2f, %0.2f]' %(fmin, fmax))
        
        df = 1.0 / (N * tstep * dt)
        startIndex = int(round(fmin / df))
        stopIndex = int(round(fmax / df))
        
        finterval = np.arange(startIndex, stopIndex, 1)
        recordedData = recordedData[:, finterval, :]
        
    else:
        # Pad data in the time domain to length 2*Nt-1 (length of circular convolution)
        # to solve Lippmann-Schwinger equation
    
        N = Nt - 1
        npad = ((0, 0), (N, 0), (0, 0))
        recordedData = np.pad(recordedData, pad_width=npad, mode='constant', constant_values=0)
    
    N = recordedData.shape[1]
    
    # Get machine precision
    eps = np.finfo(float).eps     # about 2e-16 (used in division
                                  # so we never divide by zero)
    #==============================================================================
    # Load the user-specified space-time sampling grid
    try:
        samplingGrid = np.load('samplingGrid.npz')
    except FileNotFoundError:
        samplingGrid = None
        
    if samplingGrid is None:
        sys.exit(textwrap.dedent(
                '''
                A space-time sampling grid needs to be set up before running the
                \'vzsolve\' command. Enter:
                    
                    vzgrid --help
                
                from the command-line for more information on how to set up a
                sampling grid.
                '''))
    
    if 'z' not in samplingGrid:
        # Solve Lippmann-Schwinger equation in three-dimensional space-time
        x = samplingGrid['x']
        y = samplingGrid['y']
        
        # Get number of sampling points in space
        Nx = len(x)
        Ny = len(y)
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        # Initialize the Image
        Image = np.zeros(X.shape)
        
        print('Localizing the source function...')
        if Ns == 1:
            # Compute the Tikhonov-regularized solution to the Lippmann-Schwinger equation L * chi = u.
            # 'u' is recorded data
            # 'alpha' is the regularization parameter
            # 'chi_alpha' is the regularized solution given 'alpha'
            data = np.reshape(recordedData, (N * Nr, 1))
            chi_alpha = V.dot(Sp.dot(Uh.dot(data)))
            chi_alpha = np.reshape(chi_alpha, (Nx * Ny, N))
            
            Image = np.reshape(norm(chi_alpha, axis=1), X.shape)                
            Imin = np.min(Image)
            Imax = np.max(Image)
            Image = (Image - Imin) / (Imax - Imin + eps)
                
        else:                
            for i in trange(Ns):
                data = np.reshape(recordedData[:, :, i], (N * Nr, 1))
                chi_alpha = V.dot(Sp.dot(Uh.dot(data)))
                chi_alpha = np.reshape(chi_alpha, (Nx * Ny, N))
                indicator = np.reshape(norm(chi_alpha, axis=1), X.shape)
                
                Imin = np.min(indicator)
                Imax = np.max(indicator)
                indicator = (indicator - Imin) / (Imax - Imin + eps)  # normalization
                Image += indicator**2
                    
            Image = np.sqrt(Image / Ns)
            
        np.savez('imageLSE.npz', Image=Image, alpha=alpha, X=X, Y=Y)
    
    #==============================================================================    
    else:
        # Solve Lippmann-Schwinger equation in four-dimensional space-time
        x = samplingGrid['x']
        y = samplingGrid['y']
        z = samplingGrid['z']
        
        # Get number of sampling points in space
        Nx = len(x)
        Ny = len(y)
        Nz = len(z)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Initialize the Image
        Image = np.zeros(X.shape)
        
        print('Localizing the source function...')
        if Ns == 1:
            # Compute the Tikhonov-regularized solution to the Lippmann-Schwinger equation L * chi = u.
            # 'u' is recorded data
            # 'alpha' is the regularization parameter
            # 'chi_alpha' is the regularized solution given 'alpha'
            data = np.reshape(recordedData, (N * Nr, 1))
            print('Solving system...')
            chi_alpha = V.dot(Sp.dot(Uh.dot(data)))
            chi_alpha = np.reshape(chi_alpha, (Nx * Ny * Nz, N))
            
            Image = np.reshape(norm(chi_alpha, axis=1), X.shape)                
            Imin = np.min(Image)
            Imax = np.max(Image)
            Image = (Image - Imin) / (Imax - Imin + eps)
                
        else:                
            for i in trange(Ns):
                data = np.reshape(recordedData[:, :, i], (N * Nr, 1))
                chi_alpha = V.dot(Sp.dot(Uh.dot(data)))
                chi_alpha = np.reshape(chi_alpha, (Nx * Ny * Nz, N))
                indicator = np.reshape(norm(chi_alpha, axis=1), X.shape)
                
                Imin = np.min(indicator)
                Imax = np.max(indicator)
                indicator = (indicator - Imin) / (Imax - Imin + eps)  # normalization
                Image += indicator**2
                    
            Image = np.sqrt(Image / Ns)
            
        np.savez('imageLSE.npz', Image=Image, alpha=alpha, X=X, Y=Y, Z=Z)