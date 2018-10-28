# Copyright 2017-2018 Aaron C. Prunty
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

import sys
import textwrap
import numpy as np
from vezda.Tikhonov import Tikhonov
from scipy.linalg import norm
from tqdm import trange
from pathlib import Path

def solver(s, U, V, alpha):
    
    #==============================================================================
    # Load the receiver coordinates and recording times from the data directory
    datadir = np.load('datadir.npz')
    recordingTimes = np.load(str(datadir['recordingTimes']))
    receiverPoints = np.load(str(datadir['receivers']))
    if 'sources' in datadir:
        sourcePoints = np.load(str(datadir['sources']))
    else:
        sourcePoints = None
    if Path('noisyData.npy').exists():
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
                recordedData = np.load('noisyData.npy')
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
    dt = (recordingTimes[-1] - recordingTimes[0]) / (len(recordingTimes) - 1)
    
    # Load the windowing parameters for the receiver and time axes of
    # the 3D data array
    if Path('window.npz').exists():
        windowDict = np.load('window.npz')
        
        # Time window parameters (with units of time)
        tstart = windowDict['tstart']
        tstop = windowDict['tstop']
        
        # Time window parameters (integers corresponding to indices in an array)
        twStart = int(round(tstart / dt))
        twStop = int(round(tstop / dt))
        tstep = windowDict['tstep']
        
        # Receiver window parameters
        rstart = windowDict['rstart']
        rstop = windowDict['rstop']
        rstep = windowDict['rstep']
        
        # Source window parameters
        slabel = windowDict['slabel']
        sstart = windowDict['sstart']
        sstop = windowDict['sstop']
        sstep = windowDict['sstep']
        
    else:
        # Set default window parameters if user did
        # not specify window parameters.
        
        # Time window parameters (units of time)
        tstart = recordingTimes[0]
        tstop = recordingTimes[-1]
        
        # Time window parameters (integers corresponding to indices in an array)
        twStart = 0
        twStop = len(recordingTimes)
        tstep = 1
        
        # Receiver window parameters
        rstart = 0
        rstop = receiverPoints.shape[0]
        rstep = 1
        
        # Source window parameters
        if sourcePoints is None:
            slabel = 'recordings'
        else:
            slabel = 'sources'
        sstart = 0
        sstop = recordedData.shape[2]
        sstep = 1
        
    # Slice the recording times according to the time window parameters
    # to create a time window array
    tinterval = np.arange(twStart, twStop, tstep)
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
    
    Nr = receiverPoints.shape[0]
    Ns = recordedData.shape[2]
    Nt = len(recordingTimes)    # number of samples in time window
    
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
        # Apply concurrent linear sampling method to three-dimensional space-time
        x = samplingGrid['x']
        y = samplingGrid['y']
        
        # Get number of sampling points in space
        Nx = len(x)
        Ny = len(y)
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        # Initialize the Histogram for storing images at each sampling point in time.
        # Initialize the Image (source-integrated Histogram with respect to L2 norm)
        Histogram = np.zeros((Nx, Ny, Ns))
        Image = np.zeros(X.shape)
        
        print('Localizing the source function...')
        if Ns == 1:
            # Compute the Tikhonov-regularized solution to the Lippmann-Schwinger equation L * chi = u.
            # 'u' is recorded data
            # 'alpha' is the regularization parameter
            # 'chi_alpha' is the regularized solution given 'alpha'
            data = np.reshape(recordedData, (Nt * Nr, 1))
            print('Solving system...')
            chi_alpha = Tikhonov(U, s, V, data, alpha)
            chi_alpha = np.reshape(chi_alpha, (Nx * Ny, Nt))
            
            Image = np.reshape(norm(chi_alpha, axis=1), X.shape)                
            Imin = np.min(Image)
            Imax = np.max(Image)
            Image = (Image - Imin) / (Imax - Imin + eps)
            Histogram[:, :, 0] = Image
                
        else:                
            for i in trange(Ns, desc='Summing over %s' %(slabel)):
                data = np.reshape(recordedData[:, :, i], (Nt * Nr, 1))
                chi_alpha = Tikhonov(U, s, V, data, alpha)
                chi_alpha = np.reshape(chi_alpha, (Nx * Ny, Nt))
                indicator = np.reshape(norm(chi_alpha, axis=1), X.shape)
                
                Imin = np.min(indicator)
                Imax = np.max(indicator)
                indicator = (indicator - Imin) / (Imax - Imin + eps)  # normalization
                Histogram[:, :, i] = indicator
                Image += indicator**2
                    
            Image = np.sqrt(Image / Ns)
            
        np.savez('imageLSE.npz', Image=Image, Histogram=Histogram,
                 alpha=alpha, X=X, Y=Y)
    
    #==============================================================================    
    else:
        # Apply concurrent linear sampling method to four-dimensional space-time
        x = samplingGrid['x']
        y = samplingGrid['y']
        z = samplingGrid['z']
        
        # Get number of sampling points in space
        Nx = len(x)
        Ny = len(y)
        Nz = len(z)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Initialize the Histogram for storing images at each sampling point in time.
        # Initialize the Image (source-integrated Histogram with respect to L2 norm)
        Histogram = np.zeros((Nx, Ny, Nz, Ns))
        Image = np.zeros(X.shape)
        
        print('Localizing the source function...')
        if Ns == 1:
            # Compute the Tikhonov-regularized solution to the Lippmann-Schwinger equation L * chi = u.
            # 'u' is recorded data
            # 'alpha' is the regularization parameter
            # 'chi_alpha' is the regularized solution given 'alpha'
            data = np.reshape(recordedData, (Nt * Nr, 1))
            print('Solving system...')
            chi_alpha = Tikhonov(U, s, V, data, alpha)
            chi_alpha = np.reshape(chi_alpha, (Nx * Ny * Nz, Nt))
            
            Image = np.reshape(norm(chi_alpha, axis=1), X.shape)                
            Imin = np.min(Image)
            Imax = np.max(Image)
            Image = (Image - Imin) / (Imax - Imin + eps)
            Histogram[:, :, 0] = Image
                
        else:                
            for i in trange(Ns, desc='Summing over %s' %(slabel)):
                data = np.reshape(recordedData[:, :, i], (Nt * Nr, 1))
                chi_alpha = Tikhonov(U, s, V, data, alpha)
                chi_alpha = np.reshape(chi_alpha, (Nx * Ny * Nz, Nt))
                indicator = np.reshape(norm(chi_alpha, axis=1), X.shape)
                
                Imin = np.min(indicator)
                Imax = np.max(indicator)
                indicator = (indicator - Imin) / (Imax - Imin + eps)  # normalization
                Histogram[:, :, i] = indicator
                Image += indicator**2
                    
            Image = np.sqrt(Image / Ns)
            
        np.savez('imageLSE.npz', Image=Image, Histogram=Histogram,
                 alpha=alpha, X=X, Y=Y, Z=Z)
