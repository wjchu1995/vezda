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

import os
import sys
import textwrap
import numpy as np
from vezda.Tikhonov import Tikhonov
from vezda.sampling_utils import samplingIsCurrent, sampleSpaceTime
from vezda.math_utils import timeShift
from scipy.linalg import norm
from tqdm import trange
from time import sleep
from pathlib import Path

# pulseFun module contains information about the time-dependent
# pulse function used to generate the interrogating wave. This
# module is contained in the 'pulseFun.py' file provided by the
# user.
sys.path.append(os.getcwd())
import pulseFun

def solver(medium, s, U, V, alpha):
    
    #==============================================================================
    # Load the receiver coordinates and recording times from the data directory
    datadir = np.load('datadir.npz')
    recordingTimes = np.load(str(datadir['recordingTimes']))
    receiverPoints = np.load(str(datadir['receivers']))
    sourcePoints = np.load(str(datadir['sources']))
    
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
        sstart = 0
        sstop = sourcePoints.shape[0]
        sstep = 1
        
    # Slice the recording times according to the time window parameters
    # to create a time window array
    tinterval = np.arange(twStart, twStop, tstep)
    recordingTimes = recordingTimes[tinterval]
    
    # Slice the receiverPoints array according to the receiver window parametes
    rinterval = np.arange(rstart, rstop, rstep)
    receiverPoints = receiverPoints[rinterval, :]

    # Slice the sourcePoints array according to the source window parametes
    sinterval = np.arange(sstart, sstop, sstep)
    sourcePoints = sourcePoints[sinterval, :]    
    
    Nr = receiverPoints.shape[0]
    Nt = len(recordingTimes)    # number of samples in time window
    T = tstop - tstart      # length of time window (dimensional)
    
    # Get information about the pulse function used to 
    # generate the interrogating wave (These parameters are
    # used to help Vezda decide if it needs to recompute the 
    # test functions in the case the user changes these parameters.)
    velocity = pulseFun.velocity    # only used if medium == constant
    peakFreq = pulseFun.peakFreq    # peak frequency
    peakTime = pulseFun.peakTime    # time at which the pulse amplitude is maximum
    
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
        # Apply total-energy linear sampling method to three-dimensional space-time
        x = samplingGrid['x']
        y = samplingGrid['y']
        tau = samplingGrid['tau']
        z = None
        
        # Get number of sampling points in space and time
        Nx = len(x)
        Ny = len(y)
        X, Y = np.meshgrid(x, y, indexing='ij')
        Ntau = len(tau)
        if Ntau > 1:
            deltaTau = tau[1] - tau[0] # step size for sampling in time
        
        # Initialize the Histogram for storing images at each sampling point in time.
        # Initialize the Image (time-integrated Histogram with respect to L2 norm)
        Histogram = np.zeros((Nx, Ny, Ntau))
        Image = np.zeros(X.shape)
        
        if medium == 'constant':
            # Vezda will compute free-space test functions over the space-time
            # sampling grid via function calls to 'FundamentalSolutions.py'. This is
            # much more efficient than applying a forward and inverse FFT pair to
            # shift the test functions in time corresponding to different sampling
            # points in time. FFT pairs are only used when medium == variable.
            pulse = lambda t : pulseFun.pulse(t)
            
            # Previously computed test functions and parameters from pulseFun module
            # are stored in 'VZTestFuncs.npz'. If the current space-time sampling grid
            # and pulseFun module parameters are consistent with those stored in
            # 'VZTestFuncs.npz', then Vezda will load the previously computed test
            # functions. Otherwise, Vezda will recompute the test functions. This reduces
            # computational cost by only computing test functions when necessary.
            if Path('VZTestFuncs.npz').exists():
                print('\nDetected that free-space test functions have already been computed...')
                print('Checking consistency with current space-time sampling grid and pulse function...')
                TFDict = np.load('VZTestFuncs.npz')
                
                if samplingIsCurrent(TFDict, recordingTimes, velocity, tau, x, y, z, peakFreq, peakTime):
                    print('Moving forward to imaging algorithm...')
                    TFarray = TFDict['TFarray']
                    
                else:
                    print('Recomputing test functions...')
                    TFarray, samplingPoints = sampleSpaceTime(receiverPoints, recordingTimes,
                                                              velocity, tau, x, y, z, pulse)
                    
                    np.savez('VZTestFuncs.npz', TFarray=TFarray, time=recordingTimes,
                             peakFreq=peakFreq, peakTime=peakTime, velocity=velocity,
                             x=x, y=y, tau=tau, samplingPoints=samplingPoints)
                
            else:                
                print('\nComputing free-space test functions for the current space-time sampling grid...')
                TFarray, samplingPoints = sampleSpaceTime(receiverPoints, recordingTimes,
                                                          velocity, tau, x, y, z, pulse)
                    
                np.savez('VZTestFuncs.npz', TFarray=TFarray, time=recordingTimes,
                         peakFreq=peakFreq, peakTime=peakTime, velocity=velocity,
                         x=x, y=y, tau=tau, samplingPoints=samplingPoints)
            
            print('Localizing the source function...')
            if Ntau == 1:
                # Compute the Tikhonov-regularized solution to the near-field equation N * phi = tf.
                # 'tf' is a test function
                # 'alpha' is the regularization parameter
                # 'phi_alpha' is the regularized solution given 'alpha'
                TF = TFarray[:, :, :, 0]
                l = 0 # counter for spatial sampling points
                for ix in trange(Nx, desc='Solving system'):
                    for iy in range(Ny):
                        tf = np.reshape(TF[:, :, l], (Nt * Nr, 1))
                        phi_alpha = Tikhonov(U, s, V, tf, alpha)
                        Image[ix, iy] = 1.0 / (norm(phi_alpha) + eps)
                        l += 1
                
                Imin = np.min(Image)
                Imax = np.max(Image)
                Image = (Image - Imin) / (Imax - Imin + eps)
                Histogram[:, :, 0] = Image
                
            else:
                # Store spatial reconstruction of the source function for each
                # sampling point in time in Histogram
                # Compute time-integrated Image from Histogram using L2 norm
                # Discretize L2 integration using trapezoidal rule with 
                # uniform step size deltaTau
                firstIndicator = np.zeros(X.shape)
                lastIndicator = np.zeros(X.shape)
                firstTF = TFarray[:, :, :, 0]   # test function array for first time sample
                lastTF = TFarray[:, :, :, -1]   # test function array for last time sample
                
                # Compute the Tikhonov-regularized solution to the near-field equation N * phi = tf.
                # 'tf' is a test function
                # 'alpha' is the regularization parameter
                # 'phi_alpha' is the regularized solution given 'alpha'
                l = 0 # counter for spatial sampling points
                for ix in range(Nx):
                    for iy in range(Ny):
                        tf1 = np.reshape(firstTF[:, :, l], (Nt * Nr, 1))
                        tf2 = np.reshape(lastTF[:, :, l], (Nt * Nr, 1))
                        phi_alpha1 = Tikhonov(U, s, V, tf1, alpha)
                        phi_alpha2 = Tikhonov(U, s, V, tf2, alpha)
                        firstIndicator[ix, iy] = 1.0 / (norm(phi_alpha1) + eps)
                        lastIndicator[ix, iy] = 1.0 / (norm(phi_alpha2) + eps)
                        l += 1
                        
                Imin1 = np.min(firstIndicator)
                Imax1 = np.max(firstIndicator)
                firstIndicator = (firstIndicator - Imin1) / (Imax1 - Imin1 + eps)  # normalization
                Histogram[:, :, 0] = firstIndicator
                
                Imin2 = np.min(lastIndicator)
                Imax2 = np.max(lastIndicator)
                lastIndicator = (lastIndicator - Imin2) / (Imax2 - Imin2 + eps)  # normalization
                Histogram[:, :, -1] = lastIndicator
                
                Image += 0.5 * (firstIndicator**2 + lastIndicator**2)
                
                for it in trange(1, Ntau - 1, desc='Source-time integration'):
                    indicator = np.zeros(X.shape)
                    l = 0 # counter for spatial sampling points
                    for ix in trange(Nx, desc='Solving system', leave=False):
                        for iy in range(Ny):
                            tf = np.reshape(TFarray[:, :, l, it], (Nt * Nr, 1))
                            phi_alpha = Tikhonov(U, s, V, tf, alpha)
                            indicator[ix, iy] = 1.0 / (norm(phi_alpha) + eps)
                            l += 1
                            sleep(0.001)
                
                    Imin = np.min(indicator)
                    Imax = np.max(indicator)
                    indicator = (indicator - Imin) / (Imax - Imin + eps)  # normalization
                    Histogram[:, :, it] = indicator
                    Image += indicator**2
                    
                Image *= deltaTau / T
                Image = np.sqrt(Image)
                    
        elif medium == 'variable':
            if 'testFuncs' in datadir:
                # TFarray computed outside of Vezda for variable media will only 
                # have 3 axes. A fourth axis will be added to accomodate for time shifts
                # corresponding to different sampling points in time
                tempTFarray = np.load(str(datadir['testFuncs']))
            
                # Apply the receiver window, if any
                tempTFarray = tempTFarray[rinterval, :, :]
            
                # Add new axis to TFarray to accomodate time shifts
                TFarray = np.zeros((Nr, Nt, Nx * Ny, Ntau))
                TFarray[:, :, :, 0] = tempTFarray[:, tinterval, :]
                
                # Load the sampling points
                samplingPoints = np.load(str(datadir['samplingPoints']))
            
            else:
                sys.exit(textwrap.dedent(
                        '''
                        FileNotFoundError: Attempted to load file containing test
                        functions, but no such file exists. If a file exists containing
                        the test functions, run:
                            
                            'vzdata --path=<path/to/data/>'
                        
                        and specify the file containing the test functions when prompted.
                        Otherwise, specify 'no' when asked if a file containing the test
                        functions exists.
                        '''))
            
            userResponded = False
            print(textwrap.dedent(
                 '''
                 In what order was the sampling grid spanned to compute the test functions?
                 
                 Enter 'xy' if for each x, loop over y. (Default)
                 Enter 'yx' if for each y, loop over x.
                 Enter 'q/quit' to abort the calculation.
                 '''))
            while userResponded == False:
                order = input('Order: ')
                if order == '' or order == 'xy':
                    print('Proceeding with order \'xy\'...')
                    print('Localizing the source function...')
                    if Ntau == 1:
                        # Compute the Tikhonov-regularized solution to the near-field equation N * phi = tf.
                        # 'tf' is a test function
                        # 'alpha' is the regularization parameter
                        # 'phi_alpha' is the regularized solution given 'alpha'
                        if tau[0] != 0:
                            TF = timeShift(tempTFarray, tau[0], dt)
                            TF = TF[:, tinterval, :]
                            TFarray[:, :, :, 0] = TF
                        else:
                            TF = TFarray[:, :, :, 0]
                        l = 0 # counter for spatial sampling points
                        for ix in trange(Nx, desc='Solving system'):
                            for iy in range(Ny):
                                tf = np.reshape(TF[:, :, l], (Nt * Nr, 1))
                                phi_alpha = Tikhonov(U, s, V, tf, alpha)
                                Image[ix, iy] = 1.0 / (norm(phi_alpha) + eps)
                                l += 1
                
                        Imin = np.min(Image)
                        Imax = np.max(Image)
                        Image = (Image - Imin) / (Imax - Imin + eps)
                        Histogram[:, :, 0] = Image
                        userResponded = True
                        break
                        
                    else:
                        # discretize using trapezoidal rule with uniform step size deltaTau
                        firstIndicator = np.zeros(X.shape)
                        lastIndicator = np.zeros(X.shape)
                        firstTF = TFarray[:, :, :, 0]
                        lastTF = timeShift(tempTFarray, tau[-1], dt)
                        lastTF = lastTF[:, tinterval, :]
                        TFarray[:, :, :, -1] = lastTF
                    
                        l = 0 # counter for spatial sampling points
                        for ix in range(Nx):
                            for iy in range(Ny):
                                tf1 = np.reshape(firstTF[:, :, l], (Nt * Nr, 1))
                                tf2 = np.reshape(lastTF[:, :, l], (Nt * Nr, 1))
                                phi_alpha1 = Tikhonov(U, s, V, tf1, alpha)
                                phi_alpha2 = Tikhonov(U, s, V, tf2, alpha)
                                firstIndicator[ix, iy] = 1.0 / (norm(phi_alpha1) + eps)
                                lastIndicator[ix, iy] = 1.0 / (norm(phi_alpha2) + eps)
                                l += 1
                            
                        Imin1 = np.min(firstIndicator)
                        Imax1 = np.max(firstIndicator)
                        firstIndicator = (firstIndicator - Imin1) / (Imax1 - Imin1 + eps)  # normalization
                        Histogram[:, :, 0] = firstIndicator
                    
                        Imin2 = np.min(lastIndicator)
                        Imax2 = np.max(lastIndicator)
                        lastIndicator = (lastIndicator - Imin2) / (Imax2 - Imin2 + eps)  # normalization
                        Histogram[:, :, -1] = lastIndicator
                        
                        Image += 0.5 * (firstIndicator**2 + lastIndicator**2)
                    
                        for it in trange(1, Ntau - 1, desc='Source-time integration'):
                            TF = timeShift(tempTFarray, tau[it], dt)
                            TF = TF[:, tinterval, :]
                            TFarray[:, :, :, it] = TF
                            indicator = np.zeros(X.shape)
                            
                            l = 0 # counter for spatial sampling points
                            for ix in trange(Nx, desc='Solving system', leave=False):
                                for iy in range(Ny):
                                    tf = np.reshape(TF[:, :, l], (Nt * Nr, 1))
                                    phi_alpha = Tikhonov(U, s, V, tf, alpha)
                                    indicator[ix, iy] = 1.0 / (norm(phi_alpha) + eps)
                                    l += 1
                                    sleep(0.001)
                                
                            Imin = np.min(indicator)
                            Imax = np.max(indicator)
                            indicator = (indicator - Imin) / (Imax - Imin + eps)  # normalization
                            Histogram[:, :, it] = indicator
                            Image += indicator**2
                            
                        Image *= deltaTau / T
                        Image = np.sqrt(Image)
                        userResponded = True
                        break
                
                elif order == 'yx':
                    print('Proceeding with order \'yx\'...')
                    print('Localizing the source function...')
                    if Ntau == 1:
                        # Compute the Tikhonov-regularized solution to the near-field equation N * phi = tf.
                        # 'tf' is a test function
                        # 'alpha' is the regularization parameter
                        # 'phi_alpha' is the regularized solution given 'alpha'
                        if tau[0] != 0:
                            TF = timeShift(tempTFarray, tau[0], dt)
                            TF = TF[:, tinterval, :]
                            TFarray[:, :, :, 0] = TF
                        else:
                            TF = TFarray[:, :, :, 0]
                        l = 0 # counter for spatial sampling points
                        for iy in trange(Ny, desc='Solving system'):
                            for ix in range(Nx):
                                tf = np.reshape(TF[:, :, l], (Nt * Nr, 1))
                                phi_alpha[:, l, 0] = Tikhonov(U, s, V, tf, alpha)
                                Image[ix, iy] = 1.0 / (norm(phi_alpha[:, l, 0]) + eps)
                                l += 1
                
                        Imin = np.min(Image)
                        Imax = np.max(Image)
                        Image = (Image - Imin) / (Imax - Imin + eps)
                        Histogram[:, :, 0] = Image
                        userResponded = True
                        break
                    
                    else:
                        # discretize using trapezoidal rule with uniform deltaTau
                        firstIndicator = np.zeros(X.shape)
                        lastIndicator = np.zeros(X.shape)
                        firstTF = TFarray[:, :, :, 0]
                        lastTF = timeShift(tempTFarray, tau[-1], dt)
                        lastTF = lastTF[:, tinterval, :]
                        TFarray[:, :, :, -1] = lastTF
                        
                        l = 0 # counter for spatial sampling points
                        for iy in range(Ny):
                            for ix in range(Nx):
                                tf1 = np.reshape(firstTF[:, :, l], (Nt * Nr, 1))
                                tf2 = np.reshape(lastTF[:, :, l], (Nt * Nr, 1))
                                phi_alpha1 = Tikhonov(U, s, V, tf1, alpha)
                                phi_alpha2 = Tikhonov(U, s, V, tf2, alpha)
                                firstIndicator[ix, iy] = 1.0 / (norm(phi_alpha1) + eps)
                                lastIndicator[ix, iy] = 1.0 / (norm(phi_alpha2) + eps)
                                l += 1
                            
                        Imin1 = np.min(firstIndicator)
                        Imax1 = np.max(firstIndicator)
                        firstIndicator = (firstIndicator - Imin1) / (Imax1 - Imin1 + eps)     # normalization
                        Histogram[:, :, 0] = np.sqrt(firstIndicator)
                        
                        Imin2 = np.min(lastIndicator)
                        Imax2 = np.max(lastIndicator)
                        lastIndicator = (lastIndicator - Imin2) / (Imax2 - Imin2 + eps)     # normalization
                        Histogram[:, :, -1] = np.sqrt(lastIndicator)
                        
                        Image += 0.5 * (firstIndicator**2 + lastIndicator**2)
                        
                        for it in trange(1, Ntau - 1, desc='Source-time integration'):
                            TF = timeShift(tempTFarray, tau[it], dt)
                            TF = TF[:, tinterval, :]
                            TFarray[:, :, :, it] = TF
                            indicator = np.zeros(X.shape)
                            
                            l = 0 # counter for spatial sampling points
                            for iy in trange(Ny, desc='Solving system', leave=False):
                                for ix in range(Nx):
                                    tf = np.reshape(TF[:, :, l], (Nt * Nr, 1))
                                    phi_alpha = Tikhonov(U, s, V, tf, alpha)
                                    indicator[ix, iy] = 1.0 / (norm(phi_alpha) + eps)
                                    l += 1
                                    sleep(0.001)
                                    
                            Imin = np.min(indicator)
                            Imax = np.max(indicator)
                            indicator = (indicator - Imin) / (Imax - Imin + eps)   # normalization
                            Histogram[:, :, it] = indicator
                            Image += indicator**2
                            
                        Image *= deltaTau / T
                        Image = np.sqrt(Image)
                        userResponded = True
                        break
                
                elif order == 'q' or order == 'quit':
                    sys.exit('Aborting calculation.')
                
                else:
                    print(textwrap.dedent(
                         '''
                         Invalid response. Please enter one of the following:
                         
                         Enter 'xy' if for each x, loop over y. (Default)
                         Enter 'yx' if for each y, loop over x.
                         Enter 'q/quit' to abort the calculation.
                         '''))
                        
            np.savez('VZTestFuncs.npz', TFarray=TFarray, time=recordingTimes,
                     peakFreq=peakFreq, peakTime=peakTime, velocity=velocity,
                     x=x, y=y, tau=tau, samplingPoints=samplingPoints)
            
        np.savez('imageNFE.npz', Image=Image, Histogram=Histogram,
                 alpha=alpha, X=X, Y=Y, tau=tau)
    
    #==============================================================================    
    else:
        # Apply total-energy linear sampling method to four-dimensional space-time
        x = samplingGrid['x']
        y = samplingGrid['y']
        z = samplingGrid['z']
        tau = samplingGrid['tau']
        
        # Get number of sampling points in space and time
        Nx = len(x)
        Ny = len(y)
        Nz = len(z)
        Ntau = len(tau)
        deltaTau = tau[1] - tau[0] # step size for sampling in time
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Initialize the Histogram for storing images at each sampling point in time.
        # Initialize the Image (time-integrated Histogram with respect to L2 norm)
        Histogram = np.zeros((Nx, Ny, Nz, Ntau))
        Image = np.zeros(X.shape)
        
        if medium == 'constant':
            # Vezda will compute free-space test functions over the space-time
            # sampling grid via function calls to 'FundamentalSolutions.py'. This is
            # much more efficient than applying a forward and inverse FFT pair to
            # shift the test functions in time corresponding to different sampling
            # points in time. FFT pairs are only used when medium == variable.
            pulse = lambda t : pulseFun.pulse(t)
            
            # Previously computed test functions and parameters from pulseFun module
            # are stored in 'VZTestFuncs.npz'. If the current space-time sampling grid
            # and pulseFun module parameters are consistent with those stored in
            # 'VZTestFuncs.npz', then Vezda will load the previously computed test
            # functions. Otherwise, Vezda will recompute the test functions. This reduces
            # computational cost by only computing test functions when necessary.
            
            if Path('VZTestFuncs.npz').exists():
                print('\nDetected that free-space test functions have already been computed...')
                print('Checking consistency with current space-time sampling grid and pulse function...')
                TFDict = np.load('VZTestFuncs.npz')
                
                if samplingIsCurrent(TFDict, recordingTimes, velocity, tau, x, y, z, peakFreq, peakTime):
                    print('Moving forward to imaging algorithm...')
                    TFarray = TFDict['TFarray']
                    
                else:
                    print('Recomputing test functions...')
                    TFarray, samplingPoints = sampleSpaceTime(receiverPoints, recordingTimes,
                                                              velocity, tau, x, y, z, pulse)
                    
                    np.savez('VZTestFuncs.npz', TFarray=TFarray, time=recordingTimes,
                             peakFreq=peakFreq, peakTime=peakTime, velocity=velocity,
                             x=x, y=y, z=z, tau=tau, samplingPoints=samplingPoints)
                
            else:                
                print('\nComputing free-space test functions for the current space-time sampling grid...')
                TFarray, samplingPoints = sampleSpaceTime(receiverPoints, recordingTimes,
                                                          velocity, tau, x, y, z, pulse)
                    
                np.savez('VZTestFuncs.npz', TFarray=TFarray, time=recordingTimes,
                         peakFreq=peakFreq, peakTime=peakTime, velocity=velocity,
                         x=x, y=y, z=z, tau=tau, samplingPoints=samplingPoints)
            
            print('Localizing the source function...')
            if Ntau == 1:
                # Compute the Tikhonov-regularized solution to the near-field equation N * phi = tf.
                # 'tf' is a test function
                # 'alpha' is the regularization parameter
                # 'phi_alpha' is the regularized solution given 'alpha'
                print()
                TF = TFarray[:, :, :, 0]
                l = 0 # counter for spatial sampling points
                for ix in trange(Nx, desc='Solving system'):
                    for iy in range(Ny):
                        for iz in range(Nz):
                            tf = np.reshape(TF[:, :, l], (Nt * Nr, 1))
                            phi_alpha = Tikhonov(U, s, V, tf, alpha)
                            Image[ix, iy, iz] = 1.0 / (norm(phi_alpha) + eps)
                            l += 1
                
                Imin = np.min(Image)
                Imax = np.max(Image)
                Image = (Image - Imin) / (Imax - Imin + eps)
                Histogram[:, :, :, 0] = Image
                
            else:
                # Store spatial reconstruction of the source function for each
                # sampling point in time in Histogram
                # Compute time-integrated Image from Histogram using L2 norm
                # Discretize L2 integration using trapezoidal rule with 
                # uniform step size deltaTau
                firstIndicator = np.zeros(X.shape)
                lastIndicator = np.zeros(X.shape)
                firstTF = TFarray[:, :, :, 0]   # test function array for first time sample
                lastTF = TFarray[:, :, :, -1]   # test function array for last time sample
                
                # Compute the Tikhonov-regularized solution to the near-field equation N * phi = tf.
                # 'tf' is a test function
                # 'alpha' is the regularization parameter
                # 'phi_alpha' is the regularized solution given 'alpha'
                l = 0 # counter for spatial sampling points
                for ix in range(Nx):
                    for iy in range(Ny):
                        for iz in range(Nz):
                            tf1 = np.reshape(firstTF[:, :, l], (Nt * Nr, 1))
                            tf2 = np.reshape(lastTF[:, :, l], (Nt * Nr, 1))
                            phi_alpha1 = Tikhonov(U, s, V, tf1, alpha)
                            phi_alpha2 = Tikhonov(U, s, V, tf2, alpha)
                            firstIndicator[ix, iy, iz] = 1.0 / (norm(phi_alpha1) + eps)
                            lastIndicator[ix, iy, iz] = 1.0 / (norm(phi_alpha2) + eps)
                            l += 1
            
                Imin1 = np.min(firstIndicator)
                Imax1 = np.max(firstIndicator)
                firstIndicator = (firstIndicator - Imin1) / (Imax1 - Imin1 + eps)  # normalization
                Histogram[:, :, :, 0] = firstIndicator
                
                Imin2 = np.min(lastIndicator)
                Imax2 = np.max(lastIndicator)
                lastIndicator = (lastIndicator - Imin2) / (Imax2 - Imin2 + eps)  # normalization
                Histogram[:, :, :, -1] = lastIndicator
                
                Image += 0.5 * (firstIndicator**2 + lastIndicator**2)
                
                for it in trange(1, Ntau - 1, desc='Source-time integration'):
                    indicator = np.zeros(X.shape)
                    l = 0 # counter for spatial sampling points
                    for ix in trange(Nx, desc='Solving system', leave=False):
                        for iy in range(Ny):
                            for iz in range(Nz):
                                tf = np.reshape(TFarray[:, :, l, it], (Nt * Nr, 1))
                                phi_alpha = Tikhonov(U, s, V, tf, alpha)
                                indicator[ix, iy, iz] = 1.0 / (norm(phi_alpha) + eps)
                                l += 1
                                sleep(0.001)
                                
                    Imin = np.min(indicator)
                    Imax = np.max(indicator)
                    indicator = (indicator - Imin) / (Imax - Imin + eps)  # normalization
                    Histogram[:, :, it] = indicator
                    Image += indicator**2
                    
                Image *= deltaTau / T
                Image = np.sqrt(Image)
                        
        elif medium == 'variable':
            if 'testFuncs' in datadir:
                # TFarray computed outside of Vezda for variable media will only 
                # have 3 axes. A fourth axis will be added to accomodate for time shifts
                # corresponding to different sampling points in time
                tempTFarray = np.load(str(datadir['testFuncs']))
                
                # Apply the receiver window, if any
                tempTFarray = tempTFarray[rinterval, :, :]
                
                # Add new axis to TFarray to accomodate time shifts
                TFarray = np.zeros((Nr, Nt, Nx * Ny * Nz, Ntau))
                TFarray[:, :, :, 0] = tempTFarray[:, tinterval, :]
                
                # Load the sampling points
                samplingPoints = np.load(str(datadir['samplingPoints']))
            
            else:
                sys.exit(textwrap.dedent(
                        '''
                        FileNotFoundError: Attempted to load file containing test
                        functions, but no such file exists. If a file exists containing
                        the test functions, run 'vzdata --path=<path/to/data/>' command 
                        and specify the file containing the test functions when prompted.
                        Otherwise, specify 'no' when asked if a file containing the test
                        functions exists.
                        '''))
            
            userResponded = False
            print(textwrap.dedent(
                 '''
                 In what order was the sampling grid spanned to compute the test functions?
                 
                 Enter 'xyz' if for each x, for each y, loop over z. (Default)
                 Enter 'xzy' if for each x, for each z, loop over y.
                 Enter 'yxz' if for each y, for each x, loop over z.
                 Enter 'yzx' if for each y, for each z, loop over x.
                 Enter 'zxy' if for each z, for each x, loop over y.
                 Enter 'zyx' if for each z, for each y, loop over x
                 Enter 'q/quit' to abort the calculation.
                 '''))
            while userResponded == False:
                order = input('Order: ')
                if order == '' or order == 'xyz':
                    print('Proceeding with order \'xyz\'...')
                    print('Localizing the source function...')
                    if Ntau == 1:
                        # Compute the Tikhonov-regularized solution to the near-field equation N * phi = tf.
                        # 'tf' is a test function
                        # 'alpha' is the regularization parameter
                        # 'phi_alpha' is the regularized solution given 'alpha'
                        if tau[0] != 0:
                            TF = timeShift(tempTFarray, tau[0], dt)
                            TF = TF[:, tinterval, :]
                            TFarray[:, :, :, 0] = TF
                        else:
                            TF = TFarray[:, :, :, 0]
                        l = 0 # counter for spatial sampling points
                        for ix in trange(Nx, desc='Solving system'):
                            for iy in range(Ny):
                                for iz in range(Nz):
                                    tf = np.reshape(TF[:, :, l], (Nt * Nr, 1))
                                    phi_alpha = Tikhonov(U, s, V, tf, alpha)
                                    Image[ix, iy, iz] = 1.0 / (norm(phi_alpha) + eps)
                                    l += 1
                
                        Imin = np.min(Image)
                        Imax = np.max(Image)
                        Image = (Image - Imin) / (Imax - Imin + eps)
                        Histogram[:, :, :, 0] = Image
                        userResponded = True
                        break
                    
                    else:
                        # discretize using trapezoidal rule with uniform step size deltaTau
                        firstIndicator = np.zeros(X.shape)
                        lastIndicator = np.zeros(X.shape)
                        firstTF = TFarray[:, :, :, 0]
                        lastTF = timeShift(tempTFarray, tau[-1], dt)
                        lastTF = lastTF[:, tinterval, :]
                        TFarray[:, :, :, -1] = lastTF
                        
                        l = 0 # counter for spatial sampling points
                        for ix in range(Nx):
                            for iy in range(Ny):
                                for iz in range(Nz):
                                    tf1 = np.reshape(firstTF[:, :, l], (Nt * Nr, 1))
                                    tf2 = np.reshape(lastTF[:, :, l], (Nt * Nr, 1))
                                    phi_alpha1 = Tikhonov(U, s, V, tf1, alpha)
                                    phi_alpha2 = Tikhonov(U, s, V, tf2, alpha)
                                    firstIndicator[ix, iy, iz] = 1.0 / (norm(phi_alpha1) + eps)
                                    lastIndicator[ix, iy, iz] = 1.0 / (norm(phi_alpha2) + eps)
                                    l += 1
                                    
                        Imin1 = np.min(firstIndicator)
                        Imax1 = np.max(firstIndicator)
                        firstIndicator = (firstIndicator - Imin1) / (Imax1 - Imin1 + eps)  # normalization
                        Histogram[:, :, :, 0] = firstIndicator
                        
                        Imin2 = np.min(lastIndicator)
                        Imax2 = np.max(lastIndicator)
                        lastIndicator = (lastIndicator - Imin2) / (Imax2 - Imin2 + eps)  # normalization
                        Histogram[:, :, :, -1] = lastIndicator
                        
                        Image += 0.5 * (firstIndicator**2 + lastIndicator**2)
                        
                        for it in trange(1, Ntau - 1, desc='Source-time integration'):
                            TF = timeShift(tempTFarray, tau[it], dt)
                            TF = TF[:, tinterval, :]
                            TFarray[:, :, :, it] = TF
                            indicator = np.zeros(X.shape)
                            
                            l = 0 # counter for spatial sampling points
                            for ix in trange(Nx, desc='Solving system', leave=False):
                                for iy in range(Ny):
                                    for iz in range(Nz):
                                        tf = np.reshape(TF[:, :, l], (Nt * Nr, 1))
                                        phi_alpha = Tikhonov(U, s, V, tf, alpha)
                                        indicator[ix, iy, iz] = 1.0 / (norm(phi_alpha) + eps)
                                        l += 1
                                        sleep(0.001)
                                        
                            Imin = np.min(indicator)
                            Imax = np.max(indicator)
                            indicator = (indicator - Imin) / (Imax - Imin + eps)  # normalization
                            Histogram[:, :, :, it] = indicator
                            Image += indicator**2
                            
                        Image *= deltaTau / T
                        Image = np.sqrt(Image)
                        userResponded = True
                        break
                
                elif order == 'xzy':
                    print('Proceeding with order \'xzy\'...')
                    print('Localizing the source function...')
                    if Ntau == 1:
                        # Compute the Tikhonov-regularized solution to the near-field equation N * phi = tf.
                        # 'tf' is a test function
                        # 'alpha' is the regularization parameter
                        # 'phi_alpha' is the regularized solution given 'alpha'
                        if tau[0] != 0:
                            TF = timeShift(tempTFarray, tau[0], dt)
                            TF = TF[:, tinterval, :]
                            TFarray[:, :, :, 0] = TF
                        else:
                            TF = TFarray[:, :, :, 0]
                        l = 0 # counter for spatial sampling points
                        for ix in trange(Nx, desc='Solving system'):
                            for iz in range(Nz):
                                for iy in range(Ny):
                                    tf = np.reshape(TF[:, :, l], (Nt * Nr, 1))
                                    phi_alpha = Tikhonov(U, s, V, tf, alpha)
                                    Image[ix, iy, iz] = 1.0 / (norm(phi_alpha) + eps)
                                    l += 1
                
                        Imin = np.min(Image)
                        Imax = np.max(Image)
                        Image = (Image - Imin) / (Imax - Imin + eps)
                        Histogram[:, :, :, 0] = Image
                        userResponded = True
                        break
                    
                    else:
                        # discretize using trapezoidal rule with uniform step size deltaTau
                        firstIndicator = np.zeros(X.shape)
                        lastIndicator = np.zeros(X.shape)
                        firstTF = TFarray[:, :, :, 0]
                        lastTF = timeShift(tempTFarray, tau[-1], dt)
                        lastTF = lastTF[:, tinterval, :]
                        TFarray[:, :, :, -1] = lastTF
                        
                        l = 0 # counter for spatial sampling points
                        for ix in range(Nx):
                            for iz in range(Nz):
                                for iy in range(Ny):
                                    tf1 = np.reshape(firstTF[:, :, l], (Nt * Nr, 1))
                                    tf2 = np.reshape(lastTF[:, :, l], (Nt * Nr, 1))
                                    phi_alpha1 = Tikhonov(U, s, V, tf1, alpha)
                                    phi_alpha2 = Tikhonov(U, s, V, tf2, alpha)
                                    firstIndicator[ix, iy, iz] = 1.0 / (norm(phi_alpha1) + eps)
                                    lastIndicator[ix, iy, iz] = 1.0 / (norm(phi_alpha2) + eps)
                                    l += 1
                                    
                        Imin1 = np.min(firstIndicator)
                        Imax1 = np.max(firstIndicator)
                        firstIndicator = (firstIndicator - Imin1) / (Imax1 - Imin1 + eps)  # normalization
                        Histogram[:, :, :, 0] = firstIndicator
                        
                        Imin2 = np.min(lastIndicator)
                        Imax2 = np.max(lastIndicator)
                        lastIndicator = (lastIndicator - Imin2) / (Imax2 - Imin2 + eps)  # normalization
                        Histogram[:, :, :, -1] = np.sqrt(lastIndicator)
                        
                        Image += 0.5 * (firstIndicator**2 + lastIndicator**2)
                        
                        for it in trange(1, Ntau - 1, desc='Source-time integration'):
                            TF = timeShift(tempTFarray, tau[it], dt)
                            TF = TF[:, tinterval, :]
                            TFarray[:, :, :, it] = TF
                            indicator = np.zeros(X.shape)
                            
                            l = 0 # counter for spatial sampling points
                            for ix in trange(Nx, desc='Solving system', leave=False):
                                for iz in range(Nz):
                                    for iy in range(Ny):
                                        tf = np.reshape(TF[:, :, l], (Nt * Nr, 1))
                                        phi_alpha = Tikhonov(U, s, V, tf, alpha)
                                        indicator[ix, iy, iz] = 1.0 / (norm(phi_alpha) + eps)
                                        l += 1
                                        sleep(0.001)
                                        
                            Imin = np.min(indicator)
                            Imax = np.max(indicator)
                            indicator = (indicator - Imin) / (Imax - Imin + eps)  # normalization
                            Histogram[:, :, :, it] = indicator
                            Image += indicator**2
                            
                        Image *= deltaTau / T
                        Image = np.sqrt(Image)
                        userResponded = True
                        break
                
                elif order == 'yxz':
                    print('Proceeding with order \'yxz\'...')
                    print('Localizing the source function...')
                    if Ntau == 1:
                        # Compute the Tikhonov-regularized solution to the near-field equation N * phi = tf.
                        # 'tf' is a test function
                        # 'alpha' is the regularization parameter
                        # 'phi_alpha' is the regularized solution given 'alpha'
                        if tau[0] != 0:
                            TF = timeShift(tempTFarray, tau[0], dt)
                            TF = TF[:, tinterval, :]
                            TFarray[:, :, :, 0] = TF
                        else:
                            TF = TFarray[:, :, :, 0]
                        l = 0 # counter for spatial sampling points
                        for iy in trange(Ny, desc='Solving system'):
                            for ix in range(Nx):
                                for iz in range(Nz):
                                    tf = np.reshape(TF[:, :, l], (Nt * Nr, 1))
                                    phi_alpha = Tikhonov(U, s, V, tf, alpha)
                                    Image[ix, iy, iz] = 1.0 / (norm(phi_alpha) + eps)
                                    l += 1
                
                        Imin = np.min(Image)
                        Imax = np.max(Image)
                        Image = (Image - Imin) / (Imax - Imin + eps)
                        Histogram[:, :, :, 0] = Image
                        userResponded = True
                        break
                    
                    else:
                        # discretize using trapezoidal rule with uniform step size deltaTau
                        firstIndicator = np.zeros(X.shape)
                        lastIndicator = np.zeros(X.shape)
                        firstTF = TFarray[:, :, :, 0]
                        lastTF = timeShift(tempTFarray, tau[-1], dt)
                        lastTF = lastTF[:, tinterval, :]
                        TFarray[:, :, :, -1] = lastTF
                        
                        l = 0 # counter for spatial sampling points
                        for iy in range(Ny):
                            for ix in range(Nx):
                                for iz in range(Nz):
                                    tf1 = np.reshape(firstTF[:, :, l], (Nt * Nr, 1))
                                    tf2 = np.reshape(lastTF[:, :, l], (Nt * Nr, 1))
                                    phi_alpha1 = Tikhonov(U, s, V, tf1, alpha)
                                    phi_alpha2 = Tikhonov(U, s, V, tf2, alpha)
                                    firstIndicator[ix, iy, iz] = 1.0 / (norm(phi_alpha1) + eps)
                                    lastIndicator[ix, iy, iz] = 1.0 / (norm(phi_alpha2) + eps)
                                    l += 1
                                    
                        Imin1 = np.min(firstIndicator)
                        Imax1 = np.max(firstIndicator)
                        firstIndicator = (firstIndicator - Imin1) / (Imax1 - Imin1 + eps)  # normalization
                        Histogram[:, :, :, 0] = firstIndicator
                        
                        Imin2 = np.min(lastIndicator)
                        Imax2 = np.max(lastIndicator)
                        lastIndicator = (lastIndicator - Imin2) / (Imax2 - Imin2 + eps)  # normalization
                        Histogram[:, :, :, -1] = lastIndicator
                        
                        Image += 0.5 * (firstIndicator**2 + lastIndicator**2)
                        
                        for it in trange(1, Ntau - 1, desc='Source-time integration'):
                            TF = timeShift(tempTFarray, tau[it], dt)
                            TF = TF[:, tinterval, :]
                            TFarray[:, :, :, it] = TF
                            indicator = np.zeros(X.shape)
                            
                            l = 0 # counter for spatial sampling points
                            for iy in trange(Ny, desc='Solving system', leave=False):
                                for ix in range(Nx):
                                    for iz in range(Nz):
                                        tf = np.reshape(TF[:, :, l], (Nt * Nr, 1))
                                        phi_alpha = Tikhonov(U, s, V, tf, alpha)
                                        indicator[ix, iy, iz] = 1.0 / (norm(phi_alpha) + eps)
                                        l += 1
                                        sleep(0.001)
                                        
                            Imin = np.min(indicator)
                            Imax = np.max(indicator)
                            indicator = (indicator - Imin) / (Imax - Imin + eps)  # normalization
                            Histogram[:, :, :, it] = indicator
                            Image += indicator**2
                            
                        Image *= deltaTau / T
                        Image = np.sqrt(Image)
                        userResponded = True
                        break
                
                elif order == 'yzx':
                    print('Proceeding with order \'yzx\'...')
                    print('Localizing the source function...')
                    if Ntau == 1:
                        # Compute the Tikhonov-regularized solution to the near-field equation N * phi = tf.
                        # 'tf' is a test function
                        # 'alpha' is the regularization parameter
                        # 'phi_alpha' is the regularized solution given 'alpha'
                        if tau[0] != 0:
                            TF = timeShift(tempTFarray, tau[0], dt)
                            TF = TF[:, tinterval, :]
                            TFarray[:, :, :, 0] = TF
                        else:
                            TF = TFarray[:, :, :, 0]
                        l = 0 # counter for spatial sampling points
                        for iy in trange(Ny, desc='Solving system'):
                            for iz in range(Nz):
                                for ix in range(Nx):
                                    tf = np.reshape(TF[:, :, l], (Nt * Nr, 1))
                                    phi_alpha = Tikhonov(U, s, V, tf, alpha)
                                    Image[ix, iy, iz] = 1.0 / (norm(phi_alpha) + eps)
                                    l += 1
                
                        Imin = np.min(Image)
                        Imax = np.max(Image)
                        Image = (Image - Imin) / (Imax - Imin + eps)
                        Histogram[:, :, :, 0] = Image
                        userResponded = True
                        break
                    
                    else:
                        # discretize using trapezoidal rule with uniform step size deltaTau
                        firstIndicator = np.zeros(X.shape)
                        lastIndicator = np.zeros(X.shape)
                        firstTF = TFarray[:, :, :, 0]
                        lastTF = timeShift(tempTFarray, tau[-1], dt)
                        lastTF = lastTF[:, tinterval, :]
                        TFarray[:, :, :, -1] = lastTF
                        
                        l = 0 # counter for spatial sampling points
                        for iy in range(Ny):
                            for iz in range(Nz):
                                for ix in range(Nx):
                                    tf1 = np.reshape(firstTF[:, :, l], (Nt * Nr, 1))
                                    tf2 = np.reshape(lastTF[:, :, l], (Nt * Nr, 1))
                                    phi_alpha1 = Tikhonov(U, s, V, tf1, alpha)
                                    phi_alpha2 = Tikhonov(U, s, V, tf2, alpha)
                                    firstIndicator[ix, iy, iz] = 1.0 / (norm(phi_alpha1) + eps)
                                    lastIndicator[ix, iy, iz] = 1.0 / (norm(phi_alpha2) + eps)
                                    l += 1
                                    
                        Imin1 = np.min(firstIndicator)
                        Imax1 = np.max(firstIndicator)
                        firstIndicator = (firstIndicator - Imin1) / (Imax1 - Imin1 + eps)  # normalization
                        Histogram[:, :, :, 0] = firstIndicator
                        
                        Imin2 = np.min(lastIndicator)
                        Imax2 = np.max(lastIndicator)
                        lastIndicator = (lastIndicator - Imin2) / (Imax2 - Imin2 + eps)  # normalization
                        Histogram[:, :, :, -1] = lastIndicator
                        
                        Image += 0.5 * (firstIndicator**2 + lastIndicator**2)
                        
                        for it in trange(1, Ntau - 1, desc='Source-time integration'):
                            TF = timeShift(tempTFarray, tau[it], dt)
                            TF = TF[:, tinterval, :]
                            TFarray[:, :, :, it] = TF
                            indicator = np.zeros(X.shape)
                            
                            l = 0 # counter for spatial sampling points
                            for iy in trange(Ny, desc='Solving system', leave=False):
                                for iz in range(Nz):
                                    for ix in range(Nx):
                                        tf = np.reshape(TF[:, :, l], (Nt * Nr, 1))
                                        phi_alpha = Tikhonov(U, s, V, tf, alpha)
                                        indicator[ix, iy, iz] = 1.0 / (norm(phi_alpha) + eps)
                                        l += 1
                                        sleep(0.001)
                                        
                            Imin = np.min(indicator)
                            Imax = np.max(indicator)
                            indicator = (indicator - Imin) / (Imax - Imin + eps)  # normalization
                            Histogram[:, :, :, it] = indicator
                            Image += indicator**2
                            
                        Image *= deltaTau / T
                        Image = np.sqrt(Image)
                        userResponded = True
                        break
                
                elif order == 'zxy':
                    print('Proceeding with order \'zxy\'...')
                    print('Localizing the source function...')
                    if Ntau == 1:
                        # Compute the Tikhonov-regularized solution to the near-field equation N * phi = tf.
                        # 'tf' is a test function
                        # 'alpha' is the regularization parameter
                        # 'phi_alpha' is the regularized solution given 'alpha'
                        if tau[0] != 0:
                            TF = timeShift(tempTFarray, tau[0], dt)
                            TF = TF[:, tinterval, :]
                            TFarray[:, :, :, 0] = TF
                        else:
                            TF = TFarray[:, :, :, 0]
                        l = 0 # counter for spatial sampling points
                        for iz in trange(Nz, desc='Solving system'):
                            for ix in range(Nx):
                                for iy in range(Ny):
                                    tf = np.reshape(TF[:, :, l], (Nt * Nr, 1))
                                    phi_alpha = Tikhonov(U, s, V, tf, alpha)
                                    Image[ix, iy, iz] = 1.0 / (norm(phi_alpha) + eps)
                                    l += 1
                
                        Imin = np.min(Image)
                        Imax = np.max(Image)
                        Image = (Image - Imin) / (Imax - Imin + eps)
                        Histogram[:, :, :, 0] = Image
                        userResponded = True
                        break
                    
                    else:
                        # discretize using trapezoidal rule with uniform step size deltaTau
                        firstIndicator = np.zeros(X.shape)
                        lastIndicator = np.zeros(X.shape)
                        firstTF = TFarray[:, :, :, 0]
                        lastTF = timeShift(tempTFarray, tau[-1], dt)
                        lastTF = lastTF[:, tinterval, :]
                        TFarray[:, :, :, -1] = lastTF
                        
                        l = 0 # counter for spatial sampling points
                        for iz in range(Nz):
                            for ix in range(Nx):
                                for iy in range(Ny):
                                    tf1 = np.reshape(firstTF[:, :, l], (Nt * Nr, 1))
                                    tf2 = np.reshape(lastTF[:, :, l], (Nt * Nr, 1))
                                    phi_alpha1 = Tikhonov(U, s, V, tf1, alpha)
                                    phi_alpha2 = Tikhonov(U, s, V, tf2, alpha)
                                    firstIndicator[ix, iy, iz] = 1.0 / (norm(phi_alpha1) + eps)
                                    lastIndicator[ix, iy, iz] = 1.0 / (norm(phi_alpha2) + eps)
                                    l += 1
                                    
                        Imin1 = np.min(firstIndicator)
                        Imax1 = np.max(firstIndicator)
                        firstIndicator = (firstIndicator - Imin1) / (Imax1 - Imin1 + eps)  # normalization
                        Histogram[:, :, :, 0] = firstIndicator
                        
                        Imin2 = np.min(lastIndicator)
                        Imax2 = np.max(lastIndicator)
                        lastIndicator = (lastIndicator - Imin2) / (Imax2 - Imin2 + eps)  # normalization
                        Histogram[:, :, :, -1] = lastIndicator
                        
                        Image += 0.5 * (firstIndicator**2 + lastIndicator**2)
                        
                        for it in trange(1, Ntau - 1, desc='Source-time integration'):
                            TF = timeShift(tempTFarray, tau[it], dt)
                            TF = TF[:, tinterval, :]
                            TFarray[:, :, :, it] = TF
                            indicator = np.zeros(X.shape)
                            
                            l = 0 # counter for spatial sampling points
                            for iz in trange(Nz, desc='Solving system', leave=False):
                                for ix in range(Nx):
                                    for iy in range(Ny):
                                        tf = np.reshape(TF[:, :, l], (Nt * Nr, 1))
                                        phi_alpha = Tikhonov(U, s, V, tf, alpha)
                                        indicator[ix, iy, iz] = 1.0 / (norm(phi_alpha) + eps)
                                        l += 1
                                        sleep(0.001)
                                        
                            Imin = np.min(indicator)
                            Imax = np.max(indicator)
                            indicator = (indicator - Imin) / (Imax - Imin + eps)  # normalization
                            Histogram[:, :, :, it] = indicator
                            Image += indicator**2
                            
                        Image *= deltaTau / T
                        Image = np.sqrt(Image)
                        userResponded = True
                        break
                
                elif order == 'zyx':
                    print('Proceeding with order \'zyx\'...')
                    print('Localizing the source function...')
                    if Ntau == 1:
                        # Compute the Tikhonov-regularized solution to the near-field equation N * phi = tf.
                        # 'tf' is a test function
                        # 'alpha' is the regularization parameter
                        # 'phi_alpha' is the regularized solution given 'alpha'
                        if tau[0] != 0:
                            TF = timeShift(tempTFarray, tau[0], dt)
                            TF = TF[:, tinterval, :]
                            TFarray[:, :, :, 0] = TF
                        else:
                            TF = TFarray[:, :, :, 0]
                        l = 0 # counter for spatial sampling points
                        for iz in trange(Nz, desc='Solving system'):
                            for iy in range(Ny):
                                for ix in range(Nx):
                                    tf = np.reshape(TF[:, :, l], (Nt * Nr, 1))
                                    phi_alpha = Tikhonov(U, s, V, tf, alpha)
                                    Image[ix, iy, iz] = 1.0 / (norm(phi_alpha) + eps)
                                    l += 1
                
                        Imin = np.min(Image)
                        Imax = np.max(Image)
                        Image = (Image - Imin) / (Imax - Imin + eps)
                        Histogram[:, :, :, 0] = Image
                        userResponded = True
                        break
                    
                    else:
                        # discretize using trapezoidal rule with uniform step size deltaTau
                        firstIndicator = np.zeros(X.shape)
                        lastIndicator = np.zeros(X.shape)
                        firstTF = TFarray[:, :, :, 0]
                        lastTF = timeShift(tempTFarray, tau[-1], dt)
                        lastTF = lastTF[:, tinterval, :]
                        TFarray[:, :, :, -1] = lastTF
                        
                        l = 0 # counter for spatial sampling points
                        for iz in range(Nz):
                            for iy in range(Ny):
                                for ix in range(Nx):
                                    tf1 = np.reshape(firstTF[:, :, l], (Nt * Nr, 1))
                                    tf2 = np.reshape(lastTF[:, :, l], (Nt * Nr, 1))
                                    phi_alpha1 = Tikhonov(U, s, V, tf1, alpha)
                                    phi_alpha2 = Tikhonov(U, s, V, tf2, alpha)
                                    firstIndicator[ix, iy, iz] = 1.0 / (norm(phi_alpha1) + eps)
                                    lastIndicator[ix, iy, iz] = 1.0 / (norm(phi_alpha2) + eps)
                                    l += 1
                                    
                        Imin1 = np.min(firstIndicator)
                        Imax1 = np.max(firstIndicator)
                        firstIndicator = (firstIndicator - Imin1) / (Imax1 - Imin1 + eps)  # normalization
                        Histogram[:, :, :, 0] = firstIndicator
                        
                        Imin2 = np.min(lastIndicator)
                        Imax2 = np.max(lastIndicator)
                        lastIndicator = (lastIndicator - Imin2) / (Imax2 - Imin2 + eps)  # normalization
                        Histogram[:, :, :, -1] = lastIndicator
                        
                        Image += 0.5 * (firstIndicator**2 + lastIndicator**2)
                        
                        for it in trange(1, Ntau - 1, desc='Source-time integration'):
                            TF = timeShift(tempTFarray, tau[it], dt)
                            TF = TF[:, tinterval, :]
                            TFarray[:, :, :, it] = TF
                            indicator = np.zeros(X.shape)
                            
                            l = 0 # counter for spatial sampling points
                            for iz in trange(Nz, desc='Solving system', leave=False):
                                for iy in range(Ny):
                                    for ix in range(Nx):
                                        tf = np.reshape(TF[:, :, l], (Nt * Nr, 1))
                                        phi_alpha = Tikhonov(U, s, V, tf, alpha)
                                        indicator[ix, iy, iz] = 1.0 / (norm(phi_alpha) + eps)
                                        l += 1
                                        sleep(0.001)
                                        
                            Imin = np.min(indicator)
                            Imax = np.max(indicator)
                            indicator = (indicator - Imin) / (Imax - Imin + eps)  # normalization
                            Histogram[:, :, :, it] = indicator
                            Image += indicator**2
                            
                        Image *= deltaTau / T
                        Image = np.sqrt(Image)
                        userResponded = True
                        break
                
                elif order == 'q' or order == 'quit':
                    sys.exit('Aborting calculation.')
                else:
                    print(textwrap.dedent(
                         '''
                         Invalid response. Please enter one of the following:
                         
                         Enter 'xyz' if for each x, for each y, loop over z. (Default)
                         Enter 'xzy' if for each x, for each z, loop over y.
                         Enter 'yxz' if for each y, for each x, loop over z.
                         Enter 'yzx' if for each y, for each z, loop over x.
                         Enter 'zxy' if for each z, for each x, loop over y.
                         Enter 'zyx' if for each z, for each y, loop over x
                         Enter 'q/quit' to abort the calculation.
                         '''))
                        
            np.savez('VZTestFuncs.npz', TFarray=TFarray, time=recordingTimes,
                     peakFreq=peakFreq, peakTime=peakTime, velocity=velocity,
                     x=x, y=y, z=z, tau=tau, samplingPoints=samplingPoints)
        
        np.savez('imageNFE.npz', Image=Image, Histogram=Histogram,
                 alpha=alpha, X=X, Y=Y, Z=Z, tau=tau)
