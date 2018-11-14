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
from vezda.sampling_utils import samplingIsCurrent, sampleSpace
from vezda.math_utils import nextPow2, timeShift
from vezda.plot_utils import default_params
from scipy.linalg import norm
from tqdm import trange
from time import sleep
from pathlib import Path
import pickle

# pulseFun module contains information about the time-dependent
# pulse function used to generate the interrogating wave. This
# module is contained in the 'pulseFun.py' file provided by the
# user.
sys.path.append(os.getcwd())
import pulseFun

def solver(medium, s, Uh, V, alpha, domain):
    
    #==============================================================================
    # Load the receiver coordinates and recording times from the data directory
    datadir = np.load('datadir.npz')
    recordingTimes = np.load(str(datadir['recordingTimes']))
    receiverPoints = np.load(str(datadir['receivers']))
    
    # Compute length of time step.
    # This parameter is used for FFT shifting and time windowing
    dt = recordingTimes[1] - recordingTimes[0]
    
    # Load the windowing parameters for the receiver and time axes of
    # the 3D data array
    if Path('window.npz').exists():
        windowDict = np.load('window.npz')
        
        # Time window parameters (with units of time)
        tstart = windowDict['tstart']
        tstop = windowDict['tstop']
        
        # Convert time window parameters to corresponding array indices
        tstart = int(round(tstart / dt))
        tstop = int(round(tstop / dt))
        tstep = windowDict['tstep']
        
        # Receiver window parameters
        rstart = windowDict['rstart']
        rstop = windowDict['rstop']
        rstep = windowDict['rstep']
        
    else:
        # Set default window parameters if user did
        # not specify window parameters.
        
        # Time window parameters (integers corresponding to indices in an array)
        tstart = 0
        tstop = len(recordingTimes)
        tstep = 1
        
        # Receiver window parameters
        rstart = 0
        rstop = receiverPoints.shape[0]
        rstep = 1
        
    # Slice the recording times according to the time window parameters
    # to create a time window array
    tinterval = np.arange(tstart, tstop, tstep)
    recordingTimes = recordingTimes[tinterval]
    
    # Slice the receiverPoints array according to the receiver window parametes
    rinterval = np.arange(rstart, rstop, rstep)
    receiverPoints = receiverPoints[rinterval, :]
    
    Nr = receiverPoints.shape[0]
    Nt = len(recordingTimes)    # number of samples in time window
    
    # Get information about the pulse function used to 
    # generate the interrogating wave (These parameters are
    # used to help Vezda decide if it needs to recompute the 
    # test functions in the case the user changes these parameters.)
    velocity = pulseFun.velocity    # only used if medium == constant
    peakFreq = pulseFun.peakFreq    # peak frequency
    peakTime = pulseFun.peakTime    # time at which the pulse amplitude is maximum
    
    # Used for getting time and frequency units
    if Path('plotParams.pkl').exists():
        plotParams = pickle.load(open('plotParams.pkl', 'rb'))
    else:
        plotParams = default_params()
    
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
        # Apply linear sampling method to three-dimensional space-time
        x = samplingGrid['x']
        y = samplingGrid['y']
        tau = samplingGrid['tau']
        z = None
        
        # Get number of sampling points in space and time
        Nx = len(x)
        Ny = len(y)
        X, Y = np.meshgrid(x, y, indexing='ij')
        Ntau = len(tau)
        
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
                
                if samplingIsCurrent(TFDict, receiverPoints, recordingTimes, velocity, tau, x, y, z, peakFreq, peakTime):
                    print('Moving forward to imaging algorithm...')
                    TFarray = TFDict['TFarray']
                    
                else:
                    print('Recomputing test functions...')
                    if tau[0] != 0:
                        tu = plotParams['tu']
                        if tu != '':
                            print('Shifting test functions to source time %0.2f %s...' %(tau[0], tu))
                        else:
                            print('Shifting test functions to source time %0.2f...' %(tau[0]))
                        TFarray, samplingPoints = sampleSpace(receiverPoints, recordingTimes - tau[0], velocity,
                                                              x, y, z, pulse)
                    else:
                        TFarray, samplingPoints = sampleSpace(receiverPoints, recordingTimes, velocity,
                                                              x, y, z, pulse)
                    
                    np.savez('VZTestFuncs.npz', TFarray=TFarray, time=recordingTimes, receivers=receiverPoints,
                             peakFreq=peakFreq, peakTime=peakTime, velocity=velocity,
                             x=x, y=y, tau=tau, samplingPoints=samplingPoints)
                
            else:                
                print('\nComputing free-space test functions for the current space-time sampling grid...')
                if tau[0] != 0:
                    tu = plotParams['tu']
                    if tu != '':
                        print('Shifting test functions to source time %0.2f %s...' %(tau[0], tu))
                    else:
                        print('Shifting test functions to source time %0.2f...' %(tau[0]))
                    TFarray, samplingPoints = sampleSpace(receiverPoints, recordingTimes - tau[0], velocity,
                                                          x, y, z, pulse)
                else:
                    TFarray, samplingPoints = sampleSpace(receiverPoints, recordingTimes, velocity,
                                                          x, y, z, pulse)
                    
                np.savez('VZTestFuncs.npz', TFarray=TFarray, time=recordingTimes, receivers=receiverPoints,
                         peakFreq=peakFreq, peakTime=peakTime, velocity=velocity,
                         x=x, y=y, tau=tau, samplingPoints=samplingPoints)
            #==============================================================================
            if domain == 'freq':
                # Transform test functions into the frequency domain and bandpass for efficient solution
                # to near-field equation
            
                N = nextPow2(2 * Nt)
                TFarray = np.fft.rfft(TFarray, n=N, axis=1)
        
                if plotParams['fmax'] is None:
                    freqs = np.fft.rfftfreq(N, tstep * dt)
                    plotParams['fmax'] = np.max(freqs)
                    pickle.dump(plotParams, open('plotParams.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)
                
                # Apply the frequency window
                fmin = plotParams['fmin']
                fmax = plotParams['fmax']
                fu = plotParams['fu']   # frequency units (e.g., Hz)
                
                if fu != '':
                    print('Applying bandpass filter: [%0.2f %s, %0.2f %s]' %(fmin, fu, fmax, fu))
                else:
                    print('Applying bandpass filter: [%0.2f, %0.2f]' %(fmin, fmax))
            
                df = 1.0 / (N * tstep * dt)
                startIndex = int(round(fmin / df))
                stopIndex = int(round(fmax / df))
                    
                finterval = np.arange(startIndex, stopIndex, 1)
                TFarray = TFarray[:, finterval, :]
                    
            else:
                # Pad test functions in the time domain to length 2*Nt-1 (length of circular convolution)
                # to solve near-field equation
                
                N = Nt - 1
                npad = ((0, 0), (N, 0), (0, 0))
                TFarray = np.pad(TFarray, pad_width=npad, mode='constant', constant_values=0)
                
            N = TFarray.shape[1]
                
            #==============================================================================
            # Solve the near-field equation for each sampling point
            print('Localizing the source...')
            if Ntau == 1 or domain == 'freq':
                # Compute the Tikhonov-regularized solution to the near-field equation N * phi = tf.
                # 'tf' is a test function
                # 'alpha' is the regularization parameter
                # 'phi_alpha' is the regularized solution given 'alpha'
                
                k = 0 # counter for spatial sampling points
                for ix in trange(Nx, desc='Solving system'):
                    for iy in range(Ny):
                        tf = np.reshape(TFarray[:, :, k], (N * Nr, 1))
                        phi_alpha = Tikhonov(Uh, s, V, tf, alpha)
                        Image[ix, iy] = 1.0 / (norm(phi_alpha) + eps)
                        k += 1
                
                Imin = np.min(Image)
                Imax = np.max(Image)
                Image = (Image - Imin) / (Imax - Imin + eps)
                Histogram[:, :, 0] = Image
                
            else:
                # Store spatial reconstruction of the source for each
                # sampling point in time in Histogram
                # Compute time-integrated Image from Histogram using L2 norm
                # Discretize L2 integration using trapezoidal rule with 
                # uniform step size deltaTau
                firstIndicator = np.zeros(X.shape)
                lastIndicator = np.zeros(X.shape)
                
                tau0 = tau[0]
                lastTF = timeShift(TFarray, tau[-1] - tau0, tstep * dt) # test function array for last time sample
                
                # Compute the Tikhonov-regularized solution to the near-field equation N * phi = tf.
                # 'tf' is a test function
                # 'alpha' is the regularization parameter
                # 'phi_alpha' is the regularized solution given 'alpha'
                
                k = 0 # counter for spatial sampling points
                for ix in range(Nx):
                    for iy in range(Ny):
                        tf1 = np.reshape(TFarray[:, :, k], (N * Nr, 1))
                        tf2 = np.reshape(lastTF[:, :, k], (N * Nr, 1))
                        phi_alpha1 = Tikhonov(Uh, s, V, tf1, alpha)
                        phi_alpha2 = Tikhonov(Uh, s, V, tf2, alpha)
                        firstIndicator[ix, iy] = 1.0 / (norm(phi_alpha1) + eps)
                        lastIndicator[ix, iy] = 1.0 / (norm(phi_alpha2) + eps)
                        k += 1
                        
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
                    TF = timeShift(TFarray, tau[it] - tau0, tstep * dt)
                    k = 0 # counter for spatial sampling points
                    for ix in trange(Nx, desc='Solving system', leave=False):
                        for iy in range(Ny):
                            tf = np.reshape(TF[:, :, k], (N * Nr, 1))
                            phi_alpha = Tikhonov(Uh, s, V, tf, alpha)
                            indicator[ix, iy] = 1.0 / (norm(phi_alpha) + eps)
                            k += 1
                            sleep(0.001)
                
                    Imin = np.min(indicator)
                    Imax = np.max(indicator)
                    indicator = (indicator - Imin) / (Imax - Imin + eps)  # normalization
                    Histogram[:, :, it] = indicator
                    Image += indicator**2
                    
                Image = np.sqrt(Image / Ntau)
                    
        elif medium == 'variable':
            if 'testFuncs' in datadir:
                # Load the user-provided test functions
                TFarray = np.load(str(datadir['testFuncs']))
            
                # Apply the receiver/time windows, if any
                TFarray = TFarray[rinterval, :, :]
                TFarray = TFarray[:, tinterval, :]
                
                #==============================================================================
                if domain == 'freq':
                    # Transform test functions into the frequency domain and bandpass for efficient solution
                    # to near-field equation
            
                    N = nextPow2(2 * Nt)
                    TFarray = np.fft.rfft(TFarray, n=N, axis=1)
        
                    if plotParams['fmax'] is None:
                        freqs = np.fft.rfftfreq(N, tstep * dt)
                        plotParams['fmax'] = np.max(freqs)
                        pickle.dump(plotParams, open('plotParams.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)
                
                    # Apply the frequency window
                    fmin = plotParams['fmin']
                    fmax = plotParams['fmax']
                    fu = plotParams['fu']   # frequency units (e.g., Hz)
                
                    if fu != '':
                        print('Applying bandpass filter: [%0.2f %s, %0.2f %s]' %(fmin, fu, fmax, fu))
                    else:
                        print('Applying bandpass filter: [%0.2f, %0.2f]' %(fmin, fmax))
            
                    df = 1.0 / (N * tstep * dt)
                    startIndex = int(round(fmin / df))
                    stopIndex = int(round(fmax / df))
                    
                    finterval = np.arange(startIndex, stopIndex, 1)
                    TFarray = TFarray[:, finterval, :]
                        
                else:
                    # Pad test functions in the time domain to length 2*Nt-1 (length of circular convolution)
                    # to solve near-field equation
                
                    N = Nt - 1
                    npad = ((0, 0), (N, 0), (0, 0))
                    TFarray = np.pad(TFarray, pad_width=npad, mode='constant', constant_values=0)
                
                N = TFarray.shape[1]
                
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
                    print('Localizing the source...')
                    if Ntau == 1 or domain == 'freq':
                        # Compute the Tikhonov-regularized solution to the near-field equation N * phi = tf.
                        # 'tf' is a test function
                        # 'alpha' is the regularization parameter
                        # 'phi_alpha' is the regularized solution given 'alpha'
                        
                        k = 0 # counter for spatial sampling points
                        for ix in trange(Nx, desc='Solving system'):
                            for iy in range(Ny):
                                tf = np.reshape(TFarray[:, :, k], (N * Nr, 1))
                                phi_alpha = Tikhonov(Uh, s, V, tf, alpha)
                                Image[ix, iy] = 1.0 / (norm(phi_alpha) + eps)
                                k += 1
                
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
                        
                        tau0 = tau[0]
                        lastTF = timeShift(TFarray, tau[-1] - tau0, tstep * dt) # test function array for last time sample
                    
                        k = 0 # counter for spatial sampling points
                        for ix in range(Nx):
                            for iy in range(Ny):
                                tf1 = np.reshape(TFarray[:, :, k], (N * Nr, 1))
                                tf2 = np.reshape(lastTF[:, :, k], (N * Nr, 1))
                                phi_alpha1 = Tikhonov(Uh, s, V, tf1, alpha)
                                phi_alpha2 = Tikhonov(Uh, s, V, tf2, alpha)
                                firstIndicator[ix, iy] = 1.0 / (norm(phi_alpha1) + eps)
                                lastIndicator[ix, iy] = 1.0 / (norm(phi_alpha2) + eps)
                                k += 1
                            
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
                            TF = timeShift(TFarray, tau[it] - tau0, tstep * dt)
                            k = 0 # counter for spatial sampling points
                            for ix in trange(Nx, desc='Solving system', leave=False):
                                for iy in range(Ny):
                                    tf = np.reshape(TF[:, :, k], (N * Nr, 1))
                                    phi_alpha = Tikhonov(Uh, s, V, tf, alpha)
                                    indicator[ix, iy] = 1.0 / (norm(phi_alpha) + eps)
                                    k += 1
                                    sleep(0.001)
                                
                            Imin = np.min(indicator)
                            Imax = np.max(indicator)
                            indicator = (indicator - Imin) / (Imax - Imin + eps)  # normalization
                            Histogram[:, :, it] = indicator
                            Image += indicator**2
                            
                        Image = np.sqrt(Image / Ntau)
                        userResponded = True
                        break
                
                elif order == 'yx':
                    print('Proceeding with order \'yx\'...')
                    print('Localizing the source...')
                    if Ntau == 1 or domain == 'freq':
                        # Compute the Tikhonov-regularized solution to the near-field equation N * phi = tf.
                        # 'tf' is a test function
                        # 'alpha' is the regularization parameter
                        # 'phi_alpha' is the regularized solution given 'alpha'
                        
                        k = 0 # counter for spatial sampling points
                        for iy in trange(Ny, desc='Solving system'):
                            for ix in range(Nx):
                                tf = np.reshape(TFarray[:, :, k], (N * Nr, 1))
                                phi_alpha = Tikhonov(Uh, s, V, tf, alpha)
                                Image[ix, iy] = 1.0 / (norm(phi_alpha) + eps)
                                k += 1
                
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
                        
                        tau0 = tau[0]
                        lastTF = timeShift(TFarray, tau[-1] - tau0, tstep * dt) # test function array for last time sample
                    
                        k = 0 # counter for spatial sampling points
                        for iy in range(Ny):
                            for ix in range(Nx):
                                tf1 = np.reshape(TFarray[:, :, k], (N * Nr, 1))
                                tf2 = np.reshape(lastTF[:, :, k], (N * Nr, 1))
                                phi_alpha1 = Tikhonov(Uh, s, V, tf1, alpha)
                                phi_alpha2 = Tikhonov(Uh, s, V, tf2, alpha)
                                firstIndicator[ix, iy] = 1.0 / (norm(phi_alpha1) + eps)
                                lastIndicator[ix, iy] = 1.0 / (norm(phi_alpha2) + eps)
                                k += 1
                            
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
                            TF = timeShift(TFarray, tau[it] - tau0, tstep * dt)
                            k = 0 # counter for spatial sampling points
                            for iy in trange(Ny, desc='Solving system', leave=False):
                                for ix in range(Nx):
                                    tf = np.reshape(TF[:, :, k], (N * Nr, 1))
                                    phi_alpha = Tikhonov(Uh, s, V, tf, alpha)
                                    indicator[ix, iy] = 1.0 / (norm(phi_alpha) + eps)
                                    k += 1
                                    sleep(0.001)
                                
                            Imin = np.min(indicator)
                            Imax = np.max(indicator)
                            indicator = (indicator - Imin) / (Imax - Imin + eps)  # normalization
                            Histogram[:, :, it] = indicator
                            Image += indicator**2
                            
                        Image = np.sqrt(Image / Ntau)
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
            
        np.savez('imageNFE.npz', Image=Image, Histogram=Histogram,
                 alpha=alpha, X=X, Y=Y, tau=tau)
    
    #==============================================================================    
    else:                
        # Apply linear sampling method to four-dimensional space-time
        x = samplingGrid['x']
        y = samplingGrid['y']
        z = samplingGrid['z']
        tau = samplingGrid['tau']
        
        # Get number of sampling points in space and time
        Nx = len(x)
        Ny = len(y)
        Nz = len(z)
        Ntau = len(tau)
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
                
                if samplingIsCurrent(TFDict, receiverPoints, recordingTimes, velocity, tau, x, y, z, peakFreq, peakTime):
                    print('Moving forward to imaging algorithm...')
                    TFarray = TFDict['TFarray']
                    
                else:
                    print('Recomputing test functions...')
                    if tau[0] != 0:
                        tu = plotParams['tu']
                        if tu != '':
                            print('Shifting test functions to source time %0.2f %s...' %(tau[0], tu))
                        else:
                            print('Shifting test functions to source time %0.2f...' %(tau[0]))
                        TFarray, samplingPoints = sampleSpace(receiverPoints, recordingTimes - tau[0], velocity,
                                                              x, y, z, pulse)
                    else:
                        TFarray, samplingPoints = sampleSpace(receiverPoints, recordingTimes, velocity,
                                                              x, y, z, pulse)
                    
                    np.savez('VZTestFuncs.npz', TFarray=TFarray, time=recordingTimes, receivers=receiverPoints,
                             peakFreq=peakFreq, peakTime=peakTime, velocity=velocity,
                             x=x, y=y, z=z, tau=tau, samplingPoints=samplingPoints)
                
            else:                
                print('\nComputing free-space test functions for the current space-time sampling grid...')
                if tau[0] != 0:
                    tu = plotParams['tu']
                    if tu != '':
                        print('Shifting test functions to source time %0.2f %s...' %(tau[0], tu))
                    else:
                        print('Shifting test functions to source time %0.2f...' %(tau[0]))
                    TFarray, samplingPoints = sampleSpace(receiverPoints, recordingTimes - tau[0], velocity,
                                                          x, y, z, pulse)
                else:
                    TFarray, samplingPoints = sampleSpace(receiverPoints, recordingTimes, velocity,
                                                          x, y, z, pulse)
                    
                np.savez('VZTestFuncs.npz', TFarray=TFarray, time=recordingTimes, receivers=receiverPoints,
                         peakFreq=peakFreq, peakTime=peakTime, velocity=velocity,
                         x=x, y=y, z=z, tau=tau, samplingPoints=samplingPoints)
                
            #==============================================================================
            if domain == 'freq':
                # Transform test functions into the frequency domain and bandpass for efficient solution
                # to near-field equation
            
                N = nextPow2(2 * Nt)
                TFarray = np.fft.rfft(TFarray, n=N, axis=1)
        
                if plotParams['fmax'] is None:
                    freqs = np.fft.rfftfreq(N, tstep * dt)
                    plotParams['fmax'] = np.max(freqs)
                    pickle.dump(plotParams, open('plotParams.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)
                
                # Apply the frequency window
                fmin = plotParams['fmin']
                fmax = plotParams['fmax']
                fu = plotParams['fu']   # frequency units (e.g., Hz)
                
                if fu != '':
                    print('Applying bandpass filter: [%0.2f %s, %0.2f %s]' %(fmin, fu, fmax, fu))
                else:
                    print('Applying bandpass filter: [%0.2f, %0.2f]' %(fmin, fmax))
            
                df = 1.0 / (N * tstep * dt)
                startIndex = int(round(fmin / df))
                stopIndex = int(round(fmax / df))
                    
                finterval = np.arange(startIndex, stopIndex, 1)
                TFarray = TFarray[:, finterval, :]
                    
            else:
                # Pad test functions in the time domain to length 2*Nt-1 (length of circular convolution)
                # to solve near-field equation
                
                N = Nt - 1
                npad = ((0, 0), (N, 0), (0, 0))
                TFarray = np.pad(TFarray, pad_width=npad, mode='constant', constant_values=0)
                
            N = TFarray.shape[1]
            #==============================================================================
            # Solve the near-field equation for each sampling point
            print('Localizing the source...')
            if Ntau == 1 or domain == 'freq':
                # Compute the Tikhonov-regularized solution to the near-field equation N * phi = tf.
                # 'tf' is a test function
                # 'alpha' is the regularization parameter
                # 'phi_alpha' is the regularized solution given 'alpha'
                
                k = 0 # counter for spatial sampling points
                for ix in trange(Nx, desc='Solving system'):
                    for iy in range(Ny):
                        for iz in range(Nz):
                            tf = np.reshape(TFarray[:, :, k], (N * Nr, 1))
                            phi_alpha = Tikhonov(Uh, s, V, tf, alpha)
                            Image[ix, iy, iz] = 1.0 / (norm(phi_alpha) + eps)
                            k += 1
                
                Imin = np.min(Image)
                Imax = np.max(Image)
                Image = (Image - Imin) / (Imax - Imin + eps)
                Histogram[:, :, 0] = Image
                
            else:
                # Store spatial reconstruction of the source for each
                # sampling point in time in Histogram
                # Compute time-integrated Image from Histogram using L2 norm
                # Discretize L2 integration using trapezoidal rule with 
                # uniform step size deltaTau
                firstIndicator = np.zeros(X.shape)
                lastIndicator = np.zeros(X.shape)
                
                tau0 = tau[0]
                lastTF = timeShift(TFarray, tau[-1] - tau0, tstep * dt) # test function array for last time sample
                
                # Compute the Tikhonov-regularized solution to the near-field equation N * phi = tf.
                # 'tf' is a test function
                # 'alpha' is the regularization parameter
                # 'phi_alpha' is the regularized solution given 'alpha'
                
                k = 0 # counter for spatial sampling points
                for ix in range(Nx):
                    for iy in range(Ny):
                        for iz in range(Nz):
                            tf1 = np.reshape(TFarray[:, :, k], (N * Nr, 1))
                            tf2 = np.reshape(lastTF[:, :, k], (N * Nr, 1))
                            phi_alpha1 = Tikhonov(Uh, s, V, tf1, alpha)
                            phi_alpha2 = Tikhonov(Uh, s, V, tf2, alpha)
                            firstIndicator[ix, iy, iz] = 1.0 / (norm(phi_alpha1) + eps)
                            lastIndicator[ix, iy, iz] = 1.0 / (norm(phi_alpha2) + eps)
                            k += 1
                        
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
                    TF = timeShift(TFarray, tau[it] - tau0, tstep * dt)
                    k = 0 # counter for spatial sampling points
                    for ix in trange(Nx, desc='Solving system', leave=False):
                        for iy in range(Ny):
                            for iz in range(Nz):
                                tf = np.reshape(TF[:, :, k], (N * Nr, 1))
                                phi_alpha = Tikhonov(Uh, s, V, tf, alpha)
                                indicator[ix, iy, iz] = 1.0 / (norm(phi_alpha) + eps)
                                k += 1
                                sleep(0.001)
                
                    Imin = np.min(indicator)
                    Imax = np.max(indicator)
                    indicator = (indicator - Imin) / (Imax - Imin + eps)  # normalization
                    Histogram[:, :, it] = indicator
                    Image += indicator**2
                    
                Image = np.sqrt(Image / Ntau)
                        
        elif medium == 'variable':
            if 'testFuncs' in datadir:
                # Load the user-provided test functions
                TFarray = np.load(str(datadir['testFuncs']))
            
                # Apply the receiver/time windows, if any
                TFarray = TFarray[rinterval, :, :]
                TFarray = TFarray[:, tinterval, :]
                
                #==============================================================================
                if domain == 'freq':
                    # Transform test functions into the frequency domain and bandpass for efficient solution
                    # to near-field equation
            
                    N = nextPow2(2 * Nt)
                    TFarray = np.fft.rfft(TFarray, n=N, axis=1)
        
                    if plotParams['fmax'] is None:
                        freqs = np.fft.rfftfreq(N, tstep * dt)
                        plotParams['fmax'] = np.max(freqs)
                        pickle.dump(plotParams, open('plotParams.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)
                
                    # Apply the frequency window
                    fmin = plotParams['fmin']
                    fmax = plotParams['fmax']
                    fu = plotParams['fu']   # frequency units (e.g., Hz)
                
                    if fu != '':
                        print('Applying bandpass filter: [%0.2f %s, %0.2f %s]' %(fmin, fu, fmax, fu))
                    else:
                        print('Applying bandpass filter: [%0.2f, %0.2f]' %(fmin, fmax))
            
                    df = 1.0 / (N * tstep * dt)
                    startIndex = int(round(fmin / df))
                    stopIndex = int(round(fmax / df))
                    
                    finterval = np.arange(startIndex, stopIndex, 1)
                    TFarray = TFarray[:, finterval, :]
                        
                else:
                    # Pad test functions in the time domain to length 2*Nt-1 (length of circular convolution)
                    # to solve near-field equation
                
                    N = Nt - 1
                    npad = ((0, 0), (N, 0), (0, 0))
                    TFarray = np.pad(TFarray, pad_width=npad, mode='constant', constant_values=0)
                
                N = TFarray.shape[1]
                
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
                    print('Localizing the source...')
                    if Ntau == 1 or domain == 'freq':
                        # Compute the Tikhonov-regularized solution to the near-field equation N * phi = tf.
                        # 'tf' is a test function
                        # 'alpha' is the regularization parameter
                        # 'phi_alpha' is the regularized solution given 'alpha'
                        
                        k = 0 # counter for spatial sampling points
                        for ix in trange(Nx, desc='Solving system'):
                            for iy in range(Ny):
                                for iz in range(Nz):
                                    tf = np.reshape(TFarray[:, :, k], (N * Nr, 1))
                                    phi_alpha = Tikhonov(Uh, s, V, tf, alpha)
                                    Image[ix, iy, iz] = 1.0 / (norm(phi_alpha) + eps)
                                    k += 1
                
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
                        
                        tau0 = tau[0]
                        lastTF = timeShift(TFarray, tau[-1] - tau0, tstep * dt) # test function array for last time sample
                    
                        k = 0 # counter for spatial sampling points
                        for ix in range(Nx):
                            for iy in range(Ny):
                                for iz in range(Nz):
                                    tf1 = np.reshape(TFarray[:, :, k], (N * Nr, 1))
                                    tf2 = np.reshape(lastTF[:, :, k], (N * Nr, 1))
                                    phi_alpha1 = Tikhonov(Uh, s, V, tf1, alpha)
                                    phi_alpha2 = Tikhonov(Uh, s, V, tf2, alpha)
                                    firstIndicator[ix, iy, iz] = 1.0 / (norm(phi_alpha1) + eps)
                                    lastIndicator[ix, iy, iz] = 1.0 / (norm(phi_alpha2) + eps)
                                    k += 1
                            
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
                            TF = timeShift(TFarray, tau[it] - tau0, tstep * dt)
                            k = 0 # counter for spatial sampling points
                            for ix in trange(Nx, desc='Solving system', leave=False):
                                for iy in range(Ny):
                                    for iz in range(Nz):
                                        tf = np.reshape(TF[:, :, k], (N * Nr, 1))
                                        phi_alpha = Tikhonov(Uh, s, V, tf, alpha)
                                        indicator[ix, iy, iz] = 1.0 / (norm(phi_alpha) + eps)
                                        k += 1
                                        sleep(0.001)
                                
                            Imin = np.min(indicator)
                            Imax = np.max(indicator)
                            indicator = (indicator - Imin) / (Imax - Imin + eps)  # normalization
                            Histogram[:, :, it] = indicator
                            Image += indicator**2
                            
                        Image = np.sqrt(Image / Ntau)
                        userResponded = True
                        break
                
                elif order == 'xzy':
                    print('Proceeding with order \'xzy\'...')
                    print('Localizing the source...')
                    if Ntau == 1 or domain == 'freq':
                        # Compute the Tikhonov-regularized solution to the near-field equation N * phi = tf.
                        # 'tf' is a test function
                        # 'alpha' is the regularization parameter
                        # 'phi_alpha' is the regularized solution given 'alpha'
                        
                        k = 0 # counter for spatial sampling points
                        for ix in trange(Nx, desc='Solving system'):
                            for iz in range(Nz):
                                for iy in range(Ny):
                                    tf = np.reshape(TFarray[:, :, k], (N * Nr, 1))
                                    phi_alpha = Tikhonov(Uh, s, V, tf, alpha)
                                    Image[ix, iy, iz] = 1.0 / (norm(phi_alpha) + eps)
                                    k += 1
                
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
                        
                        tau0 = tau[0]
                        lastTF = timeShift(TFarray, tau[-1] - tau0, tstep * dt) # test function array for last time sample
                    
                        k = 0 # counter for spatial sampling points
                        for ix in range(Nx):
                            for iz in range(Nz):
                                for iy in range(Ny):
                                    tf1 = np.reshape(TFarray[:, :, k], (N * Nr, 1))
                                    tf2 = np.reshape(lastTF[:, :, k], (N * Nr, 1))
                                    phi_alpha1 = Tikhonov(Uh, s, V, tf1, alpha)
                                    phi_alpha2 = Tikhonov(Uh, s, V, tf2, alpha)
                                    firstIndicator[ix, iy, iz] = 1.0 / (norm(phi_alpha1) + eps)
                                    lastIndicator[ix, iy, iz] = 1.0 / (norm(phi_alpha2) + eps)
                                    k += 1
                            
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
                            TF = timeShift(TFarray, tau[it] - tau0, tstep * dt)
                            k = 0 # counter for spatial sampling points
                            for ix in trange(Nx, desc='Solving system', leave=False):
                                for iz in range(Nz):
                                    for iy in range(Ny):
                                        tf = np.reshape(TF[:, :, k], (N * Nr, 1))
                                        phi_alpha = Tikhonov(Uh, s, V, tf, alpha)
                                        indicator[ix, iy, iz] = 1.0 / (norm(phi_alpha) + eps)
                                        k += 1
                                        sleep(0.001)
                                
                            Imin = np.min(indicator)
                            Imax = np.max(indicator)
                            indicator = (indicator - Imin) / (Imax - Imin + eps)  # normalization
                            Histogram[:, :, it] = indicator
                            Image += indicator**2
                            
                        Image = np.sqrt(Image / Ntau)
                        userResponded = True
                        break
                
                elif order == 'yxz':
                    print('Proceeding with order \'yxz\'...')
                    print('Localizing the source...')
                    if Ntau == 1 or domain == 'freq':
                        # Compute the Tikhonov-regularized solution to the near-field equation N * phi = tf.
                        # 'tf' is a test function
                        # 'alpha' is the regularization parameter
                        # 'phi_alpha' is the regularized solution given 'alpha'
                        
                        k = 0 # counter for spatial sampling points
                        for iy in trange(Ny, desc='Solving system'):
                            for ix in range(Nx):
                                for iz in range(Nz):
                                    tf = np.reshape(TFarray[:, :, k], (N * Nr, 1))
                                    phi_alpha = Tikhonov(Uh, s, V, tf, alpha)
                                    Image[ix, iy, iz] = 1.0 / (norm(phi_alpha) + eps)
                                    k += 1
                
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
                        
                        tau0 = tau[0]
                        lastTF = timeShift(TFarray, tau[-1] - tau0, tstep * dt) # test function array for last time sample
                    
                        k = 0 # counter for spatial sampling points
                        for iy in range(Ny):
                            for ix in range(Nx):
                                for iz in range(Nz):
                                    tf1 = np.reshape(TFarray[:, :, k], (N * Nr, 1))
                                    tf2 = np.reshape(lastTF[:, :, k], (N * Nr, 1))
                                    phi_alpha1 = Tikhonov(Uh, s, V, tf1, alpha)
                                    phi_alpha2 = Tikhonov(Uh, s, V, tf2, alpha)
                                    firstIndicator[ix, iy, iz] = 1.0 / (norm(phi_alpha1) + eps)
                                    lastIndicator[ix, iy, iz] = 1.0 / (norm(phi_alpha2) + eps)
                                    k += 1
                            
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
                            TF = timeShift(TFarray, tau[it] - tau0, tstep * dt)
                            k = 0 # counter for spatial sampling points
                            for iy in trange(Ny, desc='Solving system', leave=False):
                                for ix in range(Nx):
                                    for iz in range(Nz):
                                        tf = np.reshape(TF[:, :, k], (N * Nr, 1))
                                        phi_alpha = Tikhonov(Uh, s, V, tf, alpha)
                                        indicator[ix, iy, iz] = 1.0 / (norm(phi_alpha) + eps)
                                        k += 1
                                        sleep(0.001)
                                
                            Imin = np.min(indicator)
                            Imax = np.max(indicator)
                            indicator = (indicator - Imin) / (Imax - Imin + eps)  # normalization
                            Histogram[:, :, it] = indicator
                            Image += indicator**2
                            
                        Image = np.sqrt(Image / Ntau)
                        userResponded = True
                        break
                
                elif order == 'yzx':
                    print('Proceeding with order \'yzx\'...')
                    print('Localizing the source...')
                    if Ntau == 1 or domain == 'freq':
                        # Compute the Tikhonov-regularized solution to the near-field equation N * phi = tf.
                        # 'tf' is a test function
                        # 'alpha' is the regularization parameter
                        # 'phi_alpha' is the regularized solution given 'alpha'
                        
                        k = 0 # counter for spatial sampling points
                        for iy in trange(Ny, desc='Solving system'):
                            for iz in range(Nz):
                                for ix in range(Nx):
                                    tf = np.reshape(TFarray[:, :, k], (N * Nr, 1))
                                    phi_alpha = Tikhonov(Uh, s, V, tf, alpha)
                                    Image[ix, iy, iz] = 1.0 / (norm(phi_alpha) + eps)
                                    k += 1
                
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
                        
                        tau0 = tau[0]
                        lastTF = timeShift(TFarray, tau[-1] - tau0, tstep * dt) # test function array for last time sample
                    
                        k = 0 # counter for spatial sampling points
                        for iy in range(Ny):
                            for iz in range(Nz):
                                for ix in range(Nx):
                                    tf1 = np.reshape(TFarray[:, :, k], (N * Nr, 1))
                                    tf2 = np.reshape(lastTF[:, :, k], (N * Nr, 1))
                                    phi_alpha1 = Tikhonov(Uh, s, V, tf1, alpha)
                                    phi_alpha2 = Tikhonov(Uh, s, V, tf2, alpha)
                                    firstIndicator[ix, iy, iz] = 1.0 / (norm(phi_alpha1) + eps)
                                    lastIndicator[ix, iy, iz] = 1.0 / (norm(phi_alpha2) + eps)
                                    k += 1
                            
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
                            TF = timeShift(TFarray, tau[it] - tau0, tstep * dt)
                            k = 0 # counter for spatial sampling points
                            for iy in trange(Ny, desc='Solving system', leave=False):
                                for iz in range(Nz):
                                    for ix in range(Nx):
                                        tf = np.reshape(TF[:, :, k], (N * Nr, 1))
                                        phi_alpha = Tikhonov(Uh, s, V, tf, alpha)
                                        indicator[ix, iy, iz] = 1.0 / (norm(phi_alpha) + eps)
                                        k += 1
                                        sleep(0.001)
                                
                            Imin = np.min(indicator)
                            Imax = np.max(indicator)
                            indicator = (indicator - Imin) / (Imax - Imin + eps)  # normalization
                            Histogram[:, :, it] = indicator
                            Image += indicator**2
                            
                        Image = np.sqrt(Image / Ntau)
                        userResponded = True
                        break
                
                elif order == 'zxy':
                    print('Proceeding with order \'zxy\'...')
                    print('Localizing the source...')
                    if Ntau == 1 or domain == 'freq':
                        # Compute the Tikhonov-regularized solution to the near-field equation N * phi = tf.
                        # 'tf' is a test function
                        # 'alpha' is the regularization parameter
                        # 'phi_alpha' is the regularized solution given 'alpha'
                        
                        k = 0 # counter for spatial sampling points
                        for iz in trange(Nz, desc='Solving system'):
                            for ix in range(Nx):
                                for iy in range(Ny):
                                    tf = np.reshape(TFarray[:, :, k], (N * Nr, 1))
                                    phi_alpha = Tikhonov(Uh, s, V, tf, alpha)
                                    Image[ix, iy, iz] = 1.0 / (norm(phi_alpha) + eps)
                                    k += 1
                
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
                        
                        tau0 = tau[0]
                        lastTF = timeShift(TFarray, tau[-1] - tau0, tstep * dt) # test function array for last time sample
                    
                        k = 0 # counter for spatial sampling points
                        for iz in range(Nz):
                            for ix in range(Nx):
                                for iy in range(Ny):
                                    tf1 = np.reshape(TFarray[:, :, k], (N * Nr, 1))
                                    tf2 = np.reshape(lastTF[:, :, k], (N * Nr, 1))
                                    phi_alpha1 = Tikhonov(Uh, s, V, tf1, alpha)
                                    phi_alpha2 = Tikhonov(Uh, s, V, tf2, alpha)
                                    firstIndicator[ix, iy, iz] = 1.0 / (norm(phi_alpha1) + eps)
                                    lastIndicator[ix, iy, iz] = 1.0 / (norm(phi_alpha2) + eps)
                                    k += 1
                            
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
                            TF = timeShift(TFarray, tau[it] - tau0, tstep * dt)
                            k = 0 # counter for spatial sampling points
                            for iz in trange(Nz, desc='Solving system', leave=False):
                                for ix in range(Nx):
                                    for iy in range(Ny):
                                        tf = np.reshape(TF[:, :, k], (N * Nr, 1))
                                        phi_alpha = Tikhonov(Uh, s, V, tf, alpha)
                                        indicator[ix, iy, iz] = 1.0 / (norm(phi_alpha) + eps)
                                        k += 1
                                        sleep(0.001)
                                
                            Imin = np.min(indicator)
                            Imax = np.max(indicator)
                            indicator = (indicator - Imin) / (Imax - Imin + eps)  # normalization
                            Histogram[:, :, it] = indicator
                            Image += indicator**2
                            
                        Image = np.sqrt(Image / Ntau)
                        userResponded = True
                        break
                
                elif order == 'zyx':
                    print('Proceeding with order \'zyx\'...')
                    print('Localizing the source...')
                    if Ntau == 1 or domain == 'freq':
                        # Compute the Tikhonov-regularized solution to the near-field equation N * phi = tf.
                        # 'tf' is a test function
                        # 'alpha' is the regularization parameter
                        # 'phi_alpha' is the regularized solution given 'alpha'
                        
                        k = 0 # counter for spatial sampling points
                        for iz in trange(Nz, desc='Solving system'):
                            for iy in range(Ny):
                                for ix in range(Nx):
                                    tf = np.reshape(TFarray[:, :, k], (N * Nr, 1))
                                    phi_alpha = Tikhonov(Uh, s, V, tf, alpha)
                                    Image[ix, iy, iz] = 1.0 / (norm(phi_alpha) + eps)
                                    k += 1
                
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
                        
                        tau0 = tau[0]
                        lastTF = timeShift(TFarray, tau[-1] - tau0, tstep * dt) # test function array for last time sample
                    
                        k = 0 # counter for spatial sampling points
                        for iz in range(Nz):
                            for iy in range(Ny):
                                for ix in range(Nx):
                                    tf1 = np.reshape(TFarray[:, :, k], (N * Nr, 1))
                                    tf2 = np.reshape(lastTF[:, :, k], (N * Nr, 1))
                                    phi_alpha1 = Tikhonov(Uh, s, V, tf1, alpha)
                                    phi_alpha2 = Tikhonov(Uh, s, V, tf2, alpha)
                                    firstIndicator[ix, iy, iz] = 1.0 / (norm(phi_alpha1) + eps)
                                    lastIndicator[ix, iy, iz] = 1.0 / (norm(phi_alpha2) + eps)
                                    k += 1
                            
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
                            TF = timeShift(TFarray, tau[it] - tau0, tstep * dt)
                            k = 0 # counter for spatial sampling points
                            for iz in trange(Nz, desc='Solving system', leave=False):
                                for iy in range(Ny):
                                    for ix in range(Nx):
                                        tf = np.reshape(TF[:, :, k], (N * Nr, 1))
                                        phi_alpha = Tikhonov(Uh, s, V, tf, alpha)
                                        indicator[ix, iy, iz] = 1.0 / (norm(phi_alpha) + eps)
                                        k += 1
                                        sleep(0.001)
                                
                            Imin = np.min(indicator)
                            Imax = np.max(indicator)
                            indicator = (indicator - Imin) / (Imax - Imin + eps)  # normalization
                            Histogram[:, :, it] = indicator
                            Image += indicator**2
                            
                        Image = np.sqrt(Image / Ntau)
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
        
        np.savez('imageNFE.npz', Image=Image, Histogram=Histogram,
                 alpha=alpha, X=X, Y=Y, Z=Z, tau=tau)
