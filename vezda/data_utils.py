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
import pickle
from pathlib import Path
import textwrap
from vezda.math_utils import nextPow2
from vezda.signal_utils import tukey_taper
from vezda.sampling_utils import samplingIsCurrent, sampleSpace
from vezda.plot_utils import default_params
sys.path.append(os.getcwd())
import pulseFun

datadir = np.load('datadir.npz')

# Used for getting time and frequency units
if Path('plotParams.pkl').exists():
    plotParams = pickle.load(open('plotParams.pkl', 'rb'))
else:
    plotParams = default_params()

def load_data(domain, verbose=False):
    # load the recorded data    
    print('Loading recorded waveforms...')
    if Path('noisyData.npz').exists():
        userResponded = False
        print(textwrap.dedent(
              '''
              Detected that band-limited noise has been added to the data array.
              Would you like to use the noisy data? ([y]/n)
              
              Enter 'q/quit' exit the program.
              '''))
        while userResponded == False:
            answer = input('Action: ')
            if answer == '' or answer == 'y' or answer == 'yes':
                print('Proceeding with noisy data...')
                # read in the noisy data array
                data = np.load('noisyData.npz')['noisyData']
                userResponded = True
            elif answer == 'n' or answer == 'no':
                print('Proceeding with noise-free data...')
                # read in the recorded data array
                data = np.load(str(datadir['recordedData']))
                userResponded = True
            elif answer == 'q' or answer == 'quit':
                sys.exit('Exiting program.\n')
            else:
                print('Invalid response. Please enter \'y/yes\', \'n\no\', or \'q/quit\'.')
                
    else:
        # read in the recorded data array
        data = np.load(str(datadir['recordedData']))
        
    # apply user-specified windows to data array
    rinterval, tinterval, tstep, dt, sinterval = get_user_windows(verbose)
    print('Applying windows to data volume...')
    data = data[rinterval, :, :]
    data = data[:, tinterval, :]
    data = data[:, :, sinterval]
    
    # Apply tapered cosine (Tukey) window to time signals.
    # This ensures that any fast fourier transforms (FFTs) used
    # will be acting on a function that is continuous at its edges.
    data = tukey_taper(data, tstep * dt, pulseFun.peakFreq)
    
    if domain == 'freq':
        print('Transforming data to the frequency domain...')
        data = fft_and_window(data, tstep * dt, double_length=True)
        
    return data

def load_test_funcs(domain, medium, verbose=False, return_sampling_points=False):
    # load the test functions
    rinterval, tinterval, tstep, dt = get_user_windows(verbose, skip_sources=True)
    if medium == 'variable':
        if 'testFuncs' in datadir:
            # load user-provided test functions
            print('Loading user-provided test functions...')
            testFuncs = np.load(str(datadir['testFuncs']))
            
            # Apply user-specified windows to data array
            print('Applying window to test functions...')
            testFuncs = testFuncs[rinterval, :, :]
            testFuncs = testFuncs[:, tinterval, :]
    
            # Apply tapered cosine (Tukey) window to time signals.
            # This ensures that any fast fourier transforms (FFTs) used
            # will be acting on a function that is continuous at its edges.
            testFuncs = tukey_taper(testFuncs, tstep * dt, pulseFun.peakFreq)
    
            if domain == 'freq':
                print('Transforming test functions to the frequency domain...')
                testFuncs = fft_and_window(testFuncs, tstep * dt, double_length=True)
            
            if return_sampling_points:
                samplingPoints = np.load(str(datadir['samplingPoints']))
                return testFuncs, samplingPoints
            
            else:
                return testFuncs
            
            
        else:
            sys.exit(textwrap.dedent(
                    '''
                    FileNotFoundError: The file containing user-provided test functions
                    has not been specified for the current data directory. If such a file
                    exists, run
                        
                        vzdata --path=<path/to/data
                        
                    and specify the file name when prompted.
                    '''))
            
    else:
        # medium is constant
        # use Vezda-computed test functions
        receiverPoints = np.load(str(datadir['receivers']))
        recordingTimes = np.load(str(datadir['recordingTimes']))
        
        # Apply user-specified windows to data array
        receiverPoints = receiverPoints[rinterval, :]
        recordingTimes = recordingTimes[tinterval]
        
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
            samplingPoints = np.vstack(np.meshgrid(x, y, z, indexing='ij')).reshape(3, Nx * Ny * Nz).T
        else:
            samplingPoints = np.vstack(np.meshgrid(x, y, indexing='ij')).reshape(2, Nx * Ny).T
            
        pulse = lambda t : pulseFun.pulse(t)
        velocity = pulseFun.velocity
        peakFreq = pulseFun.peakFreq
        peakTime = pulseFun.peakTime
            
        tu = plotParams['tu']
        # set up the convolution times based on length of recording time interval
        T = recordingTimes[-1] - recordingTimes[0]
        convolutionTimes = np.linspace(-T, T, 2 * len(recordingTimes) - 1)
        
        if Path('VZTestFuncs.npz').exists():
            print('Detected that free-space test functions have already been computed...')
            print('Checking consistency with current sampling grid, focusing time, and pulse function...')
            TFDict = np.load('VZTestFuncs.npz')
                
            if samplingIsCurrent(TFDict, receiverPoints, convolutionTimes, samplingPoints, tau, velocity, peakFreq, peakTime):
                testFuncs = TFDict['TFarray']
                print('Test functions are up to date...')
                    
            else:
                if tau != 0.0:
                    if tu != '':
                        print('Recomputing test functions for current sampling grid and focusing time %0.2f %s...' %(tau, tu))
                    else:
                        print('Recomputing test functions for current sampling grid and focusing time %0.2f...' %(tau))
                    testFuncs = sampleSpace(receiverPoints, convolutionTimes - tau, samplingPoints,
                                            velocity, pulse)
                else:
                    print('Recomputing test functions for current sampling grid...')
                    testFuncs = sampleSpace(receiverPoints, convolutionTimes, samplingPoints,
                                            velocity, pulse)
                    
                np.savez('VZTestFuncs.npz', TFarray=testFuncs, time=convolutionTimes, receivers=receiverPoints,
                         peakFreq=peakFreq, peakTime=peakTime, velocity=velocity,
                         samplingPoints=samplingPoints, tau=tau)
                    
        else:                
            if tau != 0.0:
                if tu != '':
                    print('Computing test functions for current sampling grid and focusing time %0.2f %s...' %(tau, tu))
                else:
                    print('Computing test functions for current sampling grid and focusing time %0.2f...' %(tau))
                testFuncs = sampleSpace(receiverPoints, convolutionTimes - tau, samplingPoints,
                                        velocity, pulse)
            else:
                print('Computing test functions for current sampling grid...')
                testFuncs = sampleSpace(receiverPoints, convolutionTimes, samplingPoints,
                                        velocity, pulse)
                    
            np.savez('VZTestFuncs.npz', TFarray=testFuncs, time=convolutionTimes, receivers=receiverPoints,
                     peakFreq=peakFreq, peakTime=peakTime, velocity=velocity,
                     samplingPoints=samplingPoints, tau=tau)
    
        if domain == 'freq':
            print('Transforming test functions to the frequency domain...')
            testFuncs = fft_and_window(testFuncs, tstep * dt, double_length=False)
    
        if return_sampling_points:
            return testFuncs, samplingPoints
        
        else:
            return testFuncs
        

def get_user_windows(verbose=False, skip_sources=False):
    recordingTimes = np.load(str(datadir['recordingTimes']))
    dt = recordingTimes[1] - recordingTimes[0]
    
    if Path('window.npz').exists():
        
        windowDict = np.load('window.npz')
        
        # Receiver window parameters
        rstart = windowDict['rstart']
        rstop = windowDict['rstop']
        rstep = windowDict['rstep']
        rinterval = np.arange(rstart, rstop, rstep)
        
        # Time window parameters (with units of time)
        tstart = windowDict['tstart']
        tstop = windowDict['tstop']
        tstep = windowDict['tstep']
        tu = plotParams['tu']
        
        if verbose:
            print('Detected user-specified windows:\n')
        
            # For display/printing purposes, count receivers with one-based
            # indexing. This amounts to incrementing the rstart parameter by 1
            print('window @ receivers : start =', rstart + 1)
            print('window @ receivers : stop =', rstop)
            print('window @ receivers : step =', rstep, '\n')
        
            if tu != '':
                print('window @ time : start = %0.2f %s' %(tstart, tu))
                print('window @ time : stop = %0.2f %s' %(tstop, tu))
            else:
                print('window @ time : start =', tstart)
                print('window @ time : stop =', tstop)
            print('window @ time : step =', tstep, '\n')
        
        # Convert time window parameters to corresponding array indices
        tstart = int(round(tstart / dt))
        tstop = int(round(tstop / dt))
        tinterval = np.arange(tstart, tstop, tstep)
        
        if skip_sources:
            return rinterval, tinterval, tstep, dt
        
        else:
            # Source window parameters
            slabel = windowDict['slabel']
            sstart = windowDict['sstart']
            sstop = windowDict['sstop']
            sstep = windowDict['sstep']
            sinterval = np.arange(sstart, sstop, sstep)
        
            if verbose:
                # For display/printing purposes, count recordings/sources with one-based
                # indexing. This amounts to incrementing the sstart parameter by 1
                print('window @ %s : start = %s' %(slabel, sstart + 1))
                print('window @ %s : stop = %s' %(slabel, sstop))
                print('window @ %s : step = %s\n' %(slabel, sstep))
            
            return rinterval, tinterval, tstep, dt, sinterval
        
    else:
        # Set default window parameters if user did
        # not specify window parameters.
        X = np.load(str(datadir['recordedData']))
        Nr, Nt, Ns = X.shape
        
        # Receiver window parameters
        rstart = 0
        rstop = Nr
        rstep = 1
        rinterval = np.arange(rstart, rstop, rstep)
        
        # Time window parameters (integers corresponding to array indices)
        tstart = 0
        tstop = Nt
        tstep = 1
        tinterval = np.arange(tstart, tstop, tstep)
        
        if skip_sources:
            return rinterval, tinterval, tstep, dt
            
        else:
            # Source window parameters
            sstart = 0
            sstop = Ns
            sstep = 1
            sinterval = np.arange(sstart, sstop, sstep)
            
            return rinterval, tinterval, tstep, dt, sinterval


#==============================================================================
def fft_and_window(X, dt, double_length):
    # Transform X into the frequency domain and apply window around nonzero
    # frequency components
    
    if double_length:
        N = nextPow2(2 * X.shape[1])
    else:
        N = nextPow2(X.shape[1])
    X = np.fft.rfft(X, n=N, axis=1)
    
    if plotParams['fmax'] is None:
        freqs = np.fft.rfftfreq(N, dt)
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
        
    df = 1.0 / (N * dt)
    startIndex = int(round(fmin / df))
    stopIndex = int(round(fmax / df))
        
    finterval = np.arange(startIndex, stopIndex, 1)
    X = X[:, finterval, :]
    
    return X
