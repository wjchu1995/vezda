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
import argparse
import textwrap
from pathlib import Path
from scipy.sparse.linalg import svds
from scipy.signal import tukey
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from vezda.math_utils import nextPow2
from vezda.sampling_utils import samplingIsCurrent, sampleSpace
from vezda.plot_utils import (vector_title, remove_keymap_conflicts, plotWiggles,
                              plotFreqVectors, process_key_vectors, default_params, setFigure)
from vezda.LinearOperators import asConvolutionalOperator
import numpy as np
import pickle
import time

sys.path.append(os.getcwd())
import pulseFun

#==============================================================================
def humanReadable(seconds):
    '''
    Convert elapsed time (in seconds) to human-readable 
    format (hours : minutes : seconds)
    '''
    
    h = int(seconds / 3600)
    m = int((seconds % 3600) / 60)
    s = seconds % 60.0
    
    return '{}h : {:>02}m : {:>05.2f}s'.format(h, m, s)

#==============================================================================
def isValid(numVals):
    validType = False
    while validType == False:
        if type(numVals) == int:
            validType = True
            
            if numVals >= 1:
                isValid = True
                break
            
            else:
                print(textwrap.dedent(
                     '''
                     ValueError: Argument '-k/--numVals' must be a positive integer 
                     between 1 and the order of the square input matrix.
                     '''))
                isValid = False
                break
       
        elif type(numVals) != int:
            print(textwrap.dedent(
                 '''
                 TypeError: Argument '-k/--numVals' must be a positive integer 
                 between 1 and the order of the input matrix.
                 '''))
            break
        
    return isValid

#==============================================================================
def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nfo', action='store_true',
                        help='''Compute or plot the singular-value decomposition of the
                        near-field operator (NFO).''')
    parser.add_argument('--lso', action='store_true',
                        help='''Compute or plot the singular-value decomposition of the
                        Lippmann-Schwinger operator (LSO).''')
    parser.add_argument('--numVals', '-k', type=int,
                        help='''Specify the number of singular values/vectors to compute.
                        Must a positive integer between 1 and the order of the square
                        input matrix.''')
    parser.add_argument('--domain', '-d', type=str, choices=['time', 'freq'],
                        help='''Specify whether to compute the singular-value decomposition in
                        the time domain or frequency domain. Default is set to frequency domain
                        for faster, more accurate performance.''')
    parser.add_argument('--plot', '-p', action='store_true',
                        help='''Plot the computed singular values and vectors.''')
    parser.add_argument('--format', '-f', type=str, default='pdf', choices=['png', 'pdf', 'ps', 'eps', 'svg'],
                        help='''Specify the image format of the saved file. Accepted formats are png, pdf,
                        ps, eps, and svg. Default format is set to pdf.''')
    parser.add_argument('--mode', type=str, choices=['light', 'dark'], required=False,
                        help='''Specify whether to view plots in light mode for daytime viewing
                        or dark mode for nighttime viewing.
                        Mode must be either \'light\' or \'dark\'.''')
    args = parser.parse_args()
    
    if args.nfo and not args.lso:
        operatorType = 'near-field operator'
        inputType = 'data'
        try:
            SVD = np.load('NFO_SVD.npz')
            s = SVD['s']
            Uh = SVD['Uh']
            V = SVD['V']
            domain = SVD['domain']
        
        except FileNotFoundError:
            s, Uh, V, domain = None, None, None, 'freq'
    
    elif not args.nfo and args.lso:
        operatorType = 'Lippmann-Schwinger operator'
        inputType = 'test functions'
        try:
            SVD = np.load('LSO_SVD.npz')
            s = SVD['s']
            Uh = SVD['Uh']
            V = SVD['V']
            domain = SVD['domain']
            
        except FileNotFoundError:
            s, Uh, V, domain = None, None, None, 'freq'
            
    elif args.nfo and args.lso:
        sys.exit(textwrap.dedent(
                '''
                UsageError: Please specify only one of the arguments \'--nfo\' or \'--lso\'.
                '''))
    
    else:
        sys.exit(textwrap.dedent(
                '''
                For which operator would you like to compute or plot a singular-value decomposition?
                Enter:
                    
                    vzsvd --nfo
                
                for the near-field operator or
                
                    vzsvd --lso
                    
                for the Lippmann-Schwinger operator.
                '''))
    
    #==============================================================================
    # if an SVD already exists...    
    if any(v is not None for v in [s, Uh, V]) and args.numVals is not None and args.plot is True:
        if args.numVals >= 1 and args.numVals == len(s):
            userResponded = False
            print(textwrap.dedent(
                 '''
                 A singular-value decomposition of the {s} for {n} values/vectors already exists. 
                 What would you like to do?
                 
                 Enter '1' to specify a new number of values/vectors to compute. (Default)
                 Enter '2' to recompute a singular-value decomposition for {n} values/vectors.
                 Enter 'q/quit' to exit.
                 '''.format(s=operatorType, n=args.numVals)))
            while userResponded == False:
                answer = input('Action: ')
                if answer == '' or answer == '1':
                    k = int(input('Please specify the number of singular values/vectors to compute: '))
                    if isValid(k):
                        print('Proceeding with numVals = %s...' %(k))
                        userResponded = True
                        computeSVD = True
                        break
                    else:
                        break
                elif answer == '2':
                    k = args.numVals
                    print('Recomputing SVD of the %s for %s singular values/vectors...' %(operatorType, k))
                    userResponded = True
                    computeSVD = True
                elif answer == 'q' or answer == 'quit':
                    sys.exit('Exiting program.\n')
                else:
                    print('Invalid response. Please enter \'1\', \'2\', or \'q/quit\'.')
        
        elif args.numVals >= 1 and args.numVals != len(s):
            k = args.numVals
            computeSVD = True
                
        elif args.numVals < 1:
            userResponded = False
            print(textwrap.dedent(
                 '''
                 ValueError: Argument '-k/--numVals' must be a positive integer 
                 between 1 and the order of the square input matrix. The parameter will
                 be set to the default value of 6.
                 What would you like to do?
                 
                 Enter '1' to specify a value of the parameter. (Default)
                 Enter '2' to proceed with the default value.
                 Enter 'q/quit' exit the program.
                 '''))
            while userResponded == False:
                answer = input('Action: ')
                if answer == '' or answer == '1':
                    k = int(input('Please specify the number of singular values/vectors to compute: '))
                    if isValid(k):
                        print('Proceeding with numVals = %s...' %(k))
                        userResponded = True
                        computeSVD = True
                        break
                    else:
                        break
                elif answer == '2':
                    k = 6
                    print('Proceeding with the default value numVals = %s...' %(k))
                    computeSVD = True
                    userResponded = True
                    break
                elif answer == 'q' or answer == 'quit':
                    sys.exit('Exiting program.\n')
                else:
                    print('Invalid response. Please enter \'1\', \'2\', or \'q/quit\'.')
    
    elif all(v is not None for v in [s, Uh, V]) and args.numVals is None and args.plot is True:
        computeSVD = False
        
    elif all(v is not None for v in [s, Uh, V]) and args.numVals is not None and args.plot is False:
        if args.numVals >= 1 and args.numVals == len(s):
            userResponded = False
            print(textwrap.dedent(
                 '''
                 A singular-value decomposition of the {s} for {n} values/vectors already exists. 
                 What would you like to do?
                 
                 Enter '1' to specify a new number of values/vectors to compute. (Default)
                 Enter '2' to recompute a singular-value decomposition for {n} values/vectors.
                 Enter 'q/quit' to exit.
                 '''.format(s=operatorType, n=args.numVals)))
            while userResponded == False:
                answer = input('Action: ')
                if answer == '' or answer == '1':
                    k = int(input('Please specify the number of singular values/vectors to compute: '))
                    if isValid(k):
                        print('Proceeding with numVals = %s...' %(k))
                        userResponded = True
                        computeSVD = True
                        break
                    else:
                        break
                elif answer == '2':
                    k = args.numVals
                    print('Recomputing SVD of the %s for %s singular values/vectors...' %(operatorType, k))
                    userResponded = True
                    computeSVD = True
                elif answer == 'q' or answer == 'quit':
                    sys.exit('Exiting program.\n')
                else:
                    print('Invalid response. Please enter \'1\', \'2\', or \'q/quit\'.')
        
        elif args.numVals >= 1 and args.numVals != len(s):
            k = args.numVals
            computeSVD = True
                
        elif args.numVals < 1:
            userResponded = False
            print(textwrap.dedent(
                 '''
                 ValueError: Argument '-k/--numVals' must be a positive integer 
                 between 1 and the order of the square input matrix. The parameter will
                 be set to the default value of 6.
                 What would you like to do?
                 
                 Enter '1' to specify a value of the parameter. (Default)
                 Enter '2' to proceed with the default value.
                 Enter 'q/quit' exit the program.
                 '''))
            while userResponded == False:
                answer = input('Action: ')
                if answer == '' or answer == '1':
                    k = int(input('Please specify the number of singular values/vectors to compute: '))
                    if isValid(k):
                        print('Proceeding with numVals = %s...' %(k))
                        userResponded = True
                        computeSVD = True
                        break
                    else:
                        break
                elif answer == '2':
                    k = 6
                    print('Proceeding with the default value numVals = %s...' %(k))
                    computeSVD = True
                    userResponded = True
                    break
                elif answer == 'q' or answer == 'quit':
                    sys.exit('Exiting program.\n')
                else:
                    print('Invalid response. Please enter \'1\', \'2\', or \'q/quit\'.')
                
    elif all(v is not None for v in [s, Uh, V]) and args.numVals is None and args.plot is False:
        sys.exit(textwrap.dedent(
                '''
                No action specified. A singular-value decomposition of the %s
                for %s values/vectors already exists. Please specify at least one of '-k/--numVals'
                or '-p/--plot' arguments with 'vzsvd' command.
                ''' %(operatorType, len(s))))
    #==============================================================================
    # if an SVD does not already exist...
    elif any(v is None for v in [s, Uh, V]) and args.numVals is not None and args.plot is True:
        if args.numVals >= 1:
            computeSVD = True
            k = args.numVals
                
        elif args.numVals < 1:
            userResponded = False
            print(textwrap.dedent(
                 '''
                 ValueError: Argument '-k/--numVals' must be a positive integer 
                 between 1 and the order of the square input matrix. The parameter will
                 be set to the default value of 6.
                 What would you like to do?
                 
                 Enter '1' to specify a value of the parameter. (Default)
                 Enter '2' to proceed with the default value.
                 Enter 'q/quit' exit the program.
                 '''))
            while userResponded == False:
                answer = input('Action: ')
                if answer == '' or answer == '1':
                    k = int(input('Please specify the number of singular values/vectors to compute: '))
                    if isValid(k):
                        print('Proceeding with numVals = %s...' %(k))
                        userResponded = True
                        computeSVD = True
                        break
                    else:
                        break
                elif answer == '2':
                    k = 6
                    print('Proceeding with the default value numVals = %s...' %(k))
                    computeSVD = True
                    userResponded = True
                    break
                elif answer == 'q' or answer == 'quit':
                    sys.exit('Exiting program.\n')
                else:
                    print('Invalid response. Please enter \'1\', \'2\', or \'q/quit\'.')
    
    elif any(v is None for v in [s, Uh, V]) and args.numVals is None and args.plot is True:
        userResponded = False
        print(textwrap.dedent(
             '''
             PlotError: A singular-value decomposition of the {s} does not exist. A plot will be
             generated after a singular-value decomposition has been computed.
             
             Enter '1' to specify a number of singular values/vectors to compute. (Default)
             Enter 'q/quit' to exit.
             '''.format(s=operatorType)))
        while userResponded == False:
            answer = input('Action: ')
            if answer == '' or answer == '1':
                k = int(input('Please specify the number of singular values/vectors to compute: '))
                if isValid(k):
                    print('Proceeding with numVals = %s...' %(k))
                    userResponded = True
                    computeSVD = True
                    break
                else:
                    break
            elif answer == 'q' or answer == 'quit':
                sys.exit('Exiting program.\n')
            else:
                print('Invalid response. Please enter \'1\', or \'q/quit\'.')
        
    elif any(v is None for v in [s, Uh, V]) and args.numVals is not None and args.plot is False:
        if args.numVals >= 1:
            k = args.numVals
            computeSVD = True
                
        elif args.numVals < 1:
            userResponded = False
            print(textwrap.dedent(
                 '''
                 ValueError: Argument '-k/--numVals' must be a positive integer 
                 between 1 and the order of the square input matrix. The parameter will
                 be set to the default value of 6.
                 What would you like to do?
                 
                 Enter '1' to specify a value of the parameter. (Default)
                 Enter '2' to proceed with the default value.
                 Enter 'q/quit' exit the program.
                 '''))
            while userResponded == False:
                answer = input('Action: ')
                if answer == '' or answer == '1':
                    k = int(input('Please specify the number of singular values/vectors to compute: '))
                    if isValid(k):
                        print('Proceeding with numVals = %s...' %(k))
                        userResponded = True
                        computeSVD = True
                        break
                    else:
                        break
                elif answer == '2':
                    k = 6
                    print('Proceeding with the default value numVals = %s...' %(k))
                    computeSVD = True
                    userResponded = True
                    break
                elif answer == 'q' or answer == 'quit':
                    sys.exit('Exiting program.\n')
                else:
                    print('Invalid response. Please enter \'1\', \'2\', or \'q/quit\'.')
                
    elif any(v is None for v in [s, Uh, V]) and args.numVals is None and args.plot is False:
        sys.exit(textwrap.dedent(
                '''
                Nothing to be done. A singular-value decomposition of the {s} does not exist.
                Please specify at least one of '-k/--numVals' or '-p/--plot'
                arguments with 'vzsvd' command.
                '''.format(s=operatorType)))  
    #==============================================================================
    # Read in data files 
    datadir = np.load('datadir.npz')
    receiverPoints = np.load(str(datadir['receivers']))
    recordingTimes = np.load(str(datadir['recordingTimes']))
    dt = recordingTimes[1] - recordingTimes[0]
    
    if Path('window.npz').exists():
        windowDict = np.load('window.npz')
        
        # Apply the receiver window
        rstart = windowDict['rstart']
        rstop = windowDict['rstop']
        rstep = windowDict['rstep']
        
        # Apply the time window
        tstart = windowDict['tstart']
        tstop = windowDict['tstop']
        tstep = windowDict['tstep']
        
        # Convert time window parameters to corresponding array indices
        Tstart = int(round(tstart / dt))
        Tstop = int(round(tstop / dt))
    
    else:
        rstart = 0
        rstop = receiverPoints.shape[0]
        rstep = 1
        
        tstart = recordingTimes[0]
        tstop = recordingTimes[-1]
        
        Tstart = 0
        Tstop = len(recordingTimes)
        tstep = 1
                
    # Apply the receiver window
    rinterval = np.arange(rstart, rstop, rstep)
    receiverPoints = receiverPoints[rinterval, :]

    # Apply the time window
    tinterval = np.arange(Tstart, Tstop, tstep)
    recordingTimes = recordingTimes[tinterval]
    
    # Used for getting time and frequency units
    if Path('plotParams.pkl').exists():
        plotParams = pickle.load(open('plotParams.pkl', 'rb'))
    else:
        plotParams = default_params()
        
    if computeSVD:
        # get time units for printing time windows or time shifts
        tu = plotParams['tu']
        
        if args.nfo:
            
            if Path('noisyData.npz').exists():
                userResponded = False
                print(textwrap.dedent(
                      '''
                      Detected that band-limited noise has been added to the data array.
                      Would you like to compute an SVD of the noisy data? ([y]/n)
                      
                      Enter 'q/quit' exit the program.
                      '''))
                while userResponded == False:
                    answer = input('Action: ')
                    if answer == '' or answer == 'y' or answer == 'yes':
                        print('Proceeding with singular-value decomposition using noisy data...')
                        # read in the noisy data array
                        X = np.load('noisyData.npz')['noisyData']
                        userResponded = True
                    elif answer == 'n' or answer == 'no':
                        print('Proceeding with singular-value decomposition using noise-free data...')
                        # read in the recorded data array
                        X  = np.load(str(datadir['recordedData']))
                        userResponded = True
                    elif answer == 'q' or answer == 'quit':
                        sys.exit('Exiting program.\n')
                    else:
                        print('Invalid response. Please enter \'y/yes\', \'n\no\', or \'q/quit\'.')
                
            else:
                # read in the recorded data array
                X  = np.load(str(datadir['recordedData']))
                
            if Path('window.npz').exists():
                print('Detected user-specified window:\n')
                
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
                
                # Apply the source window
                slabel = windowDict['slabel']
                sstart = windowDict['sstart']
                sstop = windowDict['sstop']
                sstep = windowDict['sstep']
                sinterval = np.arange(sstart, sstop, sstep)
                
                # For display/printing purposes, count recordings/sources with one-based
                # indexing. This amounts to incrementing the sstart parameter by 1
                print('window @ %s : start = %s' %(slabel, sstart + 1))
                print('window @ %s : stop = %s' %(slabel, sstop))
                print('window @ %s : step = %s\n' %(slabel, sstep))                
                
                print('Applying window to data volume...')
                X = X[rinterval, :, :]
                X = X[:, tinterval, :]
                X = X[:, :, sinterval]
                Nr, Nt, Ns = X.shape
                
                # Apply tapered cosine (Tukey) window to time signals.
                # This ensures the fast fourier transform (FFT) used in
                # the definition of the matrix-vector product below is
                # acting on a function that is continuous at its edges.
                
                peakFreq = pulseFun.peakFreq
                # Np : Number of samples in the dominant period T = 1 / peakFreq
                Np = int(round(1 / (tstep * dt * peakFreq)))
                # alpha is set to taper over 6 of the dominant period of the
                # pulse function (3 periods from each end of the signal)
                alpha = 6 * Np / Nt
                print('Tapering time signals with Tukey window: %d'
                      %(int(round(alpha * 100))) + '%')
                TukeyWindow = tukey(Nt, alpha)
                X *= TukeyWindow[None, :, None]
                
            else:
                Nr, Nt, Ns = X.shape
        
        elif args.lso:
            
            if Path('samplingGrid.npz').exists():
                samplingGrid = np.load('samplingGrid.npz')
                x = samplingGrid['x']
                y = samplingGrid['y']
                tau = samplingGrid['tau']
                if 'z' in samplingGrid:
                    z = samplingGrid['z']
                else:
                    z = None
                    
            else:
                sys.exit(textwrap.dedent(
                        '''
                        A sampling grid needs to be set up before computing a
                        singular-value decomposition of the %s.
                        Enter:
                            
                            vzgrid --help
                            
                        from the command-line for more information on how to set up a
                        sampling grid.
                        ''' %(operatorType)))
            
            pulse = lambda t : pulseFun.pulse(t)
            velocity = pulseFun.velocity
            peakFreq = pulseFun.peakFreq
            peakTime = pulseFun.peakTime
            
            if Path('VZTestFuncs.npz').exists():
                print('\nDetected that free-space test functions have already been computed...')
                print('Checking consistency with current space-time sampling grid...')
                TFDict = np.load('VZTestFuncs.npz')
                
                if samplingIsCurrent(TFDict, receiverPoints, recordingTimes, velocity, tau, x, y, z, peakFreq, peakTime):
                    X = TFDict['TFarray']
                    sourcePoints = TFDict['samplingPoints']
                    print('Moving forward to SVD...')
                    
                else:
                    print('Recomputing test functions...')
                    # set up the convolution times based on length of recording time interval
                    T = recordingTimes[-1] - recordingTimes[0]
                    convolutionTimes = np.linspace(-T, T, 2 * len(recordingTimes) - 1)
                    
                    if tau[0] != 0:
                        if tu != '':
                            print('Recomputing test functions for focusing time %0.2f %s...' %(tau[0], tu))
                        else:
                            print('Recomputing test functions for focusing time %0.2f...' %(tau[0]))
                        X, sourcePoints = sampleSpace(receiverPoints, convolutionTimes - tau[0], velocity,
                                                      x, y, z, pulse)
                    else:
                        X, sourcePoints = sampleSpace(receiverPoints, convolutionTimes, velocity,
                                                      x, y, z, pulse)
                    
                    
                    if z is None:
                        np.savez('VZTestFuncs.npz', TFarray=X, time=recordingTimes, receivers=receiverPoints,
                                 peakFreq=peakFreq, peakTime=peakTime, velocity=velocity,
                                 x=x, y=y, tau=tau, samplingPoints=sourcePoints)
                    else:
                        np.savez('VZTestFuncs.npz', TFarray=X, time=recordingTimes, receivers=receiverPoints,
                                 peakFreq=peakFreq, peakTime=peakTime, velocity=velocity,
                                 x=x, y=y, z=z, tau=tau, samplingPoints=sourcePoints)
                    
            else:                
                print('\nComputing free-space test functions for the current space-time sampling grid...')
                if tau[0] != 0:
                    if tu != '':
                        print('Computing test functions for focusing time %0.2f %s...' %(tau[0], tu))
                    else:
                        print('Computing test functions for focusing time %0.2f...' %(tau[0]))
                    X, sourcePoints = sampleSpace(receiverPoints, recordingTimes - tau[0], velocity,
                                                  x, y, z, pulse)
                else:
                    X, sourcePoints = sampleSpace(receiverPoints, recordingTimes, velocity,
                                                  x, y, z, pulse)
                    
                if z is None:
                    np.savez('VZTestFuncs.npz', TFarray=X, time=recordingTimes, receivers=receiverPoints,
                             peakFreq=peakFreq, peakTime=peakTime, velocity=velocity,
                             x=x, y=y, tau=tau, samplingPoints=sourcePoints)
                else:
                    np.savez('VZTestFuncs.npz', TFarray=X, time=recordingTimes, receivers=receiverPoints,
                             peakFreq=peakFreq, peakTime=peakTime, velocity=velocity,
                             x=x, y=y, z=z, tau=tau, samplingPoints=sourcePoints)
                    
            Nr, Nt, Ns = X.shape
            
        #==============================================================================
        if args.domain is not None:
            domain = args.domain
        
        if domain == 'freq':
            # Transform convolutional operator into frequency domain and bandpass for efficient SVD
            print('Transforming %s to the frequency domain...' %(inputType))
            N = nextPow2(2 * Nt)
            X = np.fft.rfft(X, n=N, axis=1)
                
            if plotParams['fmax'] is None:
                freqs = np.fft.rfftfreq(N, tstep * dt)
                plotParams['fmax'] = np.max(freqs)
        
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
            X = X[:, finterval, :]
            
        #==============================================================================
        # Compute the k largest singular values (which='LM') of the operator A
        # Singular values are elements of the vector 's'
        # Left singular vectors are columns of 'U'
        # Right singular vectors are columns of 'V'
        
        A = asConvolutionalOperator(X)
        
        if k == 1:
            print('Computing SVD of the %s for 1 singular value/vector...' %(operatorType))
        else:
            print('Computing SVD of the %s for %s singular values/vectors...' %(operatorType, k))
        startTime = time.time()
        U, s, Vh = svds(A, k, which='LM')
        endTime = time.time()
        print('Elapsed time:', humanReadable(endTime - startTime), '\n')
        
        # sort the singular values and corresponding vectors in descending order
        # (i.e., largest to smallest)
        index = s.argsort()[::-1]   
        s = s[index]
        Uh = U[:, index].conj().T
        V = Vh[index, :].conj().T
        
        # Write binary output with numpy
        if args.nfo:
            np.savez('NFO_SVD.npz', s=s, Uh=Uh, V=V, domain=domain)        
        elif args.lso:
            np.savez('LSO_SVD.npz', s=s, Uh=Uh, V=V, domain=domain)
    
    #==============================================================================    
    if args.plot and all(v is not None for v in [s, Uh, V]):
        
        Nr = receiverPoints.shape[0]
        Nt = len(recordingTimes)
        
        try:
            k
        except NameError:
            k = len(s)
            
        if args.domain is not None and domain != args.domain:
            if domain == 'freq':
                s1 = 'time'
                s2 = 'frequency'
            else:
                s1 = 'frequency'
                s2 = 'time'
            sys.exit(textwrap.dedent(
                    '''
                    Error: Attempted to plot the singular-value decomposition in the %s
                    domain, but the decomposition was computed in the %s domain.
                    ''' %(s1, s2)))
                
        if domain == 'freq':
            # plot singular vectors in frequency domain 
            N = nextPow2(2 * Nt)
            freqs = np.fft.rfftfreq(N, tstep * dt)
            
            if plotParams['fmax'] is None:
                plotParams['fmax'] = np.max(freqs)
            
            # Apply the frequency window
            fmin = plotParams['fmin']
            fmax = plotParams['fmax']
            df = 1.0 / (N * tstep * dt)
            
            startIndex = int(round(fmin / df))
            stopIndex = int(round(fmax / df))
            finterval = np.arange(startIndex, stopIndex, 1)
            freqs = freqs[finterval]
            fmax = freqs[-1]
        
            M = len(freqs)         
            Ns = int(V.shape[0] / M)
            U = np.reshape(Uh.conj().T, (Nr, M, k))
            V = np.reshape(V, (Ns, M, k))
            
        else: # domain == 'time'
            M = 2 * Nt - 1
            Ns = int(V.shape[0] / M)
            U = np.reshape(Uh.T, (Nr, M, k))
            V = np.reshape(V, (Ns, M, k))
            T = recordingTimes[-1] - recordingTimes[0]
            times = np.linspace(-T, T, M)
        
        if args.nfo:    # Near-field operator
            try:
                sinterval
            except NameError:
                if Path('window.npz').exists():
                    sstart = windowDict['sstart']
                    sstop = windowDict['sstop']
                    sstep = windowDict['sstep']    
                else:
                    sstart = 0
                    sstop = Ns
                    sstep = 1
                
                sinterval = np.arange(sstart, sstop, sstep)
                
            if 'sources' in datadir:
                sourcePoints = np.load(str(datadir['sources']))
                sourcePoints = sourcePoints[sinterval, :]
            else:
                sourcePoints = None
            
        else:
            # if args.lso (Lippmann-Schwinger operator)
            
            # in the case of the Lippmann-Schwinger operator, 'sourcePoints'
            # correspond to sampling points, which should always exist.
            try:
                sourcePoints
            except NameError:
                if Path('VZTestFuncs.npz').exists():
                    TFDict = np.load('VZTestFuncs.npz')
                    sourcePoints = TFDict['samplingPoints']
                else:
                    sys.exit(textwrap.dedent(
                            '''
                            Error: A sampling grid must exist and test functions computed
                            before a singular-value decomposition of the Lippmann-Schwinger
                            operator can be computed or plotted.
                            '''))
                        
            sstart = 0
            sstop = sourcePoints.shape[0]
            sstep = 1
            sinterval = np.arange(sstart, sstop, sstep)
            
        # increment source/recording interval and receiver interval to be consistent
        # with one-based indexing (i.e., count from one instead of zero)
        sinterval += 1
        rinterval += 1
        rstart += 1
        sstart += 1
        
        if args.mode is not None:
            plotParams['view_mode'] = args.mode
        
        pickle.dump(plotParams, open('plotParams.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)
        
        remove_keymap_conflicts({'left', 'right', 'up', 'down', 'save'})
        if domain == 'freq':
            
            # plot the left singular vectors
            fig_lvec, ax_lvec_r, ax_lvec_i = setFigure(num_axes=2, mode=plotParams['view_mode'])
            ax_lvec_r.volume = U.real
            ax_lvec_i.volume = U.imag
            ax_lvec_r.index = 0
            ax_lvec_i.index = 0
            fig_lvec.suptitle('Left-Singular Vector', color=ax_lvec_r.titlecolor, fontsize=16)
            fig_lvec.subplots_adjust(bottom=0.27, top=0.86)
            leftTitle_r = vector_title('left', ax_lvec_r.index + 1, 'real')
            leftTitle_i = vector_title('left', ax_lvec_i.index + 1, 'imag')
            for ax, title in zip([ax_lvec_r, ax_lvec_i], [leftTitle_r, leftTitle_i]):
                left_im = plotFreqVectors(ax, ax.volume[:, :, ax.index], freqs, fmin, fmax, rstart, rinterval,
                                          receiverPoints, title, 'left', plotParams)
                
            lp0 = ax_lvec_r.get_position().get_points().flatten()
            lp1 = ax_lvec_i.get_position().get_points().flatten()
            left_cax = fig_lvec.add_axes([lp0[0], 0.12, lp1[2]-lp0[0], 0.03])
            lcbar = fig_lvec.colorbar(left_im, left_cax, orientation='horizontal')
            lcbar.outline.set_edgecolor(ax_lvec_r.cbaredgecolor)
            lcbar.ax.tick_params(axis='x', colors=ax_lvec_r.labelcolor)              
            lcbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            lcbar.set_label('Amplitude',
                           labelpad=5, rotation=0, fontsize=12, color=ax_lvec_r.labelcolor)
            fig_lvec.canvas.mpl_connect('key_press_event', lambda event: process_key_vectors(event, freqs, fmin, fmax, rstart, sstart, 
                                                                                    rinterval, sinterval, receiverPoints, 
                                                                                    sourcePoints, plotParams, 'cmplx_left'))
            
            # plot the right singular vectors
            fig_rvec, ax_rvec_r, ax_rvec_i = setFigure(num_axes=2, mode=plotParams['view_mode'])
            ax_rvec_r.volume = V.real
            ax_rvec_i.volume = V.imag
            ax_rvec_r.index = 0
            ax_rvec_i.index = 0
            fig_rvec.suptitle('Right-Singular Vector', color=ax_rvec_r.titlecolor, fontsize=16)
            fig_rvec.subplots_adjust(bottom=0.27, top=0.86)
            rightTitle_r = vector_title('right', ax_rvec_r.index + 1, 'real')
            rightTitle_i = vector_title('right', ax_rvec_i.index + 1, 'imag')
            for ax, title in zip([ax_rvec_r, ax_rvec_i], [rightTitle_r, rightTitle_i]):
                right_im = plotFreqVectors(ax, ax.volume[:, :, ax.index], freqs, fmin, fmax, sstart, sinterval,
                                           sourcePoints, title, 'right', plotParams)
                
            rp0 = ax_rvec_r.get_position().get_points().flatten()
            rp1 = ax_rvec_i.get_position().get_points().flatten()
            right_cax = fig_rvec.add_axes([rp0[0], 0.12, rp1[2]-rp0[0], 0.03])
            rcbar = fig_rvec.colorbar(right_im, right_cax, orientation='horizontal')  
            rcbar.outline.set_edgecolor(ax_rvec_r.cbaredgecolor)
            rcbar.ax.tick_params(axis='x', colors=ax_rvec_r.labelcolor)
            rcbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            rcbar.set_label('Amplitude',
                           labelpad=5, rotation=0, fontsize=12, color=ax_lvec_r.labelcolor)
            fig_rvec.canvas.mpl_connect('key_press_event', lambda event: process_key_vectors(event, freqs, fmin, fmax, rstart, sstart, 
                                                                                    rinterval, sinterval, receiverPoints, 
                                                                                    sourcePoints, plotParams, 'cmplx_right'))
            
        else:
            # domain == 'time'   
            fig_vec, ax_lvec, ax_rvec = setFigure(num_axes=2, mode=plotParams['view_mode'])
            
            ax_lvec.volume = U
            ax_lvec.index = 0
            leftTitle = vector_title('left', ax_lvec.index + 1)
            plotWiggles(ax_lvec, ax_lvec.volume[:, :, ax_lvec.index], times, -T, T, rstart, rinterval,
                        receiverPoints, leftTitle, 'left', plotParams)
      
            ax_rvec.volume = V
            ax_rvec.index = 0
            rightTitle = vector_title('right', ax_rvec.index + 1)
            plotWiggles(ax_rvec, ax_rvec.volume[:, :, ax_rvec.index], times, -T, T, sstart, sinterval,
                        sourcePoints, rightTitle, 'right', plotParams)
            fig_vec.tight_layout()
            fig_vec.canvas.mpl_connect('key_press_event', lambda event: process_key_vectors(event, times, -T, T, rstart, sstart, 
                                                                                    rinterval, sinterval, receiverPoints, 
                                                                                    sourcePoints, plotParams))
        #==============================================================================
        # plot the singular values
        # figure and axis for singular values
        fig_vals, ax_vals = setFigure(num_axes=1, mode=plotParams['view_mode'])
        
        n = np.arange(1, k + 1, 1)
        kappa = s[0] / s[-1]    # condition number = max(s) / min(s)
        ax_vals.plot(n, s, '.', clip_on=False, markersize=9, label=r'Condition Number: %0.1e' %(kappa), color=ax_vals.pointcolor)
        ax_vals.set_xlabel('n', color=ax_vals.labelcolor)
        ax_vals.set_ylabel('$\sigma_n$', color=ax_vals.labelcolor)
        legend = ax_vals.legend(title='Singular Values', loc='upper center', bbox_to_anchor=(0.5, 1.25),
                           markerscale=0, handlelength=0, handletextpad=0, fancybox=True, shadow=True,
                           fontsize='large')
        legend.get_title().set_fontsize('large')
        ax_vals.set_xlim([1, k])
        ax_vals.set_ylim(bottom=0)
        ax_vals.locator_params(axis='y', nticks=6)
        ax_vals.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        fig_vals.tight_layout()
        fig_vals.savefig('singularValues.' + args.format, format=args.format, bbox_inches='tight', facecolor=fig_vals.get_facecolor())
        
        plt.show()
