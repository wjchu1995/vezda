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
from pathlib import Path
from scipy.sparse.linalg import LinearOperator, eigsh
from scipy.fftpack import fft, ifft
from scipy.signal import tukey
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np
import time

sys.path.append(os.getcwd())
import pulseFun

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
                print('''
                      ValueError: Argument '-k/--numVals' must be a positive integer 
                      between 1 and the order of the square input matrix.
                      ''')
                isValid = False
                break
        elif type(numVals) != int:
            print('''
                  TypeError: Argument '-k/--numVals' must be a positive integer 
                  between 1 and the order of the square input matrix.
                  ''')
            break
        
    return isValid

def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--numVals', '-k', type=int,
                        help='''specify the number of singular values/vectors to compute.
                        Must a positive integer between 1 and the order of the square
                        input matrix.''')
    parser.add_argument('--plot', '-p', action='store_true',
                        help='''Plot the computed singular values.''')
    parser.add_argument('--format', '-f', type=str, default='pdf', choices=['png', 'pdf', 'ps', 'eps', 'svg'],
                        help='''specify the image format of the saved file. Accepted formats are png, pdf,
                        ps, eps, and svg. Default format is set to pdf.''')
    args = parser.parse_args()
    
    try:
        s = np.load('singularValues.npy')
    except FileNotFoundError:
        s = None
    
    try:
        U = np.load('leftVectors.npy')
    except FileNotFoundError:
        U = None
    
    try:
        V = np.load('rightVectors.npy')
    except FileNotFoundError:
        V = None
    
    #==============================================================================
    # if an SVD already exists...    
    if all(v is not None for v in [s, U, V]) and args.numVals is not None and args.plot is True:
        if args.numVals >= 1 and args.numVals == len(s):
            userResponded = False
            print('''
                  A singular-value decomposition for {n} values/vectors already exists. 
                  What would you like to do?
                  Enter '1' to specify a new number of values/vectors to compute. (Default)
                  Enter '2' to recompute a singular-value decomposition for {n} values/vectors.
                  Enter 'q/quit' to exit.
                  '''.format(n=args.numVals))
            while userResponded == False:
                answer = input('Action: ')
                if answer == '' or answer == '1':
                    k = int(input('Please specify the number of singular values/vectors to compute: numVals = '))
                    if isValid(k):
                        print('Proceeding with numVals = %s...' %(k))
                        userResponded = True
                        computeSVD = True
                        break
                    else:
                        break
                elif answer == '2':
                    k = args.numVals
                    print('Recomputing SVD for %s singular values/vectors...' %(k))
                    userResponded = True
                    computeSVD = True
                elif answer == 'q' or answer == 'quit':
                    sys.exit('Exiting program.')
                else:
                    print('Invalid response. Please enter \'1\', \'2\', or \'q/quit\'.')
        
        elif args.numVals >= 1 and args.numVals != len(s):
            k = args.numVals
            computeSVD = True
                
        elif args.numVals < 1:
            userResponded = False
            print('''
                  ValueError: Argument '-k/--numVals' must be a positive integer 
                  between 1 and the order of the square input matrix. The parameter will
                  be set to the default value of 6.
                  What would you like to do?
                  Enter '1' to specify a value of the parameter. (Default)
                  Enter '2' to proceed with the default value.
                  Enter 'q/quit' exit the program.
                  ''')
            while userResponded == False:
                answer = input('Action: ')
                if answer == '' or answer == '1':
                    k = int(input('Please specify the number of singular values/vectors to compute: numVals = '))
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
                    sys.exit('Exiting program.')
                else:
                    print('Invalid response. Please enter \'1\', \'2\', or \'q/quit\'.')
    
    elif all(v is not None for v in [s, U, V]) and args.numVals is None and args.plot is True:
        computeSVD = False
        
    elif all(v is not None for v in [s, U, V]) and args.numVals is not None and args.plot is False:
        if args.numVals >= 1 and args.numVals == len(s):
            userResponded = False
            print('''
                  A singular-value decomposition for {n} values/vectors already exists. 
                  What would you like to do?
                  Enter '1' to specify a new number of values/vectors to compute. (Default)
                  Enter '2' to recompute a singular-value decomposition for {n} values/vectors.
                  Enter 'q/quit' to exit.
                  '''.format(n=args.numVals))
            while userResponded == False:
                answer = input('Action: ')
                if answer == '' or answer == '1':
                    k = int(input('Please specify the number of singular values/vectors to compute: numVals = '))
                    if isValid(k):
                        print('Proceeding with numVals = %s...' %(k))
                        userResponded = True
                        computeSVD = True
                        break
                    else:
                        break
                elif answer == '2':
                    k = args.numVals
                    print('Recomputing SVD for %s singular values/vectors...' %(k))
                    userResponded = True
                    computeSVD = True
                elif answer == 'q' or answer == 'quit':
                    sys.exit('Exiting program.')
                else:
                    print('Invalid response. Please enter \'1\', \'2\', or \'q/quit\'.')
        
        elif args.numVals >= 1 and args.numVals != len(s):
            k = args.numVals
            computeSVD = True
                
        elif args.numVals < 1:
            userResponded = False
            print('''
                  ValueError: Argument '-k/--numVals' must be a positive integer 
                  between 1 and the order of the square input matrix. The parameter will
                  be set to the default value of 6.
                  What would you like to do?
                  Enter '1' to specify a value of the parameter. (Default)
                  Enter '2' to proceed with the default value.
                  Enter 'q/quit' exit the program.
                  ''')
            while userResponded == False:
                answer = input('Action: ')
                if answer == '' or answer == '1':
                    k = int(input('Please specify the number of singular values/vectors to compute: numVals = '))
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
                    sys.exit('Exiting program.')
                else:
                    print('Invalid response. Please enter \'1\', \'2\', or \'q/quit\'.')
                
    elif all(v is not None for v in [s, U, V]) and args.numVals is None and args.plot is False:
        sys.exit('''
                 No action specified. A singular-value decomposition for %s values/vectors
                 already exists. Please specify at least one of '-k/--numVals' or '-p/--plot'
                 arguments with 'vzsvd' command.
                 ''' %(args.numVals))
    #==============================================================================
    # if an SVD does not already exist...
    elif any(v is None for v in [s, U, V]) and args.numVals is not None and args.plot is True:
        if args.numVals >= 1:
            computeSVD = True
            k = args.numVals
                
        elif args.numVals < 1:
            userResponded = False
            print('''
                  ValueError: Argument '-k/--numVals' must be a positive integer 
                  between 1 and the order of the square input matrix. The parameter will
                  be set to the default value of 6.
                  What would you like to do?
                  Enter '1' to specify a value of the parameter. (Default)
                  Enter '2' to proceed with the default value.
                  Enter 'q/quit' exit the program.
                  ''')
            while userResponded == False:
                answer = input('Action: ')
                if answer == '' or answer == '1':
                    k = int(input('Please specify the number of singular values/vectors to compute: numVals = '))
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
                    sys.exit('Exiting program.')
                else:
                    print('Invalid response. Please enter \'1\', \'2\', or \'q/quit\'.')
    
    elif any(v is None for v in [s, U, V]) and args.numVals is None and args.plot is True:
        userResponded = False
        print('''
              PlotError: A singular-value decomposition does not exist. A plot will be
              generated after a singular-value decomposition has been computed.
              Enter '1' to specify a number of singular values/vectors to compute. (Default)
              Enter 'q/quit' to exit.
              ''')
        while userResponded == False:
            answer = input('Action: ')
            if answer == '' or answer == '1':
                k = int(input('Please specify the number of singular values/vectors to compute: numVals = '))
                if isValid(k):
                    print('Proceeding with numVals = %s...' %(k))
                    userResponded = True
                    computeSVD = True
                    break
                else:
                    break
            elif answer == 'q' or answer == 'quit':
                sys.exit('Exiting program.')
            else:
                print('Invalid response. Please enter \'1\', or \'q/quit\'.')
        
    elif any(v is None for v in [s, U, V]) and args.numVals is not None and args.plot is False:
        if args.numVals >= 1:
            k = args.numVals
            computeSVD = True
                
        elif args.numVals < 1:
            userResponded = False
            print('''
                  ValueError: Argument '-k/--numVals' must be a positive integer 
                  between 1 and the order of the square input matrix. The parameter will
                  be set to the default value of 6.
                  What would you like to do?
                  Enter '1' to specify a value of the parameter. (Default)
                  Enter '2' to proceed with the default value.
                  Enter 'q/quit' exit the program.
                  ''')
            while userResponded == False:
                answer = input('Action: ')
                if answer == '' or answer == '1':
                    k = int(input('Please specify the number of singular values/vectors to compute: numVals = '))
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
                    sys.exit('Exiting program.')
                else:
                    print('Invalid response. Please enter \'1\', \'2\', or \'q/quit\'.')
                
    elif any(v is None for v in [s, U, V]) and args.numVals is None and args.plot is False:
        sys.exit('''
                 Nothing to be done. A singular-value decomposition does not exist.
                 Please specify at least one of '-k/--numVals' or '-p/--plot'
                 arguments with 'vzsvd' command.
                 ''')  
    #==============================================================================
    if computeSVD:
        # Read in the input file and set up the 3D scattered data array
        datadir = np.load('datadir.npz')
        scatteredData  = np.load(str(datadir['scatteredData']))
        if Path('window.npz').exists():
            print('Detected user-specified window:')
            windowDict = np.load('window.npz')
            
            # Apply the receiver window
            rstart = windowDict['rstart']
            rstop = windowDict['rstop']
            rstep = windowDict['rstep']
            rinterval = np.arange(rstart, rstop, rstep)
            print('\nwindow @ receivers : start = ', rstart)
            print('window @ receivers : stop = ', rstop)
            print('window @ receivers : step = ', rstep)
            
            # Apply the time window
            tstart = windowDict['tstart']
            tstop = windowDict['tstop']
            tstep = windowDict['tstep']
            print('window @ time : start = ', tstart)
            print('window @ time : stop = ', tstop)
            print('window @ time : step = ', tstep)
            
            recordingTimes = np.load(str(datadir['recordingTimes']))
            dt = (recordingTimes[-1] - recordingTimes[0]) / (len(recordingTimes) - 1)
            
            Tstart = int(round(tstart / dt))
            Tstop = int(round(tstop / dt))
            
            tinterval = np.arange(Tstart, Tstop, tstep)
            
            # Apply the source window
            sstart = windowDict['sstart']
            sstop = windowDict['sstop']
            sstep = windowDict['sstep']
            sinterval = np.arange(sstart, sstop, sstep)
            print('window @ sources : start = ', sstart)
            print('window @ sources : stop = ', sstop)
            print('window @ sources : step = ', sstep)
            
            print('\nApplying window to data volume...')
            scatteredData = scatteredData[rinterval, :, :]
            scatteredData = scatteredData[:, tinterval, :]
            scatteredData = scatteredData[:, :, sinterval]
            Nr, Nt, Ns = scatteredData.shape
            
            # Apply tapered cosine (Tukey) window to time signals.
            # This ensures the fast fourier transform (FFT) used in
            # the definition of the matrix-vector product below is
            # acting on a function that is continuous at its edges.
            
            # Np : Number of samples in the dominant period T = 1 / peakFreq
            peakFreq = pulseFun.peakFreq
            Np = int(round(1 / (tstep * dt * peakFreq)))
            # alpha is set to taper over 6 of the dominant period of the
            # pulse function (3 periods from each end of the signal)
            alpha = 6 * Np / Nt
            print('Tapering time signals with Tukey window: %d'
                  %(int(round(alpha * 100))) + '%')
            TukeyWindow = tukey(Nt, alpha)
            scatteredData *= TukeyWindow[None, :, None]
            
        else:
            Nr, Nt, Ns = scatteredData.shape
        #==============================================================================
        # MatVec: Definition of the near-field matrix-vector product
        #
        # x: a random test vector
        # Nt: number of time samples
        # Nr: number of receivers
        # Ns: number of sources
        #
        # M = (  zeros     nearFieldMatrix 
        #      nearFieldMatrix.T    zeros  )
        #
        # shape(nearFieldMatrix) = (Nt * Nr) x (Nt * Ns)
        # shape(nearFieldMatrix.T) = (Nt * Ns) x (Nt * Nr)
        #
        # x = ( x1 
        #       x2 )
        #
        # shape(x1) = (Nt * Nr) x 1
        # shape(x2) = (Nt * Ns) x 1
        #
        # Output:
        #       y = Mx
        
        if Nr == Ns:
            def MatVec(x):        
                #global scatteredData
                Nr, Nt, Ns = scatteredData.shape
                
                # x1 is the first Nt * Nr elements of the vector x
                x1 = x[:(Nt * Nr)]
                
                # x2 is the last Nt * Ns elements of the vector x
                x2 = x[-(Nt * Ns):]
                
                # reshape x1 and x2 into matrices X1 and X2
                # X1 multiplies nearFieldMatrix.T (time reversal), so flip up-down
                X1 = np.flipud(np.reshape(x1, (Nt, Nr), order='F'))
                X2 = np.reshape(x2, (Nt, Ns), order='F')
                
                Y1 = np.zeros((Nt, Nr))
                Y2 = np.zeros((Nt, Ns))
                
                for i in range(Nr):
                    # Compute the matrix-vector product for nearFieldMatrix * X2
                    U = scatteredData[i, :, :]
                    # Circular convolution: pad time axis with zeros to length 2Nt - 1
                    circularConvolution = ifft(fft(U, n=2*Nt-1, axis=0) * fft(X2, n=2*Nt-1, axis=0), axis=0).real
                    convolutionMatrix = circularConvolution[:Nt, :]
                    Y1[:, i] = np.sum(convolutionMatrix, axis=1) # sum over all sources
                    
                    # Compute the matrix-vector product for nearFieldMatrix.T * X1
                    UT = scatteredData[:, :, i].T
                    # Circular convolution: pad time axis with zeros to length 2Nt - 1
                    circularConvolutionT = ifft(fft(UT, n=2*Nt-1, axis=0) * fft(X1, n=2*Nt-1, axis=0), axis=0).real
                    convolutionMatrixT = np.flipud(circularConvolutionT[:Nt, :])        
                    Y2[:, i] = np.sum(convolutionMatrixT, axis=1) # sum over all receivers
                    
                y1 = np.reshape(Y1, (Nt * Nr, 1), order='F')
                y2 = np.reshape(Y2, (Nt * Ns, 1), order='F')
                
                return np.concatenate((y1, y2))
            
        else:   # if Nr != Ns
            def MatVec(x):        
                #global scatteredData
                Nr, Nt, Ns = scatteredData.shape
                
                # x1 is the first Nt * Nr elements of the vector x
                x1 = x[:(Nt * Nr)]
                
                # x2 is the last Nt * Ns elements of the vector x
                x2 = x[-(Nt * Ns):]
                
                # reshape x1 and x2 into matrices X1 and X2
                # X1 multiplies nearFieldMatrix.T (time reversal), so flip up-down
                X1 = np.flipud(np.reshape(x1, (Nt, Nr), order='F'))
                X2 = np.reshape(x2, (Nt, Ns), order='F')
                
                Y1 = np.zeros((Nt, Nr))
                Y2 = np.zeros((Nt, Ns))
                
                for i in range(Nr):
                    # Compute the matrix-vector product for nearFieldMatrix * X2
                    U = scatteredData[i, :, :]
                    # Circular convolution: pad time axis with zeros to length 2Nt - 1
                    circularConvolution = ifft(fft(U, n=2*Nt-1, axis=0) * fft(X2, n=2*Nt-1, axis=0), axis=0).real
                    convolutionMatrix = circularConvolution[:Nt, :]
                    Y1[:, i] = np.sum(convolutionMatrix, axis=1) # sum over all sources
                    
                for j in range(Ns):
                    # Compute the matrix-vector product for nearFieldMatrix.T * X1
                    UT = scatteredData[:, :, j].T
                    # Circular convolution: pad time axis with zeros to length 2Nt - 1
                    circularConvolutionT = ifft(fft(UT, n=2*Nt-1, axis=0) * fft(X1, n=2*Nt-1, axis=0), axis=0).real
                    convolutionMatrixT = np.flipud(circularConvolutionT[:Nt, :])        
                    Y2[:, j] = np.sum(convolutionMatrixT, axis=1) # sum over all receivers
                    
                y1 = np.reshape(Y1, (Nt * Nr, 1), order='F')
                y2 = np.reshape(Y2, (Nt * Ns, 1), order='F')
                
                return np.concatenate((y1, y2))
            
        A = LinearOperator(shape=(Nt*Nr + Nt*Ns, Nt*Nr + Nt*Ns), matvec=MatVec)
        
        # Compute the k largest algebraic eigenvalues (which='LA') of the operator A
        # Eigenvalues are elements of the vector 's'
        # Eigenvectors are columns of 'W'
        # Singular values of nearFieldMatrix are equivalent to eigenvalues of A
        # Left singular vectors are the first Nt * Nr eigenvectors of W
        # Right singular vectors are the last Nt * Ns eigenvectors of W
        
        if k == 1:
            print('Computing SVD for 1 singular value/vector...')
        else:
            print('Computing SVD for %s singular values/vectors...' %(k))
        startTime = time.time()
        s, W = eigsh(A, k, which='LA')
        endTime = time.time()
        print('Elapsed time:', endTime - startTime, 'seconds')
        
        # sort the eigenvalues and corresponding eigenvectors in descending order
        # (i.e., largest to smallest)
        index = s.argsort()[::-1]   
        s = s[index]
        W = W[:, index]
        
        U = np.sqrt(2) * W[:(Nt * Nr), :]         # left singular vectors
        V = np.sqrt(2) * W[-(Nt * Ns):, :]        # right singular vectors
        
        # Write binary output with numpy    
        np.save('singularValues.npy', s)
        np.save('leftVectors.npy', U)
        np.save('rightVectors.npy', V)
    
    #==============================================================================    
    if args.plot and all(v is not None for v in [s, U, V]):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.spines['left'].set_color('k')
        ax.spines['right'].set_color('k')
        ax.spines['top'].set_color('k')
        ax.spines['bottom'].set_color('k')
        ax.plot(s, 'k.', clip_on=False)
        ax.set_title('Singular Values')
        ax.set_xlabel('n')
        ax.set_ylabel('$\sigma_n$')
        ax.set_xlim([0, len(s)])
        ax.set_ylim(bottom=0)
        ax.locator_params(axis='y', nticks=6)
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        plt.tight_layout()
        fig.savefig('singularValues.' + args.format, format=args.format, bbox_inches='tight')
        plt.show()
