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
from scipy.sparse.linalg import eigsh
from scipy.signal import tukey
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from vezda.plot_utils import (vector_title, remove_keymap_conflicts, plotWiggles,
                              process_key_vectors, default_params, setFigure)
from vezda.LinearOperators import asSymmetricOperator
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
                 between 1 and the order of the square input matrix.
                 '''))
            break
        
    return isValid

#==============================================================================
def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--numVals', '-k', type=int,
                        help='''specify the number of singular values/vectors to compute.
                        Must a positive integer between 1 and the order of the square
                        input matrix.''')
    parser.add_argument('--plot', '-p', action='store_true',
                        help='''Plot the computed singular values and vectors.''')
    parser.add_argument('--format', '-f', type=str, default='pdf', choices=['png', 'pdf', 'ps', 'eps', 'svg'],
                        help='''specify the image format of the saved file. Accepted formats are png, pdf,
                        ps, eps, and svg. Default format is set to pdf.''')
    parser.add_argument('--mode', type=str, choices=['light', 'dark'], required=False,
                        help='''Specify whether to view plots in light mode for daytime viewing
                        or dark mode for night viewing.
                        Mode must be either \'light\' or \'dark\'.''')
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
            print(textwrap.dedent(
                 '''
                 A singular-value decomposition for {n} values/vectors already exists. 
                 What would you like to do?
                 
                 Enter '1' to specify a new number of values/vectors to compute. (Default)
                 Enter '2' to recompute a singular-value decomposition for {n} values/vectors.
                 Enter 'q/quit' to exit.
                 '''.format(n=args.numVals)))
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
            print(textwrap.dedent(
                 '''
                 A singular-value decomposition for {n} values/vectors already exists. 
                 What would you like to do?
                 
                 Enter '1' to specify a new number of values/vectors to compute. (Default)
                 Enter '2' to recompute a singular-value decomposition for {n} values/vectors.
                 Enter 'q/quit' to exit.
                 '''.format(n=args.numVals)))
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
        sys.exit(textwrap.dedent(
                '''
                No action specified. A singular-value decomposition for %s values/vectors
                already exists. Please specify at least one of '-k/--numVals' or '-p/--plot'
                arguments with 'vzsvd' command.
                ''' %(len(s))))
    #==============================================================================
    # if an SVD does not already exist...
    elif any(v is None for v in [s, U, V]) and args.numVals is not None and args.plot is True:
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
        print(textwrap.dedent(
             '''
             PlotError: A singular-value decomposition does not exist. A plot will be
             generated after a singular-value decomposition has been computed.
             
             Enter '1' to specify a number of singular values/vectors to compute. (Default)
             Enter 'q/quit' to exit.
             '''))
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
        sys.exit(textwrap.dedent(
                '''
                Nothing to be done. A singular-value decomposition does not exist.
                Please specify at least one of '-k/--numVals' or '-p/--plot'
                arguments with 'vzsvd' command.
                '''))  
    #==============================================================================
    # Read in input files 
    datadir = np.load('datadir.npz')
    recordingTimes = np.load(str(datadir['recordingTimes']))
    receiverPoints = np.load(str(datadir['receivers']))
    sourcePoints = np.load(str(datadir['sources']))
    
    t0 = recordingTimes[0]
    tf = recordingTimes[-1]
    
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
        
        dt = (tf - t0) / (len(recordingTimes) - 1)
        
        twStart = int(round(tstart / dt))
        twStop = int(round(tstop / dt))
        
        # Apply the source window
        sstart = windowDict['sstart']
        sstop = windowDict['sstop']
        sstep = windowDict['sstep']
        
    else:
        rstart = 0
        rstop = receiverPoints.shape[0]
        rstep = 1
        
        sstart = 0
        sstop = sourcePoints.shape[0]
        sstep = 1
        
        twStart = 0
        twStop = len(recordingTimes)
        tstep = 1
        
    # Apply the receiver window
    rinterval = np.arange(rstart, rstop, rstep)
    receiverPoints = receiverPoints[rinterval, :]
    
    # Apply the time window
    tinterval = np.arange(twStart, twStop, tstep)
    recordingTimes = recordingTimes[tinterval]
    
    # Apply the source window
    sinterval = np.arange(sstart, sstop, sstep)
    sourcePoints = sourcePoints[sinterval, :]
    
    if computeSVD:
        # set up the 3D data array
        data  = np.load(str(datadir['scatteredData']))
        if Path('window.npz').exists():
            print('Detected user-specified window:\n')
            
            print('window @ receivers : start =', rstart)
            print('window @ receivers : stop =', rstop)
            print('window @ receivers : step =', rstep, '\n')
            
            print('window @ time : start =', tstart)
            print('window @ time : stop =', tstop)
            print('window @ time : step =', tstep, '\n')
            
            print('window @ sources : start =', sstart)
            print('window @ sources : stop =', sstop)
            print('window @ sources : step =', sstep, '\n')
            
            print('Applying window to data volume...')
            data = data[rinterval, :, :]
            data = data[:, tinterval, :]
            data = data[:, :, sinterval]
            Nr, Nt, Ns = data.shape
            
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
            data *= TukeyWindow[None, :, None]
            
        else:
            Nr, Nt, Ns = data.shape
            
        #==============================================================================
        A = asSymmetricOperator(data)
        
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
        print('Elapsed time:', humanReadable(endTime - startTime), '\n')
        
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
        remove_keymap_conflicts({'left', 'right', 'up', 'down', 'save'})
        
        Nr = receiverPoints.shape[0]
        Nt = len(recordingTimes)
        Ns = sourcePoints.shape[0]
        k = len(s)
        
        # Reshape singular vectors for plotting
        U = np.reshape(U, (Nr, Nt, k))
        V = np.reshape(V, (Ns, Nt, k))
        
        if Path('plotParams.pkl').exists():
            plotParams = pickle.load(open('plotParams.pkl', 'rb'))
        
        else:
            plotParams = default_params()
            
        if args.mode is not None:
            plotParams['view_mode'] = args.mode
            pickle.dump(plotParams, open('plotParams.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)
        
        fig_vec, ax1_vec, ax2_vec = setFigure(num_axes=2, mode=plotParams['view_mode'])
        fig_vals, ax_vals = setFigure(num_axes=1, mode=plotParams['view_mode'])
        
        ax1_vec.volume = U
        ax1_vec.index = 0
        leftTitle = vector_title('left', ax1_vec.index)
        plotWiggles(ax1_vec, U[:, :, ax1_vec.index], recordingTimes, t0, tf, rstart, rinterval,
                    receiverPoints, leftTitle, 'left', plotParams)
      
        ax2_vec.volume = V
        ax2_vec.index = ax1_vec.index
        rightTitle = vector_title('right', ax2_vec.index)
        plotWiggles(ax2_vec, V[:, :, ax2_vec.index], recordingTimes, t0, tf, sstart, sinterval,
                    sourcePoints, rightTitle, 'right', plotParams)
        plt.tight_layout()
        fig_vec.canvas.mpl_connect('key_press_event', lambda event: process_key_vectors(event, recordingTimes, t0, tf, rstart, sstart, 
                                                                                    rinterval, sinterval, receiverPoints, 
                                                                                    sourcePoints, plotParams))
        #==============================================================================
        # plot the singular values
        kappa = s[0] / s[-1]    # condition number = max(s) / min(s)
        ax_vals.plot(s, '.', clip_on=False, markersize=9, label=r'Condition Number: %0.1e' %(kappa), color=ax_vals.pointcolor)
        ax_vals.set_xlabel('n', color=ax_vals.labelcolor)
        ax_vals.set_ylabel('$\sigma_n$', color=ax_vals.labelcolor)
        legend = ax_vals.legend(title='Singular Values', loc='upper center', bbox_to_anchor=(0.5, 1.25),
                           markerscale=0, handlelength=0, handletextpad=0, fancybox=True, shadow=True,
                           fontsize='large')
        legend.get_title().set_fontsize('large')
        ax_vals.set_xlim([0, k])
        ax_vals.set_ylim(bottom=0)
        ax_vals.locator_params(axis='y', nticks=6)
        ax_vals.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        plt.tight_layout()
        fig_vals.savefig('singularValues.' + args.format, format=args.format, bbox_inches='tight')
        
        plt.show()
