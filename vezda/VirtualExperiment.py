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
from pathlib import Path
from vezda.sampling_utils import testFunc
from vezda.LinearOperators import asNearFieldOperator
import textwrap

sys.path.append(os.getcwd())
import pulseFun

datadir = np.load('datadir.npz')
recordingTimes = np.load(str(datadir['recordingTimes']))
receiverPoints = np.load(str(datadir['receivers']))

if 'testFuncs' in datadir and not Path('VZTestFuncs.npz').exists():
    TFtype = 'user'
    TFarray = np.load(str(datadir['testFuncs']))
    #time = recordingTimes
    samplingPoints = np.load(str(datadir['samplingPoints']))
    sourcePoints = samplingPoints[:, :-1]
                    
elif not 'testFuncs' in datadir and Path('VZTestFuncs.npz').exists():
    TFtype = 'vezda'
    testFuncs = np.load('VZTestFuncs.npz')
    TFarray = testFuncs['TFarray']
    time = testFuncs['time'] 
    samplingPoints = testFuncs['samplingPoints']
    # Extract all but last column of sampling points,
    # which corresponds to sampling points in time
    #sourcePoints = samplingPoints[:, :-1]
            
elif 'testFuncs' in datadir and Path('VZTestFuncs.npz').exists():
    userResponded = False
    print(textwrap.dedent(
          '''
          Two files are available containing simulated test functions.
          
          Enter '1' to view the user-provided test functions. (Default)
          Enter '2' to view the test functions computed by Vezda.
          Enter 'q/quit' to exit.
          '''))
    while userResponded == False:
        answer = input('Action: ')
        
        if answer == '' or answer == '1':
            TFtype = 'user'
            TFarray = np.load(str(datadir['testFuncs']))
            #time = recordingTimes
            samplingPoints = np.load(str(datadir['samplingPoints']))
            sourcePoints = samplingPoints[:, :-1]
            userResponded = True
            break
        
        elif answer == '2':
            TFtype = 'vezda'
            testFuncs = np.load('VZTestFuncs.npz')
            TFarray = testFuncs['TFarray']
            X = TFarray[:, :, :, 0]
            time = testFuncs['time'] 
            samplingPoints = testFuncs['samplingPoints']
            # Extract all but last column of sampling points,
            # which corresponds to sampling points in time
            sourcePoints = samplingPoints[:, :-1]
            userResponded = True
        
        elif answer == 'q' or answer == 'quit':
            sys.exit('Exiting program.')
        
        else:
            print('Invalid response. Please enter \'1\', \'2\', or \'q/quit\'.')
        
else:
    sys.exit(textwrap.dedent(
            '''
            Error: No test functions have been found to plot.
            '''))
    
pulse = lambda t : pulseFun.pulse(t)
velocity = pulseFun.velocity    # only used if medium == constant

# Compute the incident waves Ui to each sampling point from the source surface
Ui = np.zeros((Nx * Ny, Nt, Ns))
for s in range(Ns):
    Ui[:, :, s] = testFunc(pulse, samplingPoints[:, :2], sourcePoints[i, :],
      recordingTimes, velocity)
    


# Construct the Herglotz wave function v_phi
N = asNearFieldOperator(Ui)
v_phi = N.matvec(phi)

Gz = FundSol(pulse, recordingTimes - tau[it], velocity, samplingPoints[k, :2],
             focusingPoints[k, :2])

virtual_wave = v_phi + Gz

# load test functions from sampling points to receivers
TFarray

for i in range(Nr):
    # Compute the matrix-vector product for nearFieldMatrix * X2
    U = incidentField[i, :, :]
    # Circular convolution: pad time axis with zeros to length 2Nt - 1
    circularConvolution = ifft(fft(U, n=N, axis=0) * fft(phi, n=N, axis=0), axis=0).real
    convolutionMatrix = circularConvolution[:Nt, :]
    Y1[:, i] = np.sum(convolutionMatrix, axis=1) # sum over all sources
            
    
