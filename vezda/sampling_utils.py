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
import numpy as np
from scipy.linalg import norm
from tqdm import trange
from time import sleep

def testFunc(receiverPoints, recordingTimes, sourcePoint, velocity, pulseFunc):
    '''
    Computes test functions for two- or three-dimensional free space. Test functions
    are simply free-space Green functions convolved with a smooth time-dependent pulse
    function.
    
     Inputs:
         receiverPoints: Nr x (2,3) array, specifies the coordinates of the 
              receiver points, where 'Nr' is the number of the points.
         recordingTimes: recording time array (row or column vector of length 'Nt')
         sourcePoint: 1 x (2,3) array specifying the source point. 
         velocity: a scalar specifying the wave speed
         pulseFunc: a function handle, gives the time depedence of the pulse function
    
     Output:
         testFunc: Nr x Nt array containing the computed test function
    '''
    # get the number of space dimensions (2 or 3)
    dim = receiverPoints.shape[1]
    
    # compute the distance between the receiver points and the source point
    r = norm(receiverPoints - sourcePoint, axis=1) # |x - z|
    
    T, R = np.meshgrid(recordingTimes, r)
    retardedTime = T - R / velocity
    
    # get machine precision
    eps = np.finfo(float).eps     # about 2e-16 (so we never divide by zero)
    pulse = pulseFunc(retardedTime)
    if dim == 2:
        sqrtTR = np.lib.scimath.sqrt(T**2 - (R / velocity)**2)
        testFunc = np.divide(pulse, 2 * np.pi * sqrtTR + eps)
    
    elif dim == 3:
        testFunc = np.divide(pulse, 4 * np.pi * R + eps)
        
    testFunc[retardedTime<=0] = 0    # causality
    testFunc = np.real(testFunc)        
    
    return testFunc


def sampleSpace(receiverPoints, recordingTimes, samplingPoints, velocity, pulse):
    '''
    Compute the free-space Green functions or test functions for a specified 
    sampling grid.
    
    Inputs:
    receiverPoints: an array of the receiver locations in 2D or 3D space
    recordingTimes: an array of the recording times
    samplingPoints: an array of sampling points in 2D or 3D space
    velocity: the velocity of the (constant) medium through which the waves propagate
    pulse: a function of time that describes the shape of the wave
    '''
    
    # get the number of receivers, time samples, and sources
    Nr = receiverPoints.shape[0]
    Nt = len(recordingTimes)
    Ns = samplingPoints.shape[0]
        
    if Nr < Ns:
        # Use source-receiver reciprocity to efficiently compute test functions
        funcArray = np.zeros((Ns, Nt, Nr))
        for i in trange(Nr, desc='Sampling space'):
            funcArray[:, :, i] = testFunc(samplingPoints, recordingTimes,
                     receiverPoints[i, :], velocity, pulse)
            sleep(0.001)
        funcArray = np.swapaxes(funcArray, 0, 2)
        
    else:
        funcArray = np.zeros((Nr, Nt, Ns))
        for i in trange(Ns, desc='Sampling space'):
            funcArray[:, :, i] = testFunc(receiverPoints, recordingTimes,
                     samplingPoints[i, :], velocity, pulse)
            sleep(0.001)
        
    return funcArray


def samplingIsCurrent(Dict, receiverPoints, recordingTimes, samplingPoints, tau, velocity, peakFreq=None, peakTime=None):
    if peakFreq is None and peakTime is None:
        
        if np.array_equal(Dict['samplingPoints'], samplingPoints) and np.array_equal(Dict['tau'], tau):
            print('Sampling grid and focusing time are consistent...')
        
            if Dict['velocity'] == velocity:
                print('Background velocity is consistent...')
                    
                if np.array_equal(Dict['receivers'], receiverPoints):
                    print('Receiver points are consistent...')
                        
                    if np.array_equal(Dict['time'], recordingTimes):
                        print('Recording time interval is consistent...')
                        return True
                        
                    else:
                        print('Current recording time interval is inconsistent...')
                        return False
                    
                else:
                    print('Current receiver points are inconsistent...')
                    return False
                    
            else:
                print('Current pulse function is inconsistent...')
                return False
        
        else:
            print('Current sampling grid or focusing time is inconsistent...')
            return False
            
    else:
        
        if np.array_equal(Dict['samplingPoints'], samplingPoints) and np.array_equal(Dict['tau'], tau):
            print('Sampling grid and focusing time are consistent...')
            
            if Dict['peakFreq'] == peakFreq and Dict['peakTime'] == peakTime and Dict['velocity'] == velocity:
                print('Pulse function and background velocity are consistent...')
                    
                if np.array_equal(Dict['receivers'], receiverPoints):
                    print('Receiver points are consistent...')
                        
                    if np.array_equal(Dict['time'], recordingTimes):
                        print('Recording time interval is consistent...')
                        return True
                
                    else:
                        print('Current recording time interval is inconsistent...')
                        return False                                            
                    
                else:
                    print('Current receiver points are inconsistent...')
                    return False
                    
            else:
                print('Current pulse function is inconsistent...')
                return False
        
        else:
            print('Current sampling grid or focusing time is inconsistent...')
            return False