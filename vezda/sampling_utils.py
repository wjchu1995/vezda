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
import numpy as np
from scipy.linalg import norm
from tqdm import trange
from time import sleep

def testFunc(pulseFunc, observationPoints, sourcePoint, recordingTimes, velocity):
    '''
    Computes test functions for two- or three-dimensional free space. Test functions
    are simply free-space Green functions convolved with a smooth time-dependent pulse
    function.
    
     Inputs:
         pulseFunc: a function handle, gives the time depedence of the pulse function
         observationPoints: Nr x (2,3) array, specifies the coordinates of the 
              observation points, where 'Nr' is the number of the points.
         sourcePoint: 1 x (2,3) array specifying the source point. 
         recordingTimes: recording time array (row or column vector of length 'Nt')
         velocity: a scalar specifying the wave speed
    
     Output:
         testFunc: Nr x Nt array containing the computed test function
    '''
    # get the number of space dimensions (2 or 3)
    dim = observationPoints.shape[1]
    
    # compute the distance between the observation points and the source point
    r = norm(observationPoints - sourcePoint, axis=1) # |x - z|
    
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


def sampleSpace(receiverPoints, recordingTimes, velocity, x, y, z=None, pulse=None):
    '''
    Compute the free-space Green functions or test functions for a specified 
    sampling grid.
    
    Inputs:
    receiverPoints: an array of the receiver locations in 2D or 3D space
    recordingTimes: an array of the recording times
    x: an array of sampling points along the x-axis
    y: an array of sampling points along the y-axis
    z: an array of sampling points along the z-axis (optional)
    pulse: a function of time that describes the shape of the wave (optional)
    velocity: the velocity of the (constant) medium through which the waves propagate
    '''
    
    # get the number of receivers Nr and the number of recording times Nt
    Nr = receiverPoints.shape[0]
    Nt = len(recordingTimes)
    
    # get the number of sampling points along the space and time axes
    Nx = len(x)
    Ny = len(y)
        
    if z is None:
        # space is two dimensional
        
        # initialize arrays for sampling points and computed Green functions
        samplingPoints = np.zeros((Nx * Ny, 2))
        funcArray = np.zeros((Nr, Nt, Nx * Ny))
            
        # Compute free-space test functions
        k = 0 # counter for spatial sampling points
        for ix in trange(Nx, desc='Sampling space', leave=False):
            for iy in range(Ny):
                samplingPoints[k, :] = np.asarray([x[ix], y[iy]])
                funcArray[:, :, k] = testFunc(pulse, receiverPoints, samplingPoints[k, :],
                         recordingTimes, velocity)
                k += 1
                sleep(0.001)
    
    else:
        # space is three dimensional
        Nz = len(z)
        
        # initialize arrays for sampling points and computed test functions
        samplingPoints = np.zeros((Nx * Ny * Nz, 3))
        funcArray = np.zeros((Nr, Nt, Nx * Ny * Nz))
        
        # Compute free-space test functions
        k = 0 # counter for spatial sampling points
        for ix in trange(Nx, desc='Sampling space', leave=False):
            for iy in range(Ny):
                for iz in range(Nz):
                    samplingPoints[k, :] = np.asarray([x[ix], y[iy], z[iz]])
                    funcArray[:, :, k] = testFunc(pulse, receiverPoints, samplingPoints[k, :],
                             recordingTimes, velocity)
                    k += 1
                    sleep(0.001)
                        
    return funcArray, samplingPoints



def samplingIsCurrent(Dict, receiverPoints, recordingTimes, velocity, tau, x, y, z=None, peakFreq=None, peakTime=None):
    if z is None:
        
        if peakFreq is None and peakTime is None:
            
            if np.array_equal(Dict['x'], x) and np.array_equal(Dict['y'], y) and np.array_equal(Dict['tau'], tau):
                print('Space-time sampling grid is consistent...')
            
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
                print('Current space-time sampling grid is inconsistent...')
                return False
            
        else:
            
            if np.array_equal(Dict['x'], x) and np.array_equal(Dict['y'], y) and np.array_equal(Dict['tau'], tau):
                print('Space-time sampling grid is consistent...')
            
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
                print('Current space-time sampling grid is inconsistent...')
                return False
        
    
    else:
        
        if peakFreq is None and peakTime is None:
            
            if np.array_equal(Dict['x'], x) and np.array_equal(Dict['y'], y) and np.array_equal(Dict['z'], z) and np.array_equal(Dict['tau'], tau):
                print('Space-time sampling grid is consistent...')
            
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
                print('Current space-time sampling grid is inconsistent...')
                return False
            
        else:
            
            if np.array_equal(Dict['x'], x) and np.array_equal(Dict['y'], y) and np.array_equal(Dict['z'], z) and np.array_equal(Dict['tau'], tau):
                print('Space-time sampling grid is consistent...')
            
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
                print('Current space-time sampling grid is inconsistent...')
                return False