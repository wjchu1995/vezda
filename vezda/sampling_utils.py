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

def GreenFunc(observationPoints, sourcePoint, recordingTimes, velocity):
    '''
    Computes the Green function (radiating fundamental solution) in 
    two- or three-dimensional space.
    
     Inputs:
         observationPoints: Nr x (2,3) array, specifies the coordinates of the 
              observation points, where 'Nr' is the number of the points.
         sourcePoint: 1 x (2,3) array specifying the source point. 
         recordingTimes: recording time array (row or column vector of length 'Nt')
         velocity: a scalar specifying the wave speed
    
     Output:
         G: Nr x Nt array containing the computed Green function
    '''
    # get the number of space dimensions (2 or 3)
    dim = observationPoints.shape[1]
    
    # compute the distance between the observation points and the source point
    r = norm(observationPoints - sourcePoint, axis=1) # |x - z|
    
    T, R = np.meshgrid(recordingTimes, r)
    retardedTime = T - R / velocity
    
    # get machine precision
    eps = np.finfo(float).eps     # about 2e-16 (so we never divide by zero)
    
    if dim == 2:
        sqrtTR = np.lib.scimath.sqrt(T**2 - (R / velocity)**2)
        G = np.divide(1, 2 * np.pi * sqrtTR + eps)
    
    elif dim == 3:
        tol = recordingTimes[1] - recordingTimes[0]
        impulse = np.zeros(retardedTime.shape)
        impulse[np.abs(retardedTime) < tol] = 1
        G = np.divide(impulse, 4 * np.pi * R + eps)
        
    G[retardedTime<=0] = 0    # causality
    G = np.real(G)        
    
    return G


def testFunc(pulseFunc, observationPoints, sourcePoint, recordingTimes, velocity):
    '''
    Computes the right-hand side test function in the near-field equation in 
    two- or three-dimensional space.
    
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


def sampleSpaceTime(receiverPoints, recordingTimes, velocity, tau, x, y, z=None, pulse=None):
    '''
    Compute the free-space Green functions or test functions for a specified space-time 
    sampling grid.
    
    Inputs:
    receiverPoints: an array of the receiver locations in 2D or 3D space
    recordingTimes: an array of the recording times
    x: an array of sampling points along the x-axis
    y: an array of sampling points along the y-axis
    z: an array of sampling points along the z-axis (optional)
    tau: an array of sampling points along the time axis
    pulse: a function of time that describes the shape of the wave (optional)
    velocity: the velocity of the (constant) medium through which the waves propagate
    '''
    
    # get the number of receivers Nr and the number of recording times Nt
    Nr = receiverPoints.shape[0]
    Nt = len(recordingTimes)
    
    # get the number of sampling points along the space and time axes
    Nx = len(x)
    Ny = len(y)
    Ntau = len(tau)
    
    if pulse is None:
        # Compute free-space Green functions
        
        if z is None:
            # space-time is three dimensional
            
            # initialize arrays for sampling points and computed Green functions
            samplingPoints = np.zeros((Nx * Ny, 2))
            funcArray = np.zeros((Nr, Nt, Nx * Ny))
            
            k = 0
            for ix in trange(Nx, desc='Sampling space'):
                for iy in range(Ny):
                    samplingPoints[k, :] = np.asarray([x[ix], y[iy]])
                    funcArray[:, :, k] = GreenFunc(receiverPoints, samplingPoints[k, :],
                             recordingTimes - tau[0], velocity)
                    k += 1
                    sleep(0.001)
            
        else:
            # space-time is four dimensional
            Nz = len(z)
            
            # initialize arrays for sampling points and computed test functions
            samplingPoints = np.zeros((Nx * Ny * Nz, 3))
            funcArray = np.zeros((Nr, Nt, Nx * Ny * Nz))
            
            k = 0
            for ix in trange(Nx, desc='Sampling space'):
                for iy in range(Ny):
                    for iz in range(Nz):
                        samplingPoints[k, :] = np.asarray([x[ix], y[iy], z[iz]])
                        funcArray[:, :, k] = GreenFunc(receiverPoints, samplingPoints[k, :],
                                 recordingTimes - tau[0], velocity)
                        k += 1
                        sleep(0.001)
    
    else:
        # Compute free-space test functions
        
        if z is None:
            # space-time is three dimensional
            
            # initialize arrays for sampling points and computed test functions
            samplingPoints = np.zeros((Nx * Ny * Ntau, 3))
            funcArray = np.zeros((Nr, Nt, Nx * Ny, Ntau))
            
            k = 0 # counter for space-time sampling points
            for it in trange(Ntau, desc='Sampling time'):
                l = 0 # counter for spatial sampling points
                for ix in trange(Nx, desc='Sampling space', leave=False):
                    for iy in range(Ny):
                        samplingPoints[k, :] = np.asarray([x[ix], y[iy], tau[it]])
                        funcArray[:, :, l, it] = testFunc(pulse, receiverPoints, samplingPoints[k, :2],
                                 recordingTimes - tau[it], velocity)
                        k += 1
                        l += 1
                        sleep(0.001)
            
        else:
            # space-time is four dimensional
            Nz = len(z)
            
            # initialize arrays for sampling points and computed test functions
            samplingPoints = np.zeros((Nx * Ny * Nz * Ntau, 4))
            funcArray = np.zeros((Nr, Nt, Nx * Ny * Nz, Ntau))
            
            k = 0 # counter for space-time sampling points
            for it in trange(Ntau, desc='Sampling time'):
                l = 0 # counter for spatial sampling points
                for ix in trange(Nx, desc='Sampling space', leave=False):
                    for iy in range(Ny):
                        for iz in range(Nz):
                            samplingPoints[k, :] = np.asarray([x[ix], y[iy], z[iz], tau[it]])
                            funcArray[:, :, l, it] = testFunc(pulse, receiverPoints, samplingPoints[k, :3],
                                     recordingTimes - tau[it], velocity)
                            k += 1
                            l += 1
                            sleep(0.001)
                        
    return funcArray, samplingPoints



def samplingIsCurrent(Dict, recordingTimes, velocity, tau, x, y, z=None, peakFreq=None, peakTime=None):
    if z is None:
        
        if peakFreq is None and peakTime is None:
            
            if np.array_equal(Dict['x'], x) and np.array_equal(Dict['y'], y) and np.array_equal(Dict['tau'], tau):
                print('Space-time sampling grid is consistent...')
            
                if Dict['velocity'] == velocity:
                    print('Background velocity is consistent...')
                    
                    if np.array_equal(Dict['time'], recordingTimes):
                        print('Recording time interval is consistent...')
                        return True
                
                    else:
                        print('Current recording time interval is inconsistent...')
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
                    
                    if np.array_equal(Dict['time'], recordingTimes):
                        print('Recording time interval is consistent...')
                        return True
                
                    else:
                        print('Current recording time interval is inconsistent...')
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
                    
                    if np.array_equal(Dict['time'], recordingTimes):
                        print('Recording time interval is consistent...')
                        return True
                
                    else:
                        print('Current recording time interval is inconsistent...')
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
                    
                    if np.array_equal(Dict['time'], recordingTimes):
                        print('Recording time interval is consistent...')
                        return True
                
                    else:
                        print('Current recording time interval is inconsistent...')
                        return False                                            
                    
                else:
                    print('Current pulse function is inconsistent...')
                    return False
        
            else:
                print('Current space-time sampling grid is inconsistent...')
                return False