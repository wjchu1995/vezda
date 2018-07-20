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
from scipy.fftpack import fft, ifft, fftfreq

def nextPow2(i):
    '''
    Input: a positive integer i
    Output: the next power of 2 greater than or equal to i
    '''

    n = 1
    while n < i:
        n *= 2
    
    return n


def timeShift(data, tau, dt):
    '''
    Apply a time shift 'tau' to the data via FFT
    
    data: a 3D array with time on axis=1
    tau: a constant representing the time shift
    dt: the length of the time step (used to generate the discretized frequency bins)
    '''
    
    # Get the shape of the data array
    # Nr : number of receivers
    # Nt : number of time samples
    # Ns : number of sources
    Nr, Nt, Ns = data.shape
    
    # get the next power of 2 greater than or equal to 2 * Nt
    # for efficient FFT
    N = nextPow2(2 * Nt)
    
    # FFT the 'data' array into the frequency domain along the time axis=1
    freqData = fft(data, n=N, axis=1)
    
    # Set up the phase vector e^(-i * omega * tau)
    iomega = 2j * np.pi * fftfreq(N, dt)
    phase = np.exp(-iomega * tau)
    
    # Apply time shift in the frequency domain (element-wise array multiplication)
    # and inverse FFT back into the time domain (keeping only the real part)
    shiftedData = ifft(freqData * phase[None, :, None], axis=1).real
    
    # Return the shifted data array with the first Nt components along
    # the time axis (axis=1) corresponding to the original length of the
    # time axis.
    return shiftedData[:, :Nt, :]