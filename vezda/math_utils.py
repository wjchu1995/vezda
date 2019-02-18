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

def humanReadable(seconds):
    '''
    Convert elapsed time (in seconds) to human-readable 
    format (hours : minutes : seconds)
    '''
    
    h = int(seconds / 3600)
    m = int((seconds % 3600) / 60)
    s = seconds % 60.0
    
    return '{}h : {:>02}m : {:>05.2f}s'.format(h, m, s)

def nextPow2(i):
    '''
    Input: a positive integer i
    Output: the next power of 2 greater than or equal to i
    '''

    n = 2
    while n < i:
        n *= 2
    
    return n


def timeShift(data, tau, dt):
    '''
    Apply a time shift 'tau' to the data in the frequency domain
    
    data: a 3D array with time on axis=1
    tau: a constant representing the time shift
    dt: the length of the time step (used to generate the discretized frequency bins)
    '''
    Nt = data.shape[1]
    N = nextPow2(Nt)
    fftData = np.fft.rfft(data, n=N, axis=1)
    
    # Set up the phase vector e^(-i * omega * tau)
    iomega = 2j * np.pi * np.fft.rfftfreq(N, dt)
    phase = np.exp(-iomega * tau)
    
    # Apply time shift in the frequency domain (element-wise array multiplication)
    shiftedData = np.fft.irfft(fftData * phase[None, :, None], axis=1)
    
    return shiftedData[:, :Nt, :]