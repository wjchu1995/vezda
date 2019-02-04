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
from scipy.sparse import csc_matrix, csr_matrix

def saveSVD(U, s, Vh, args, domain):
    
    if args.nfo:
        filename = 'NFO_SVD.npz'
    elif args.lso:
        filename = 'LSO_SVD.npz'
    
    if domain == 'freq':        
        np.savez(filename,
                 U_data=U.data, U_indices=U.indices, U_indptr=U.indptr,
                 U_shape=U.shape,
                 Vh_data=Vh.data, Vh_indices=Vh.indices, Vh_indptr=Vh.indptr,
                 Vh_shape=Vh.shape,
                 s=s, domain=domain)
    
    elif domain == 'time':
        np.savez(filename, U=U, s=s, Vh=Vh, domain=domain)        
    

def loadSVD(filename):
    loader = np.load(filename)
    s = loader['s']
    domain = loader['domain']
    
    if domain == 'freq':
        U = csc_matrix((loader['U_data'], loader['U_indices'], loader['U_indptr']),
                       shape=loader['U_shape'])
        Vh = csr_matrix((loader['Vh_data'], loader['Vh_indices'], loader['Vh_indptr']),
                        shape=loader['Vh_shape'])
    
    elif domain == 'time':
        U = loader['U']
        Vh = loader['Vh']
    
    return U, s, Vh, domain
    

def makeSparse(A, r, compressedFormat):
    '''
    Return a sparse representation of a matrix A based on the r largest nonzero
    row/column elements.
    
    A: a dense matrix (2D array) to make sparse
    r: a positive integer giving the number of nonzero row/column elements to extract from A
    compressedFormat: compressed storage format of resulting sparse matrix.
    
    'csc' format results in a sparse column matrix. Extracts the r largest nonzero
    elements along each column of A. Best for 'tall' matrices.
    
    'csr' format results in a sparse row matrix. Extracts the r largest nonzero
    elements along each row of A. Best for 'wide' matrices.
    '''    
    M, N = A.shape
    
    if compressedFormat == 'csc':
        indx = np.argpartition(-np.abs(A), r, axis=0)[:r, :]
        data = A[indx, np.arange(N)].reshape(r * N, order='F')
        
        rows = indx.reshape(-1, order='F')
        cols = (np.ones((N, r), np.int_) * np.arange(N)[:, None]).reshape(-1)
        
        return csc_matrix((data, (rows, cols)), shape=(M, N))
        
    
    elif compressedFormat == 'csr':
        indx = np.argpartition(-np.abs(A), r, axis=1)[:, :r]
        data = A[np.arange(M)[:, None], indx].reshape(r * M)

        rows = (np.ones((M, r), np.int_) * np.arange(M)[:, None]).reshape(-1)
        cols = indx.reshape(-1)
        
        return csr_matrix((data, (rows, cols)), shape=(M, N))
    

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