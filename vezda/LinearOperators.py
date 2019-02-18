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
from scipy.sparse.linalg import LinearOperator
from vezda.math_utils import nextPow2

#==============================================================================
def asConvolutionalOperator(kernel):
    '''
    This function takes a 3D data array as input and defines matrix-vector products 
    for both the forward and adjoint convolution operators. The convolutional operator
    can either be the near-field operator (NFO) or the Lippmann-Schwinger operator (LSO).
    
    Input: a three-dimensional data array of shape Nr x Nm x Ns
    
    matvec: Definition of the forward matrix-vector product
    rmatvec: Definition of the adjoint matrix-vector product
    
    *** Note *** 
    All reshape comands in this function definition use Fortran column-based
    indexing (order='F'). This is because SciPy uses the Fortran library
    ARPACK as a backend to compute spectral decompositions.
    
    x: an arbitrary input vector
    Nm: number of time/frequency samples
    Nr: number of receivers
    Ns: number of sources
    
    Matrix: discretized kernel of the convolutional operator. The input data
            form the kernel of the operator.
    
    shape(Matrix) = (Nm * Nr) x (Nm * Ns)
    
    shape(x) = (Nm * Ns) x 1 for forward operator
    shape(x) = (Nm * Nr) x 1 for adjoint operator
    
    Output: the operator M such that y = Mx
    '''
    
    Nr, Nm, Ns = kernel.shape
    if np.issubdtype(kernel.dtype, np.floating):
        # input data are real (time domain)
    
        # get the next power of 2 greater than or equal to 2*Nm
        # for efficient circular convolution via FFT
        N = nextPow2(2 * Nm)
        
        # Fourier transform the data over the time axis=1
        U = np.fft.rfft(kernel, n=N, axis=1)
        
        def forwardOperator(x):        
            # definition of the forward convolutional operator
            
            #reshape x into a matrix and FFT over time axis=0
            x = x.reshape((2*Nm-1, Ns), order='F')
            x = np.fft.rfft(x, n=N, axis=0)
            
            # initialize the output array y for the range of Matrix
            y = np.zeros((2*Nm-1, Nr), dtype=kernel.dtype)
                
            for i in range(Nr):
                # Compute the matrix-vector product for Matrix * x
                circularConvolution = np.fft.irfft(U[i, :, :] * x, axis=0)
                circularConvolution = circularConvolution[:(2*Nm-1), :]
                y[:, i] = np.sum(circularConvolution, axis=1) # sum over sources
                
            y = y.reshape(((2*Nm-1) * Nr, 1), order='F')
        
            return y
    
        def adjointOperator(y):               
            # definition of the adjoint convolutional operator
 
            #reshape y into a matrix and FFT over time axis=0
            y = y.reshape((2*Nm-1, Nr), order='F')
            y = np.fft.rfft(y, n=N, axis=0)
        
            # initialize the output array x for the range of Matrix.T
            x = np.zeros((2*Nm-1, Ns), dtype=kernel.dtype)
            
            for j in range(Ns):
                # Compute the matrix-vector product for Matrix.T * y
                circularConvolutionT = np.fft.irfft(U[:, :, j].conj().T * y, axis=0)
                circularConvolutionT = circularConvolutionT[:(2*Nm-1), :]
                x[:, j] = np.sum(circularConvolutionT, axis=1) # sum over receivers
            
            x = x.reshape(((2*Nm-1) * Ns, 1), order='F')
        
            return x
        
        return LinearOperator(shape=((2*Nm-1) * Nr, (2*Nm-1) * Ns), matvec=forwardOperator,
                              rmatvec=adjointOperator, dtype=kernel.dtype)
        
    else:
        # input data are complex (frequency domain)
        
        def forwardOperator(x):        
            # definition of the forward convolutional operator

            #reshape x into a matrix
            x = x.reshape((Nm, Ns), order='F')
            
            # initialize the output array y for the range of Matrix
            y = np.zeros((Nm, Nr), dtype=kernel.dtype)
            
            for i in range(Nr):
                y[:, i] = np.sum(kernel[i, :, :] * x, axis=1) # sum over sources
            
            y = y.reshape((Nm * Nr, 1), order='F')
    
            return y
    
        def adjointOperator(y):               
            # definition of the adjoint convolutional operator

            #reshape y into a matrix
            y = y.reshape((Nm, Nr), order='F')
            
            # initialize the output array x for the range of Matrix.H
            x = np.zeros((Nm, Ns), dtype=kernel.dtype)
        
            for j in range(Ns):
                x[:, j] = np.sum(kernel[:, :, j].conj().T * y, axis=1) # sum over receivers
        
            x = x.reshape((Nm * Ns, 1), order='F')
        
            return x
    
        return LinearOperator(shape=(Nm * Nr, Nm * Ns), matvec=forwardOperator,
                              rmatvec=adjointOperator, dtype=kernel.dtype)
