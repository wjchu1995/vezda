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
from numpy.fft import rfft, irfft
from scipy.sparse.linalg import LinearOperator
from vezda.math_utils import nextPow2

#==============================================================================
def asConvolutionalOperator(data):
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
    
    Nr, Nm, Ns = data.shape
    if np.issubdtype(data.dtype, np.float_):
        # input data are real (time domain)
    
        # get the next power of 2 greater than or equal to 2*Nm
        # for efficient circular convolution via FFT
        N = nextPow2(2 * Nm)
    
        def forwardOperator(x):        
            # definition of the forward convolutional operator
            
            #reshape x into a matrix X
            X = np.reshape(x, (2*Nm-1, Ns), order='F')
            
            # initialize the output array Y for the range of Matrix
            Y = np.zeros((2*Nm-1, Nr), dtype=data.dtype)
                
            for i in range(Nr):
                # Compute the matrix-vector product for Matrix * X
                U = data[i, :, :]
                # Circular convolution: pad time axis (axis=0) with zeros to length N
                circularConvolution = irfft(rfft(U, n=N, axis=0) * rfft(X, n=N, axis=0), axis=0)
                convolutionMatrix = circularConvolution[:(2*Nm-1), :]
                Y[:, i] = np.sum(convolutionMatrix, axis=1) # sum over sources
                
            y = np.reshape(Y, ((2*Nm-1) * Nr, 1), order='F')
        
            return y
    
        def adjointOperator(x):               
            # definition of the adjoint convolutional operator
 
            # X multiplies Matrix.T (time reversal), so time reverse X
            X = np.flipud(np.reshape(x, (2*Nm-1, Nr), order='F'))
        
            # initialize the output array Y for the range of Matrix.T
            Y = np.zeros((2*Nm-1, Ns), dtype=data.dtype)
            
            for j in range(Ns):
                # Compute the matrix-vector product for Matrix.T * X
                UT = data[:, :, j].T
                # Circular convolution: pad time axis (axis=0) with zeros to length N
                circularConvolutionT = irfft(rfft(UT, n=N, axis=0) * rfft(X, n=N, axis=0), axis=0)
                convolutionMatrixT = np.flipud(circularConvolutionT[:(2*Nm-1), :])
                Y[:, j] = np.sum(convolutionMatrixT, axis=1) # sum over receivers
            
            y = np.reshape(Y, ((2*Nm-1) * Ns, 1), order='F')
        
            return y
        
        return LinearOperator(shape=((2*Nm-1) * Nr, (2*Nm-1) * Ns), matvec=forwardOperator,
                              rmatvec=adjointOperator, dtype=data.dtype)
        
    else:
        # input data are complex (frequency domain)
        
        def forwardOperator(x):        
            # definition of the forward convolutional operator
            
            #reshape x into a matrix X
            X = np.reshape(x, (Nm, Ns), order='F')
            
            # initialize the output array Y for the range of Matrix
            Y = np.zeros((Nm, Nr), dtype=data.dtype)
            
            for i in range(Nr):
                # Compute the matrix-vector product for Matrix * X
                U = data[i, :, :]
                # Circular convolution is element-wise multiplication in frequency domain
                circularConvolution = U * X
                Y[:, i] = np.sum(circularConvolution, axis=1) # sum over sources
            
            y = np.reshape(Y, (Nm * Nr, 1), order='F')
    
            return y
    
        def adjointOperator(x):               
            # definition of the adjoint convolutional operator
            
            #reshape x into a matrix X
            X = np.reshape(x, (Nm, Nr), order='F')
            
            # initialize the output array Y for the range of Matrix.T
            Y = np.zeros((Nm, Ns), dtype=data.dtype)
        
            for j in range(Ns):
                # Compute the matrix-vector product for Matrix.H * X
                UH = data[:, :, j].conj().T
                # Adjoint of circular convolution is element-wise multiplication in frequency domain
                circularConvolutionH = UH * X
                Y[:, j] = np.sum(circularConvolutionH, axis=1) # sum over receivers
        
            y = np.reshape(Y, (Nm * Ns, 1), order='F')
        
            return y
    
        return LinearOperator(shape=(Nm * Nr, Nm * Ns), matvec=forwardOperator,
                              rmatvec=adjointOperator, dtype=data.dtype)
