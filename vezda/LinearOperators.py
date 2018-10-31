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
def asConvolutionOperator(data):
    '''
    This function takes the data and defines a matrix-vector product for the 
    either the near-field operator (NFO) or the Lippmann-Schwinger operator (LSO).
    
    Input: a three-dimensional data array of shape Nr x Nt x Ns
    
    MatVec: Definition of the forward matrix-vector product
    rMatVec: Definition of the adjoint matrix-vector product
    
    *** Note *** 
    All reshape comands in this function definition use Fortran column-based
    indexing (order='F'). This is because SciPy uses the Fortran library
    ARPACK as a backend to 'eigsh' to compute spectral decompositions.
    
    x: an arbitrary input vector
    Nt: number of time samples
    Nr: number of receivers
    Ns: number of sources
    
    Matrix: discretized operator
    
    shape(Matrix) = (Nt * Nr) x (Nt * Ns)
    
    shape(x) = (Nt * Ns) x 1
    
    Output: the operator M such that y = Mx
    '''
    
    Nr, Nt, Ns = data.shape
    
    # get the next power of 2 greater than or equal to 2 * Nt
    # for efficient circular convolution via FFT
    N = nextPow2(2 * Nt)
    
    def MatVec(x):        
        
        # reshape x into a matrix X
        X = np.reshape(x, (Nt, Ns), order='F')
        
        Y = np.zeros((Nt, Nr))
            
        for i in range(Nr):
            # Compute the matrix-vector product for Matrix * X
            U = data[i, :, :]
            # Circular convolution: pad time axis (axis=0) with zeros to length N
            circularConvolution = irfft(rfft(U, n=N, axis=0) * rfft(X, n=N, axis=0), axis=0)
            convolutionMatrix = circularConvolution[:Nt, :]
            Y[:, i] = np.sum(convolutionMatrix, axis=1) # sum over sources
            
        y = np.reshape(Y, (Nt * Nr, 1), order='F')
        
        return y
    
    def rMatVec(x):        
        
        # X multiplies Matrix.T (time reversal), so time reverse X
        X = np.flipud(np.reshape(x, (Nt, Nr), order='F'))
        
        Y = np.zeros((Nt, Ns))
            
        for j in range(Ns):
            # Compute the matrix-vector product for Matrix.T * X
            UT = data[:, :, j].T
            # Circular convolution: pad time axis (axis=0) with zeros to length N
            circularConvolutionT = irfft(rfft(UT, n=N, axis=0) * rfft(X, n=N, axis=0), axis=0)
            convolutionMatrixT = np.flipud(circularConvolutionT[:Nt, :])
            Y[:, j] = np.sum(convolutionMatrixT, axis=1) # sum over receivers
            
        y = np.reshape(Y, (Nt * Ns, 1), order='F')
        
        return y
    
    return LinearOperator(shape=(Nt * Nr, Nt * Ns), matvec=MatVec, rmatvec=rMatVec)