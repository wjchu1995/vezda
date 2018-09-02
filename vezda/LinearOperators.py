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
# Time-domain implementations

def asSymmetricConvolutionOperator(data):
    '''
    This function is used to create a symmetric (square) matrix for the
    singular-value decomposition of either the near-field operator (NFO)
    or the Lippmann-Schwinger operator (LSO). It takes the data and defines
    a matrix-vector product for both the operator and its adjoint.
    
    Input: a three-dimensional data array of shape Nr x Nt x Ns
    
    MatVec: Definition of a matrix-vector product
    
    *** Note *** 
    All reshape comands in this function definition use Fortran column-based
    indexing (order='F'). This is because SciPy uses the Fortran library
    ARPACK as a backend to 'eigsh' to compute spectral decompositions.
    
    x: an arbitrary input vector
    Nt: number of time samples
    Nr: number of receivers
    Ns: number of sources
    
    M = (  zeros     Matrix 
         Matrix.T    zeros  )
    
    shape(Matrix) = (Nt * Nr) x (Nt * Ns)
    shape(Matrix.T) = (Nt * Ns) x (Nt * Nr)
    
    x = ( x1 
          x2 )
    
    shape(x1) = (Nt * Nr) x 1
    shape(x2) = (Nt * Ns) x 1
    
    Output: the operator M such that y = Mx
    '''
    
    Nr, Nt, Ns = data.shape
    
    # get the next power of 2 greater than or equal to 2 * Nt
    # for efficient circular convolution via FFT
    N = nextPow2(2 * Nt)
    
    if Nr == Ns:
        # optimize the computation of Matrix * X2 and Matrix.T * X1 using a single for loop
        def MatVec(x):        
            
            # x1 is the first Nt * Nr elements of the vector x
            x1 = x[:(Nt * Nr)]
            
            # x2 is the last Nt * Ns elements of the vector x
            x2 = x[-(Nt * Ns):]
            
            # reshape x1 and x2 into matrices X1 and X2
            # X1 multiplies Matrix.T (time reversal), so time reverse X1
            X1 = np.flipud(np.reshape(x1, (Nt, Nr), order='F'))
            X2 = np.reshape(x2, (Nt, Ns), order='F')
            
            Y1 = np.zeros((Nt, Nr))
            Y2 = np.zeros((Nt, Ns))
            
            for i in range(Nr):
                # Compute the matrix-vector product for Matrix * X2
                U = data[i, :, :]
                # Circular convolution: pad time axis (axis=0) with zeros to length N
                circularConvolution = irfft(rfft(U, n=N, axis=0) * rfft(X2, n=N, axis=0), axis=0)
                convolutionMatrix = circularConvolution[:Nt, :]
                Y1[:, i] = np.sum(convolutionMatrix, axis=1) # sum over sources
                
                # Compute the matrix-vector product for Matrix.T * X1
                UT = data[:, :, i].T
                # Circular convolution: pad time axis (axis=0) with zeros to length N
                circularConvolutionT = irfft(rfft(UT, n=N, axis=0) * rfft(X1, n=N, axis=0), axis=0)
                convolutionMatrixT = np.flipud(circularConvolutionT[:Nt, :])
                Y2[:, i] = np.sum(convolutionMatrixT, axis=1) # sum over receivers
                
            y1 = np.reshape(Y1, (Nt * Nr, 1), order='F')
            y2 = np.reshape(Y2, (Nt * Ns, 1), order='F')
            
            return np.concatenate((y1, y2))
            
    else:   # if Nr != Ns
        # compute Matrix * X2 and Matrix.T * X1 over separate for loops of different ranges
        def MatVec(x):        
            
            # x1 is the first Nt * Nr elements of the vector x
            x1 = x[:(Nt * Nr)]
            
            # x2 is the last Nt * Ns elements of the vector x
            x2 = x[-(Nt * Ns):]
            
            # reshape x1 and x2 into matrices X1 and X2
            # X1 multiplies Matrix.T (time reversal), so flip up-down
            X1 = np.flipud(np.reshape(x1, (Nt, Nr), order='F'))
            X2 = np.reshape(x2, (Nt, Ns), order='F')
            
            Y1 = np.zeros((Nt, Nr))
            Y2 = np.zeros((Nt, Ns))
        
            for i in range(Nr):
                # Compute the matrix-vector product for Matrix * X2
                U = data[i, :, :]
                # Circular convolution: pad time axis (axis=0) with zeros to length N
                circularConvolution = irfft(rfft(U, n=N, axis=0) * rfft(X2, n=N, axis=0), axis=0)
                convolutionMatrix = circularConvolution[:Nt, :]
                Y1[:, i] = np.sum(convolutionMatrix, axis=1) # sum over sources
            
            for j in range(Ns):
                # Compute the matrix-vector product for Matrix.T * X1
                UT = data[:, :, j].T
                # Circular convolution: pad time axis (axis=0) with zeros to length N
                circularConvolutionT = irfft(rfft(UT, n=N, axis=0) * rfft(X1, n=N, axis=0), axis=0)
                convolutionMatrixT = np.flipud(circularConvolutionT[:Nt, :])
                Y2[:, j] = np.sum(convolutionMatrixT, axis=1) # sum over receivers
                
            y1 = np.reshape(Y1, (Nt * Nr, 1), order='F')
            y2 = np.reshape(Y2, (Nt * Ns, 1), order='F')
            
            return np.concatenate((y1, y2))
    
    return LinearOperator(shape=(Nt * (Nr + Ns), Nt * (Nr + Ns)), matvec=MatVec)

#==============================================================================

def asConvolutionOperator(data):
    '''
    This function takes the data and defines a matrix-vector product for the 
    either the near-field operator (NFO) or the Lippmann-Schwinger operator (LSO).
    (This function is not intended to be used to compute a singular-value
    decomposition -- see the definition for 'asSymmetricConvolutionOperator')
    
    Input: a three-dimensional data array of shape Nr x Nt x Ns
    
    MatVec: Definition of a matrix-vector product
    
    x: an arbitrary input vector
    Nt: number of time samples
    Nr: number of receivers
    Ns: number of sources
    
    Matrix: discretized near-field operator
    
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
    
    return LinearOperator(shape=(Nt * Nr, Nt * Ns), matvec=MatVec)