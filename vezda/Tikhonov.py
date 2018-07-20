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
from scipy.linalg import diagsvd

def Tikhonov(U, s, V, b, alpha):
    '''
    Tikhonov: Tikhonov procedure for solving Ax = b via SVD.
    
    Inputs: 
       U, s, V = svd(A) is the singular-value decomposition of matrix A,
            i.e., A = U * S * V.T,  s = diag(S)
       b: right-hand side of the linear system
       alpha: Tikhonov regularizaton parameter
    
    Output:
       x_alpha: the regularized solution
    '''
   
    # Construct the pseudoinverse 'Sp' of the diagonal matrix 'S'
    sigma = np.divide(s, alpha + s**2)
    Sp = diagsvd(sigma, V.shape[1], U.shape[1])
    
    # Return the Tikhonov-regularized solution
    return np.dot(V, np.dot(Sp, np.dot(U.T, b)))