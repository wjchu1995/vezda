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

def Morozov(U, s, b, delta, alpha):
    #Morozov  generalized Morozov discrepancy functional for choosing the 
    #         regularization parameter.
    #
    # F(alpha) := || A * x_alpha - b ||^2 - delta^2 * ||x_alpha||^2 
    # where 'x_alpha' is the Tikhonov solution with regularization
    # parameter 'alpha'.
    #
    # input: 
    #    U, S, V = svd(A) is the singular-value decomposition of matrix A,
    #        i.e., A = U @ S @ V.T,   s = diag(S)
    #    b: right-hand side of Ax = b
    #    delta: noise level (a positive constant)
    #    alpha: the regularization parameter
    #
    # output:
    #    value: the value of the functional 
    
    value = 0
    for n in range(len(s)):
        value += (alpha**2 - delta**2 * s[n]**2) / (s[n]**2 + alpha)**2 * np.abs(U[:,n].T @ b)**2    
    
    return value
