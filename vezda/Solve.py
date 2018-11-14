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

import sys
import numpy as np
import argparse
import textwrap
import vezda.NFE_Solver
import vezda.LSE_Solver

def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nfe', action='store_true',
                        help='Solve the near-field equation (NFE).')
    parser.add_argument('--lse', action='store_true',
                        help='Solve the Lippmann-Schwinger equation (LSE).')
    parser.add_argument('--regPar', '--alpha', type=float,
                        help='Specify the value of the regularization parameter')
    parser.add_argument('--medium', type=str, default='constant', choices=['constant', 'variable'],
                        help='''Specify whether the background medium is constant or variable
                        (inhomogeneous). If argument is set to 'constant', the velocity defined in
                        the required 'pulsesFun.py' file is used. Default is set to 'constant'.''')
    args = parser.parse_args()
    
    if args.regPar is not None and args.regPar >= 0:
        # alpha is the value of the regularization parameter
        alpha = args.regPar
    elif args.regPar is not None and args.regPar < 0:
        sys.exit(textwrap.dedent(
                '''
                Error: Optional argument '--regPar/--alpha' cannot be negative. 
                The regularization parameter must be greater than or equal to zero.
                '''))
    else:
        # if args.regPar is None
        alpha = 0
        
    if args.nfe:
        # Solve the near-field equation
        # Read in the computed singular-value decomposition of the near-field operator
        # Singular values are stored in vector s
        # Left vectors are columns of U
        # Right vectors are columns of V
        
        SVD = np.load('NFO_SVD.npz')
        s = SVD['s']
        Uh = SVD['Uh']
        V = SVD['V']
        domain = SVD['domain']
        
        vezda.NFE_Solver.solver(args.medium, s, Uh, V, alpha, domain)
    
    elif args.lse:
        # Solve the Lippmann-Schwinger equation
        # Read in the computed singular-value decomposition of the Lippmann-Schwinger operator
        # Singular values are stored in vector s
        # Left vectors are columns of U
        # Right vectors are columns of V
        
        SVD = np.load('LSO_SVD.npz')
        s = SVD['s']
        Uh = SVD['Uh']
        V = SVD['V']
        domain = SVD['domain']
        
        vezda.LSE_Solver.solver(s, Uh, V, alpha, domain)
        
    else:
        sys.exit(textwrap.dedent(
                '''
                Please specify which linear sampling equation you would like to solve. Enter:
                    
                    vzsolve --nfe
                
                to solve the near-field equation or
                    
                    vzsolve --lse
                    
                to solve the Lippmann-Schwinger equation.
                '''))