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
import argparse
import textwrap
import vezda.TELSM

def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--regPar', '--alpha', type=float,
                        help='Specify the value of the regularization parameter')
    parser.add_argument('--medium', type=str, default='constant', choices=['constant', 'variable'],
                        help='''Specify whether the background medium is constant or variable
                        (inhomogeneous). If argument is set to 'constant', the velocity defined in
                        the required 'pulseFun.py' file is used. Default is set to 'constant'.''')
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
    elif args.regPar is None:
        alpha = 0
        
    vezda.TELSM.solver(args.medium, alpha)