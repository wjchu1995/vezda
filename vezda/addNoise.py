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
import numpy as np
from vezda.signal_utils import add_noise

def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fmin', type=float, required=True,
                        help='Specify the minimum frequency component of the noise.')
    parser.add_argument('--fmax', type=float, required=True,
                        help='Specify the maximum frequency component of the noise.')
    parser.add_argument('--snr', type=float, default=2.0, required=False,
                        help='''Specify the desired signal-to-noise ratio. Must be a positive
                                real number.''')
    args = parser.parse_args()
    
    #==============================================================================
    if args.fmax < args.fmin:
        sys.exit(textwrap.dedent(
                '''
                RelationError: The maximum frequency component of the nosie must be greater
                than or equal to the mininum frequency component.
                '''))
    elif args.fmin <= 0:
        sys.exit(textwrap.dedent(
                '''
                ValueError: The minimum frequency component of the noise must be strictly positive.
                '''))
    elif args.snr <= 0:
        sys.exit(textwrap.dedent(
                '''
                ValueError: The signal-to-noise ratio (SNR) must be strictly positive.
                '''))
        
    print(textwrap.dedent(
          '''
          Adding band-limited white noise:
          
          Minimum frequency: {:0.2f} Hz
          Maximum frequency: {:0.2f} Hz
          Signal-to-noise ratio: {:0.2f}
          '''.format(args.fmin, args.fmax, args.snr)))
    
    # Load the 3D data array and recording times from data directory
    datadir = np.load('datadir.npz')
    recordedData = np.load(str(datadir['recordedData']))
    recordingTimes = np.load(str(datadir['recordingTimes']))
    dt = recordingTimes[1] - recordingTimes[0]
    
    noisyData = add_noise(recordedData, dt, args.fmin, args.fmax, args.snr)
    np.save('noisyData.npy', noisyData)