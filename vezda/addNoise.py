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

import sys
import argparse
import textwrap
import numpy as np
import pickle
from pathlib import Path
from vezda.signal_utils import add_noise
from vezda.plot_utils import default_params
from vezda.plot_utils import FontColor

def info():
    commandName = FontColor.BOLD + 'vznoise:' + FontColor.END
    description = ' add band-limited white noise to the recorded data'
    
    return commandName + description

def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fmin', type=float,
                        help='Specify the minimum frequency component of the noise.')
    parser.add_argument('--fmax', type=float,
                        help='Specify the maximum frequency component of the noise.')
    parser.add_argument('--snr', type=float,
                        help='''Specify the desired signal-to-noise ratio. Must be a positive
                                real number.''')
    args = parser.parse_args()
    
    #==============================================================================
    try:
        Dict = np.load('noisyData.npz')
        fmin = Dict['fmin']
        fmax = Dict['fmax']
        snr = Dict['snr']
    except FileNotFoundError:
        fmin, fmax, snr = None, None, None
        
    # Used for getting frequency units
    if Path('plotParams.pkl').exists():
        plotParams = pickle.load(open('plotParams.pkl', 'rb'))
    else:
        plotParams = default_params()
    
    if all(v is None for v in [args.fmin, args.fmax, args.snr]):
        # if no arguments are passed
        
        if all(v is None for v in [fmin, fmax, snr]):
            # and no parameters have been assigned values
            sys.exit(textwrap.dedent(
                    '''
                    No noise has been added to the data.
                    '''))
        else:
            # print fmin, fmax, snr and exit
            fu = plotParams['fu']
            if fu != '':
                sys.exit(textwrap.dedent(
                        '''
                        Band-limited white noise has already been added to the data:
                            
                        Minimum frequency: {:0.2f} {}
                        Maximum frequency: {:0.2f} {}
                        Signal-to-noise ratio: {:0.2f}
                        '''.format(fmin, fu, fmax, fu, snr)))
            else:
                sys.exit(textwrap.dedent(
                        '''
                        Band-limited white noise has already been added to the data:
                            
                        Minimum frequency: {:0.2f}
                        Maximum frequency: {:0.2f}
                        Signal-to-noise ratio: {:0.2f}
                        '''.format(fmin, fmax, snr)))
                    
    elif all(v is not None for v in [args.fmin, args.fmax, args.snr]):
        # if all arguments were passed
        
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
        
        fmin = args.fmin
        fmax = args.fmax
        snr = args.snr
        fu = plotParams['fu']
        if fu != '':
            print(textwrap.dedent(
                  '''
                  Adding band-limited white noise:
          
                  Minimum frequency: {:0.2f} {}
                  Maximum frequency: {:0.2f} {}
                  Signal-to-noise ratio: {:0.2f}
                  '''.format(fmin, fu, fmax, fu, snr)))
        else:
            print(textwrap.dedent(
                  '''
                  Adding band-limited white noise:
          
                  Minimum frequency: {:0.2f}
                  Maximum frequency: {:0.2f}
                  Signal-to-noise ratio: {:0.2f}
                  '''.format(fmin, fmax, snr)))
        
        # Load the 3D data array and recording times from data directory
        datadir = np.load('datadir.npz')
        recordedData = np.load(str(datadir['recordedData']))
        recordingTimes = np.load(str(datadir['recordingTimes']))
        dt = recordingTimes[1] - recordingTimes[0]
    
        noisyData = add_noise(recordedData, dt, fmin, fmax, snr)
        np.savez('noisyData.npz', noisyData=noisyData, fmin=fmin, fmax=fmax, snr=snr)
        
    else:
        sys.exit(textwrap.dedent(
                '''
                Error: All command-line arguments \'--fmin\', \'--fmax\', and \'--snr\' must
                be used when parameterizing the noise.
                '''))