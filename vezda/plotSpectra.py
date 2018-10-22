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
from vezda.plot_utils import setFigure, default_params
from vezda.signal_utils import compute_spectrum
import matplotlib.pyplot as plt
from pathlib import Path
import pickle

def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--power', action='store_true',
                        help='''Plot the mean power spectrum of the data. Default is to plot the
                        mean amplitude spectrum of the Fourier transform.''')
    parser.add_argument('--fmin', type=float, default=0,
                        help='Specify the minimum frequency of the power spectrum plot. Default is set to 0.')
    parser.add_argument('--fmax', type=float,
                        help='''Specify the maximum frequency of the power spectrum plot. Default is set to the
                        maximum frequency bin based on the length of the time signal.''')
    parser.add_argument('--format', '-f', type=str, default='pdf', choices=['png', 'pdf', 'ps', 'eps', 'svg'],
                        help='''specify the image format of the saved file. Accepted formats are png, pdf,
                        ps, eps, and svg. Default format is set to pdf.''')
    parser.add_argument('--mode', type=str, choices=['light', 'dark'], required=False,
                        help='''Specify whether to view plots in light mode for daytime viewing
                        or dark mode for nighttime viewing.
                        Mode must be either \'light\' or \'dark\'.''')
    args = parser.parse_args()
    
    #==============================================================================        
    # Load the receiver coordinates and recording times from the data directory
    datadir = np.load('datadir.npz')
    recordingTimes = np.load(str(datadir['recordingTimes']))
    receiverPoints = np.load(str(datadir['receivers']))
    if 'sources' in datadir:
        sourcePoints = np.load(str(datadir['sources']))
    else:
        sourcePoints = None
    if Path('noisyData.npy').exists():
        userResponded = False
        print(textwrap.dedent(
              '''
              Detected that band-limited noise has been added to the data array.
              Would you like to plot the power spectrum of the noisy data? ([y]/n)
              
              Enter 'q/quit' exit the program.
              '''))
        while userResponded == False:
            answer = input('Action: ')
            if answer == '' or answer == 'y' or answer == 'yes':
                print('Proceeding with noisy data...')
                # read in the noisy data array
                recordedData = np.load('noisyData.npy')
                userResponded = True
            elif answer == 'n' or answer == 'no':
                print('Proceeding with noise-free data...')
                # read in the recorded data array
                recordedData  = np.load(str(datadir['recordedData']))
                userResponded = True
            elif answer == 'q' or answer == 'quit':
                sys.exit('Exiting program.\n')
            else:
                print('Invalid response. Please enter \'y/yes\', \'n\no\', or \'q/quit\'.')
                
    else:
        # read in the recorded data array
        recordedData  = np.load(str(datadir['recordedData']))
    
    # Compute length of time step.
    dt = recordingTimes[1] - recordingTimes[0]
    
    # Load the windowing parameters for the receiver and time axes of
    # the 3D data array
    if Path('window.npz').exists():
        windowDict = np.load('window.npz')
        
        # Receiver window parameters
        rstart = windowDict['rstart']
        rstop = windowDict['rstop']
        rstep = windowDict['rstep']
        
        # Source window parameters
        sstart = windowDict['sstart']
        sstop = windowDict['sstop']
        sstep = windowDict['sstep']
        
    else:
        # Set default window parameters if user did
        # not specify window parameters.
        
        # Receiver window parameters
        rstart = 0
        rstop = receiverPoints.shape[0]
        rstep = 1
        
        # Source window parameters
        sstart = 0
        sstop = recordedData.shape[2]
        sstep = 1
    
    # Slice the receiverPoints array according to the receiver window parametes
    rinterval = np.arange(rstart, rstop, rstep)
    receiverPoints = receiverPoints[rinterval, :]
    
    sinterval = np.arange(sstart, sstop, sstep)
    if sourcePoints is not None:
        # Slice the sourcePoints array according to the source window parametes
        sourcePoints = sourcePoints[sinterval, :]

    recordedData = recordedData[rinterval, :, :]
    recordedData = recordedData[:, :, sinterval]
    
    if Path('plotParams.pkl').exists():
        plotParams = pickle.load(open('plotParams.pkl', 'rb'))
        
    else:
        plotParams = default_params()
            
    if args.mode is not None:
        plotParams['view_mode'] = args.mode
        pickle.dump(plotParams, open('plotParams.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)
        
    freqs, amplitudes = compute_spectrum(recordedData, dt, args.power)
    if args.power:
        title = 'Mean Power Spectrum'
        ylabel = 'Power'
    else:
        title = 'Mean Amplitude Spectrum'
        ylabel = 'Amplitude'
        
    
    if args.fmax is not None and args.fmax <= args.fmin:
        sys.exit(textwrap.dedent(
            '''
            RelationError: The maximum frequency of the power spectrum plot must
            be greater than the mininum frequency.
            '''))
    elif args.fmin < 0:
        sys.exit(textwrap.dedent(
                '''
                ValueError: The minimum frequency of the power spectrum plot must be nonnegative.
                '''))
    fmin = args.fmin
    fmax = args.fmax
    
    fig, ax = setFigure(num_axes=1, mode=plotParams['view_mode'])
    ax.plot(freqs, amplitudes, color=ax.linecolor, linewidth=ax.linewidth)
    ax.set_title(title, color=ax.titlecolor)
    ax.set_xlabel('Frequency (Hz)', color=ax.labelcolor)
    ax.set_ylabel(ylabel, color=ax.labelcolor)
    ax.set_xlim([fmin, fmax])
    ax.set_ylim(bottom=0)
    ax.fill_between(freqs, 0, amplitudes, where=(amplitudes > 0), color='m', alpha=ax.alpha)
    ax.locator_params(axis='y', nticks=6)
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.tight_layout()
    fig.savefig('FourierSpectrum.' + args.format, format=args.format, bbox_inches='tight')
    plt.show()