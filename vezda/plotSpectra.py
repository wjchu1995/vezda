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
from vezda.plot_utils import setFigure, default_params
from vezda.data_utils import load_data, load_test_funcs, get_user_windows
from vezda.signal_utils import compute_spectrum
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
from vezda.plot_utils import FontColor

def info():
    commandName = FontColor.BOLD + 'vzspectra:' + FontColor.END
    description = ' plot the amplitude or power spectrum of time signals in the frequency domain'
    
    return commandName + description

def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', action='store_true',
                        help='Plot the frequency spectrum of the recorded data. (Default)')
    parser.add_argument('--testfunc', action='store_true',
                        help='Plot the frequency spectrum of the simulated test functions.')
    parser.add_argument('--power', action='store_true',
                        help='''Plot the mean power spectrum of the input signals. Default is to plot the
                        mean amplitude spectrum of the Fourier transform.''')
    parser.add_argument('--fmin', type=float,
                        help='Specify the minimum frequency of the amplitude/power spectrum plot. Default is set to 0.')
    parser.add_argument('--fmax', type=float,
                        help='''Specify the maximum frequency of the amplitude/power spectrum plot. Default is set to the
                        maximum frequency bin based on the length of the time signal.''')
    parser.add_argument('--fu', type=str,
                        help='Specify the frequency units (e.g., Hz)')
    parser.add_argument('--format', '-f', type=str, default='pdf', choices=['png', 'pdf', 'ps', 'eps', 'svg'],
                        help='''specify the image format of the saved file. Accepted formats are png, pdf,
                        ps, eps, and svg. Default format is set to pdf.''')
    parser.add_argument('--mode', type=str, choices=['light', 'dark'], required=False,
                        help='''Specify whether to view plots in light mode for daytime viewing
                        or dark mode for nighttime viewing.
                        Mode must be either \'light\' or \'dark\'.''')
    args = parser.parse_args()
    
    #==============================================================================        
    # Get time window parameters
    tinterval, tstep, dt = get_user_windows()[1:4]
    datadir = np.load('datadir.npz')
    recordingTimes = np.load(str(datadir['recordingTimes']))
    recordingTimes = recordingTimes[tinterval]
    
    # Used for getting time and frequency units
    if Path('plotParams.pkl').exists():
        plotParams = pickle.load(open('plotParams.pkl', 'rb'))
    else:
        plotParams = default_params()
    
    if all(v is True for v in [args.data, args.testfunc]):
        sys.exit(textwrap.dedent(
                '''
                Error: Cannot plot frequency spectrum of both recorded data and
                simulated test functions. Use
                
                    vzspectra --data
                    
                to plot the frequency spectrum of the recorded data or
                
                    vzspectra --testfuncs
                    
                to plot the frequency spectrum of the simulated test functions.
                '''))
    
    elif (args.data and not args.testfunc) or all(v is not True for v in [args.data, args.testfunc]):
        # default is to plot spectra of data if user does not specify either args.data or args.testfunc
        X = load_data(domain='time', verbose=True)
        
    elif not args.data and args.testfunc:
        if 'testFuncs' not in datadir and not Path('VZTestFuncs.npz').exists():
            X = load_test_funcs(domain='time', medium='constant', verbose=True)
        
        elif 'testFuncs' in datadir and not Path('VZTestFuncs.npz').exists():
            X = load_test_funcs(domain='time', medium='variable', verbose=True)
            
        elif not 'testFuncs' in datadir and Path('VZTestFuncs.npz').exists():
            X = load_test_funcs(domain='time', medium='constant', verbose=True)
                    
        elif 'testFuncs' in datadir and Path('VZTestFuncs.npz').exists():
            userResponded = False
            print(textwrap.dedent(
                 '''
                 Two files are available containing simulated test functions.
                 
                 Enter '1' to view the user-provided test functions. (Default)
                 Enter '2' to view the test functions computed by Vezda.
                 Enter 'q/quit' to exit.
                 '''))
            while userResponded == False:
                answer = input('Action: ')
                
                if answer == '' or answer == '1':
                    X = load_test_funcs(domain='time', medium='variable', verbose=True)
                    userResponded = True
                    break
                
                elif answer == '2':
                    X = load_test_funcs(domain='time', medium='constant', verbose=True)
                    userResponded = True
                    break
                
                elif answer == 'q' or answer == 'quit':
                    sys.exit('Exiting program.')
                
                else:
                    print('Invalid response. Please enter \'1\', \'2\', or \'q/quit\'.')
        
    #==============================================================================
    # compute spectra
    freqs, amplitudes = compute_spectrum(X, tstep * dt, args.power)
        
    if args.power:
        plotLabel = 'power'
        plotParams['freq_title'] = 'Mean Power Spectrum'
        plotParams['freq_ylabel'] = 'Power'
    else:
        plotLabel = 'amplitude'
        plotParams['freq_title'] = 'Mean Amplitude Spectrum'
        plotParams['freq_ylabel'] = 'Amplitude'
            
    if args.data or all(v is not True for v in [args.data, args.testfunc]):
        plotParams['freq_title'] += ' [' + plotParams['data_title'] + ']'
    elif args.testfunc:
        plotParams['freq_title'] += ' [' + plotParams['tf_title'] + 's]'
        
    if args.fmin is not None: 
        if args.fmin >= 0:
            if args.fmax is not None:
                if args.fmax > args.fmin:
                    plotParams['fmin'] = args.fmin
                    plotParams['fmax'] = args.fmax
                else:
                    sys.exit(textwrap.dedent(
                            '''
                            RelationError: The maximum frequency of the %s spectrum plot must
                            be greater than the mininum frequency.
                            ''' %(plotLabel)))   
            else:
                fmax = plotParams['fmax']
                if fmax > args.fmin:
                    plotParams['fmin'] = args.fmin
                else:
                    sys.exit(textwrap.dedent(
                            '''
                            RelationError: The specified minimum frequency of the %s spectrum 
                            plot must be less than the maximum frequency.
                            ''' %(plotLabel)))                                        
        else:
            sys.exit(textwrap.dedent(
                    '''
                    ValueError: The specified minimum frequency of the %s spectrum 
                    plot must be nonnegative.
                    ''' %(plotLabel)))
            
    #===============================================================================
    if args.fmax is not None:
        if args.fmin is not None:
            if args.fmin >= 0:
                if args.fmax > args.fmin:
                    plotParams['fmin'] = args.fmin
                    plotParams['fmax'] = args.fmax
                else:
                    sys.exit(textwrap.dedent(
                            '''
                            RelationError: The maximum frequency of the %s spectrum plot must
                            be greater than the mininum frequency.
                            ''' %(plotLabel)))
            else:
                sys.exit(textwrap.dedent(
                        '''
                        ValueError: The specified minimum frequency of the %s spectrum 
                        plot must be nonnegative.
                        ''' %(plotLabel)))
        else:
            fmin = plotParams['fmin']
            if args.fmax > fmin:
                plotParams['fmax'] = args.fmax
            else:
                sys.exit(textwrap.dedent(
                        '''
                        RelationError: The specified maximum frequency of the %s spectrum 
                        plot must be greater than the minimum frequency.
                        ''' %(plotLabel)))
    elif plotParams['fmax'] is None:
        plotParams['fmax'] = np.max(freqs)
                
    #===================================================================================
    if args.fu is not None:
        plotParams['fu'] = args.fu
            
    if args.mode is not None:
        plotParams['view_mode'] = args.mode
    
    pickle.dump(plotParams, open('plotParams.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)
    
    fig, ax = setFigure(num_axes=1, mode=plotParams['view_mode'])
    ax.plot(freqs, amplitudes, color=ax.linecolor, linewidth=ax.linewidth)
    ax.set_title(plotParams['freq_title'], color=ax.titlecolor)
    
    # get frequency units from plotParams
    fu = plotParams['fu']
    fmin = plotParams['fmin']
    fmax = plotParams['fmax']
    if fu != '':
        ax.set_xlabel('Frequency (%s)' %(fu), color=ax.labelcolor)
    else:
        ax.set_xlabel('Frequency', color=ax.labelcolor)
    ax.set_ylabel(plotParams['freq_ylabel'], color=ax.labelcolor)
    ax.set_xlim([fmin, fmax])
    ax.set_ylim(bottom=0)
    ax.fill_between(freqs, 0, amplitudes, where=(amplitudes > 0), color='m', alpha=ax.alpha)
    ax.locator_params(axis='y', nticks=6)
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.tight_layout()
    fig.savefig(plotLabel + 'Spectrum.' + args.format, format=args.format, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.show()
    