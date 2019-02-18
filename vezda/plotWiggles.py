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
import pickle
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from vezda.data_utils import get_user_windows, load_data, load_test_funcs
from vezda.plot_utils import (default_params, remove_keymap_conflicts, process_key_waves,
                              wave_title, plotWiggles, plotMap, setFigure)
from vezda.plot_utils import FontColor

def info():
    commandName = FontColor.BOLD + 'vzwiggles:' + FontColor.END
    description = ' plot time signals as waveforms'
    
    return commandName + description

def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', action='store_true',
                        help='Plot the recorded data. (Default)')
    parser.add_argument('--testfunc', action='store_true',
                        help='Plot the simulated test functions.')
    parser.add_argument('--tu', type=str,
                        help='Specify the time units (e.g., \'s\' or \'ms\').')
    parser.add_argument('--au', type=str,
                        help='Specify the amplitude units (e.g., \'m\' or \'mm\').')
    parser.add_argument('--pclip', type=float,
                        help='''Specify the percentage (0-1) of the peak amplitude to display. This
                        parameter is used for pcolormesh plots only. Default is set to 1.''')
    parser.add_argument('--title', type=str,
                        help='''Specify a title for the wiggle plot. Default title is
                        \'Data\' if \'--data\' is passed and 'Test Function' if \'--testfunc\'
                        is passed.''')
    parser.add_argument('--format', '-f', type=str, default='pdf', choices=['png', 'pdf', 'ps', 'eps', 'svg'],
                        help='''Specify the image format of the saved file. Accepted formats are png, pdf,
                        ps, eps, and svg. Default format is set to pdf.''')
    parser.add_argument('--map', action='store_true',
                        help='''Plot a map of the receiver and source/sampling point locations. The current
                        source/sampling point will be highlighted. The boundary of the scatterer will also
                        be shown if available.''')
    parser.add_argument('--mode', type=str, choices=['light', 'dark'], required=False,
                        help='''Specify whether to view plots in light mode for daytime viewing
                        or dark mode for nighttime viewing.
                        Mode must be either \'light\' or \'dark\'.''')
    
    args = parser.parse_args()
    #==============================================================================
    # if a plotParams.pkl file already exists, load relevant parameters
    if Path('plotParams.pkl').exists():
        plotParams = pickle.load(open('plotParams.pkl', 'rb'))
        
        # update parameters for wiggle plots based on passed arguments
        if args.mode is not None:
            plotParams['view_mode'] = args.mode
        
        if args.tu is not None:
            plotParams['tu'] = args.tu
        
        if args.au is not None:
            plotParams['au'] = args.au
            
        if args.pclip is not None:
            if args.pclip >= 0 and args.pclip <= 1:
                plotParams['pclip'] = args.pclip
            else:
                print(textwrap.dedent(
                      '''
                      Warning: Invalid value passed to argument \'--pclip\'. Value must be
                      between 0 and 1.
                      '''))
            
        if args.title is not None:
            if args.data:
                plotParams['data_title'] = args.title
            elif args.testfunc:
                plotParams['tf_title'] = args.title
    
    else: # create a plotParams dictionary file with default values
        plotParams = default_params()
        
        # update parameters for wiggle plots based on passed arguments
        if args.mode is not None:
            plotParams['view_mode'] = args.mode
        
        if args.tu is not None:
            plotParams['tu'] = args.tu
        
        if args.au is not None:
            plotParams['au'] = args.au
        
        if args.title is not None:
            if args.data:
                plotParams['data_title'] = args.title
            elif args.testfunc:
                plotParams['tf_title'] = args.title
    
    pickle.dump(plotParams, open('plotParams.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)

    #==============================================================================
    # Load the relevant data to plot
    datadir = np.load('datadir.npz')
    receiverPoints = np.load(str(datadir['receivers']))
    time = np.load(str(datadir['recordingTimes']))
    
    # Apply any user-specified windows
    rinterval, tinterval, tstep, dt, sinterval = get_user_windows()
    receiverPoints = receiverPoints[rinterval, :]
    time = time[tinterval]
    
    # Load the scatterer boundary, if it exists
    if 'scatterer' in datadir:
        scatterer = np.load(str(datadir['scatterer']))
    else:
        scatterer = None    
    
    if all(v is True for v in [args.data, args.testfunc]):
        # User specified both data and testfuncs for plotting
        # Send error message and exit.
        sys.exit(textwrap.dedent(
                '''
                Error: Cannot plot both recorded data and simulated test functions. Use
                
                    vzwiggles --data
                    
                to plot the recorded data or
                
                    vzwiggles --testfuncs
                    
                to plot the simulated test functions.
                '''))
    
    elif all(v is not True for v in [args.data, args.testfunc]):
        # User did not specify which wiggles to plot.
        # Plot recorded data by default.
        # load the 3D data array into variable 'X'
        # X[receiver, time, source]
        wiggleType = 'data'
        X = load_data(domain='time', verbose=True)
        
        if 'sources' in datadir:
            sourcePoints = np.load(str(datadir['sources']))
            sourcePoints = sourcePoints[sinterval, :]
        else:
            sourcePoints = None
    
    elif args.data:
        # load the 3D data array into variable 'X'
        # X[receiver, time, source]
        wiggleType = 'data'
        X = load_data(domain='time', verbose=True)
        
        if 'sources' in datadir:
            sourcePoints = np.load(str(datadir['sources']))
            sourcePoints = sourcePoints[sinterval, :]
        else:
            sourcePoints = None
        
    elif args.testfunc:
        wiggleType = 'testfunc'
        
        # Update time to convolution times
        T = time[-1] - time[0]
        time = np.linspace(-T, T, 2 * len(time) - 1)
        
        if 'testFuncs' not in datadir and not Path('VZTestFuncs.npz').exists():
            X, sourcePoints = load_test_funcs(domain='time', medium='constant',
                                              verbose=True, return_sampling_points=True)
            
        if 'testFuncs' in datadir and not Path('VZTestFuncs.npz').exists():
            X, sourcePoints = load_test_funcs(domain='time', medium='variable',
                                              verbose=True, return_sampling_points=True)
            
            # Pad time axis with zeros to length of convolution 2Nt-1
            npad = ((0, 0), (X.shape[1] - 1, 0), (0, 0))
            X = np.pad(X, pad_width=npad, mode='constant', constant_values=0)
            
        elif not 'testFuncs' in datadir and Path('VZTestFuncs.npz').exists():
            X, sourcePoints = load_test_funcs(domain='time', medium='constant',
                                              verbose=True, return_sampling_points=True)
                    
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
                    X, sourcePoints = load_test_funcs(domain='time', medium='variable',
                                                      verbose=True, return_sampling_points=True)
            
                    # Pad time axis with zeros to length of convolution 2Nt-1
                    npad = ((0, 0), (X.shape[1] - 1, 0), (0, 0))
                    X = np.pad(X, pad_width=npad, mode='constant', constant_values=0)
                    
                    userResponded = True
                    break
                
                elif answer == '2':
                    X, sourcePoints = load_test_funcs(domain='time', medium='constant',
                                                      verbose=True, return_sampling_points=True)
                    
                    userResponded = True
                    break
                
                elif answer == 'q' or answer == 'quit':
                    sys.exit('Exiting program.')
                
                else:
                    print('Invalid response. Please enter \'1\', \'2\', or \'q/quit\'.')
            
    
    #==============================================================================        
    # increment source/recording interval and receiver interval to be consistent
    # with one-based indexing (i.e., count from one instead of zero)
    sinterval += 1
    rinterval += 1
    
    Ns = X.shape[2]
    
    remove_keymap_conflicts({'left', 'right', 'up', 'down', 'save'})
    if args.map:
        fig, ax1, ax2 = setFigure(num_axes=2, mode=plotParams['view_mode'],
                                  ax2_dim=receiverPoints.shape[1])
            
        ax1.volume = X
        ax1.index = Ns // 2
        title = wave_title(ax1.index, sinterval, sourcePoints, wiggleType, plotParams)
        plotWiggles(ax1, X[:, :, ax1.index], time, rinterval, receiverPoints, title, wiggleType, plotParams)
        
        ax2.index = ax1.index
        plotMap(ax2, ax2.index, receiverPoints, sourcePoints, scatterer, wiggleType, plotParams)
        plt.tight_layout()
        fig.canvas.mpl_connect('key_press_event', lambda event: process_key_waves(event, time, rinterval, sinterval,
                                                                                  receiverPoints, sourcePoints, Ns, scatterer,
                                                                                  args.map, wiggleType, plotParams))
    
    else:
        fig, ax = setFigure(num_axes=1, mode=plotParams['view_mode'])
            
        ax.volume = X
        ax.index = Ns // 2
        title = wave_title(ax.index, sinterval, sourcePoints, wiggleType, plotParams)
        plotWiggles(ax, X[:, :, ax.index], time, rinterval, receiverPoints, title, wiggleType, plotParams)
        plt.tight_layout()
        fig.canvas.mpl_connect('key_press_event', lambda event: process_key_waves(event, time, rinterval, sinterval,
                                                                                  receiverPoints, sourcePoints, Ns, scatterer,
                                                                                  args.map, wiggleType, plotParams))
    
    plt.show()
