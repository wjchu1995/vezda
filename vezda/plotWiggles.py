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

import os
import sys
import argparse
import textwrap
import pickle
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from vezda.plot_utils import (default_params, remove_keymap_conflicts, process_key_waves,
                              wave_title, plotWiggles, plotMap, setFigure)
from vezda.sampling_utils import samplingIsCurrent, sampleSpace

sys.path.append(os.getcwd())
import pulseFun
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
    recordingTimes = np.load(str(datadir['recordingTimes']))
    dt = recordingTimes[1] - recordingTimes[0]
    
    if 'scatterer' in datadir:
        scatterer = np.load(str(datadir['scatterer']))
    else:
        scatterer = None
    
    if Path('window.npz').exists():
        windowDict = np.load('window.npz')
        
        # Apply the receiver window
        rstart = windowDict['rstart']
        rstop = windowDict['rstop']
        rstep = windowDict['rstep']
        
        # Apply the time window
        tstart = windowDict['tstart']
        tstop = windowDict['tstop']
        tstep = windowDict['tstep']
        
        # Convert time window parameters to corresponding array indices
        Tstart = int(round(tstart / dt))
        Tstop = int(round(tstop / dt))
    
    else:
        rstart = 0
        rstop = receiverPoints.shape[0]
        rstep = 1
        
        tstart = recordingTimes[0]
        tstop = recordingTimes[-1]
        
        Tstart = 0
        Tstop = len(recordingTimes)
        tstep = 1
    
    rinterval = np.arange(rstart, rstop, rstep)
    receiverPoints = receiverPoints[rinterval, :]
    
    tinterval = np.arange(Tstart, Tstop, tstep)
        
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
        if Path('noisyData.npz').exists():
            userResponded = False
            print(textwrap.dedent(
                  '''
                  Detected that band-limited noise has been added to the data array.
                  Would you like to plot the noisy data? ([y]/n)
                    
                  Enter 'q/quit' exit the program.
                  '''))
            while userResponded == False:
                answer = input('Action: ')
                if answer == '' or answer == 'y' or answer == 'yes':
                    print('Proceeding with plot of noisy data...')
                    # read in the noisy data array
                    X = np.load('noisyData.npz')['noisyData']
                    userResponded = True
                elif answer == 'n' or answer == 'no':
                    print('Proceeding with plot of noise-free data...')
                    # read in the recorded data array
                    X  = np.load(str(datadir['recordedData']))
                    userResponded = True
                elif answer == 'q' or answer == 'quit':
                    sys.exit('Exiting program.\n')
                else:
                    print('Invalid response. Please enter \'y/yes\', \'n\no\', or \'q/quit\'.')
            
        else:
            # read in the recorded data array
            X  = np.load(str(datadir['recordedData']))
        
        time = recordingTimes
        if 'sources' in datadir:
            sourcePoints = np.load(str(datadir['sources']))
        else:
            sourcePoints = None
        X = X[rinterval, :, :]
    
    elif args.data:
        # load the 3D data array into variable 'X'
        # X[receiver, time, source]
        wiggleType = 'data'
        if Path('noisyData.npz').exists():
            userResponded = False
            print(textwrap.dedent(
                  '''
                  Detected that band-limited noise has been added to the data array.
                  Would you like to plot the noisy data? ([y]/n)
                    
                  Enter 'q/quit' exit the program.
                  '''))
            while userResponded == False:
                answer = input('Action: ')
                if answer == '' or answer == 'y' or answer == 'yes':
                    print('Proceeding with plot of noisy data...')
                    # read in the noisy data array
                    X = np.load('noisyData.npz')['noisyData']
                    userResponded = True
                elif answer == 'n' or answer == 'no':
                    print('Proceeding with plot of noise-free data...')
                    # read in the recorded data array
                    X  = np.load(str(datadir['recordedData']))
                    userResponded = True
                elif answer == 'q' or answer == 'quit':
                    sys.exit('Exiting program.\n')
                else:
                    print('Invalid response. Please enter \'y/yes\', \'n\no\', or \'q/quit\'.')
            
        else:
            # read in the recorded data array
            X  = np.load(str(datadir['recordedData']))
        
        time = recordingTimes
        if 'sources' in datadir:
            sourcePoints = np.load(str(datadir['sources']))
        else:
            sourcePoints = None
        X = X[rinterval, :, :]
        
    elif args.testfunc:
        wiggleType = 'testfunc'
        if 'testFuncs' in datadir and not Path('VZTestFuncs.npz').exists():
            X = np.load(str(datadir['testFuncs']))
            T = recordingTimes[-1] - recordingTimes[0]
            time = np.linspace(-T, T, 2 * len(recordingTimes) - 1)
            sourcePoints = np.load(str(datadir['samplingPoints']))     
            X = X[rinterval, :, :]
            npad = ((0, 0), (len(recordingTimes) - 1, 0), (0, 0))
            X = np.pad(X, pad_width=npad, mode='constant', constant_values=0)
        
        elif not 'testFuncs' in datadir and Path('VZTestFuncs.npz').exists():
            print('\nDetected that free-space test functions have already been computed...')
            print('Checking consistency with current space-time sampling grid...')
            TFDict = np.load('VZTestFuncs.npz')
            
            samplingGrid = np.load('samplingGrid.npz')
            x = samplingGrid['x']
            y = samplingGrid['y']
            if 'z' in samplingGrid:
                z = samplingGrid['z']
            else:
                z = None
            tau = samplingGrid['tau']
            
            pulse = lambda t : pulseFun.pulse(t)
            velocity = pulseFun.velocity
            peakFreq = pulseFun.peakFreq
            peakTime = pulseFun.peakTime
            
            # set up the convolution times based on the length of the recording time interval
            time = recordingTimes[tinterval]
            T = time[-1] - time[0]
            time = np.linspace(-T, T, 2 * len(time) - 1)
            if samplingIsCurrent(TFDict, receiverPoints, time, velocity, tau, x, y, z, peakFreq, peakTime):
                print('Moving forward to plot test functions...')
                X = TFDict['TFarray']
                sourcePoints = TFDict['samplingPoints']
                
            else:
                if tau[0] != 0:
                    tu = plotParams['tu']
                    if tu != '':
                        print('Recomputing test functions for focusing time %0.2f %s...' %(tau[0], tu))
                    else:
                        print('Recomputing test functions for focusing time %0.2f...' %(tau[0]))
                    X, sourcePoints = sampleSpace(receiverPoints, time - tau[0], velocity,
                                                  x, y, z, pulse)
                else:
                    print('Recomputing test functions...')
                    X, sourcePoints = sampleSpace(receiverPoints, time, velocity,
                                                  x, y, z, pulse)
                    
                if z is None:
                    np.savez('VZTestFuncs.npz', TFarray=X, time=time, receivers=receiverPoints,
                             peakFreq=peakFreq, peakTime=peakTime, velocity=velocity,
                             x=x, y=y, tau=tau, samplingPoints=sourcePoints)
                else:
                    np.savez('VZTestFuncs.npz', TFarray=X, time=time, receivers=receiverPoints,
                             peakFreq=peakFreq, peakTime=peakTime, velocity=velocity,
                             x=x, y=y, z=z, tau=tau, samplingPoints=sourcePoints)
                    
                    
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
                    X = np.load(str(datadir['testFuncs']))
                    T = recordingTimes[-1] - recordingTimes[0]
                    time = np.linspace(-T, T, 2 * len(recordingTimes) - 1)
                    sourcePoints = np.load(str(datadir['samplingPoints']))     
                    X = X[rinterval, :, :]
                    npad = ((0, 0), (len(recordingTimes) - 1, 0), (0, 0))
                    X = np.pad(X, pad_width=npad, mode='constant', constant_values=0)                   
                    userResponded = True
                    break
                
                elif answer == '2':
                    print('\nDetected that free-space test functions have already been computed...')
                    print('Checking consistency with current spatial sampling grid...')
                    TFDict = np.load('VZTestFuncs.npz')
            
                    samplingGrid = np.load('samplingGrid.npz')
                    x = samplingGrid['x']
                    y = samplingGrid['y']
                    if 'z' in samplingGrid:
                        z = samplingGrid['z']
                    else:
                        z = None
                    tau = samplingGrid['tau']
            
                    pulse = lambda t : pulseFun.pulse(t)
                    velocity = pulseFun.velocity
                    peakFreq = pulseFun.peakFreq
                    peakTime = pulseFun.peakTime
                    
                    # set up the convolution times based on the length of the recording time interval
                    time = recordingTimes[tinterval]
                    T = time[-1] - time[0]
                    time = np.linspace(-T, T, 2 * len(time) - 1)
                    if samplingIsCurrent(TFDict, receiverPoints, time, velocity, tau, x, y, z, peakFreq, peakTime):
                        print('Moving forward to plot test functions...')
                        X = TFDict['TFarray']
                        sourcePoints = TFDict['samplingPoints']
                    
                    else:                
                        if tau[0] != 0:
                            tu = plotParams['tu']
                            if tu != '':
                                print('Recomputing test functions for focusing time %0.2f %s...' %(tau[0], tu))
                            else:
                                print('Recomputing test functions for focusing time %0.2f...' %(tau[0]))
                            X, sourcePoints = sampleSpace(receiverPoints, time - tau[0], velocity,
                                                          x, y, z, pulse)
                        else:
                            print('Recomputing test functions...')
                            X, sourcePoints = sampleSpace(receiverPoints, time, velocity,
                                                          x, y, z, pulse)
                    
                        if z is None:
                            np.savez('VZTestFuncs.npz', TFarray=X, time=time, receivers=receiverPoints,
                                     peakFreq=peakFreq, peakTime=peakTime, velocity=velocity,
                                     x=x, y=y, tau=tau, samplingPoints=sourcePoints)
                        else:
                            np.savez('VZTestFuncs.npz', TFarray=X, time=time, receivers=receiverPoints,
                                     peakFreq=peakFreq, peakTime=peakTime, velocity=velocity,
                                     x=x, y=y, z=z, tau=tau, samplingPoints=sourcePoints)
                    
                    userResponded = True
                
                elif answer == 'q' or answer == 'quit':
                    sys.exit('Exiting program.')
                
                else:
                    print('Invalid response. Please enter \'1\', \'2\', or \'q/quit\'.')
        
        else:                
            print('\nComputing free-space test functions for the current space-time sampling grid...')
            # set up the convolution times based on the length of the recording time interval
            time = recordingTimes[tinterval]
            T = time[-1] - time[0]
            time = np.linspace(-T, T, 2 * len(time) - 1)
            
            samplingGrid = np.load('samplingGrid.npz')
            x = samplingGrid['x']
            y = samplingGrid['y']
            if 'z' in samplingGrid:
                z = samplingGrid['z']
            else:
                z = None
            tau = samplingGrid['tau']
            
            pulse = lambda t : pulseFun.pulse(t)
            velocity = pulseFun.velocity
            peakFreq = pulseFun.peakFreq
            peakTime = pulseFun.peakTime
            
            if tau[0] != 0:
                tu = plotParams['tu']
                if tu != '':
                    print('Computing test functions for focusing time %0.2f %s...' %(tau[0], tu))
                else:
                    print('Computing test functions for focusing time %0.2f...' %(tau[0]))
                X, sourcePoints = sampleSpace(receiverPoints, time - tau[0], velocity,
                                              x, y, z, pulse)
            else:
                X, sourcePoints = sampleSpace(receiverPoints, time, velocity,
                                              x, y, z, pulse)
                    
            if z is None:
                np.savez('VZTestFuncs.npz', TFarray=X, time=time, receivers=receiverPoints,
                         peakFreq=peakFreq, peakTime=peakTime, velocity=velocity,
                         x=x, y=y, tau=tau, samplingPoints=sourcePoints)
            else:
                np.savez('VZTestFuncs.npz', TFarray=X, time=time, receivers=receiverPoints,
                         peakFreq=peakFreq, peakTime=peakTime, velocity=velocity,
                         x=x, y=y, z=z, tau=tau, samplingPoints=sourcePoints)
    
    #==============================================================================    
    if Path('window.npz').exists() and wiggleType == 'data':   
        t0 = tstart
        tf = tstop
         
        # Apply the source window
        sstart = windowDict['sstart']
        sstop = windowDict['sstop']
        sstep = windowDict['sstep']
            
    else:
        t0 = time[0]
        tf = time[-1]
        
        sstart = 0
        sstop = X.shape[2]
        sstep = 1
            
    sinterval = np.arange(sstart, sstop, sstep)
        
    X = X[:, :, sinterval]
    if sourcePoints is not None:
        sourcePoints = sourcePoints[sinterval, :]
    
    # increment source/recording interval and receiver interval to be consistent
    # with one-based indexing (i.e., count from one instead of zero)
    sinterval += 1
    rinterval += 1
    rstart += 1
    
    Ns = X.shape[2]
    
    remove_keymap_conflicts({'left', 'right', 'up', 'down', 'save'})
    if args.map:
        fig, ax1, ax2 = setFigure(num_axes=2, mode=plotParams['view_mode'],
                                  ax2_dim=receiverPoints.shape[1])
            
        ax1.volume = X
        ax1.index = Ns // 2
        title = wave_title(ax1.index, sinterval, sourcePoints, wiggleType, plotParams)
        plotWiggles(ax1, X[:, :, ax1.index], time, t0, tf, rstart, rinterval, receiverPoints, title, wiggleType, plotParams)
        
        ax2.index = ax1.index
        plotMap(ax2, ax2.index, receiverPoints, sourcePoints, scatterer, wiggleType, plotParams)
        plt.tight_layout()
        fig.canvas.mpl_connect('key_press_event', lambda event: process_key_waves(event, time, t0, tf, rstart, rinterval,
                                                                                  sinterval, receiverPoints, sourcePoints,
                                                                                  Ns, scatterer, args.map, wiggleType, plotParams))
    
    else:
        fig, ax = setFigure(num_axes=1, mode=plotParams['view_mode'])
            
        ax.volume = X
        ax.index = Ns // 2
        title = wave_title(ax.index, sinterval, sourcePoints, wiggleType, plotParams)
        plotWiggles(ax, X[:, :, ax.index], time, t0, tf, rstart, rinterval, receiverPoints, title, wiggleType, plotParams)
        plt.tight_layout()
        fig.canvas.mpl_connect('key_press_event', lambda event: process_key_waves(event, time, t0, tf, rstart, rinterval,
                                                                                  sinterval, receiverPoints, sourcePoints,
                                                                                  Ns, scatterer, args.map, wiggleType, plotParams))
    
    plt.show()
