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
from vezda.sampling_utils import samplingIsCurrent, sampleSpaceTime

sys.path.append(os.getcwd())
import pulseFun

def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, choices=['data', 'testfunc'], required=True,
                        help='''Specify whether to plot the recorded data or simulated test functions.
                        Type must be either \'data\' or \'testfunc\'.''')
    parser.add_argument('--tu', type=str,
                        help='Specify the time units (e.g., \'s\' or \'ms\').')
    parser.add_argument('--au', type=str,
                        help='Specify the amplitude units (e.g., \'m\' or \'mm\').')
    parser.add_argument('--title', type=str,
                        help='''Specify a title for the wiggle plot. Default title is
                        \'Data\' if \'--type=data\' and 'Test Function' if \'--type=testfunc\'.''')
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
            
        if args.title is not None:
            if args.type == 'data':
                plotParams['data_title'] = args.title
            elif args.type == 'testfunc':
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
            if args.type == 'data':
                plotParams['data_title'] = args.title
            elif args.type == 'testfunc':
                plotParams['tf_title'] = args.title
        
    pickle.dump(plotParams, open('plotParams.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)

    #==============================================================================
    # Load the relevant data to plot
    datadir = np.load('datadir.npz')
    receiverPoints = np.load(str(datadir['receivers']))
    recordingTimes = np.load(str(datadir['recordingTimes']))
    
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
    
        dt = (recordingTimes[-1] - recordingTimes[0]) / (len(recordingTimes) - 1)
    
        twStart = int(round(tstart / dt))
        twStop = int(round(tstop / dt))
    
    else:
        rstart = 0
        rstop = receiverPoints.shape[0]
        rstep = 1
        
        tstart = recordingTimes[0]
        tstop = recordingTimes[-1]
        
        twStart = 0
        twStop = len(recordingTimes)
        tstep = 1
    
    rinterval = np.arange(rstart, rstop, rstep)
    receiverPoints = receiverPoints[rinterval, :]
    
    tinterval = np.arange(twStart, twStop, tstep)
        
    if args.type == 'data':
        # load the 3D data array into variable 'X'
        # X[receiver, time, source]
        X = np.load(str(datadir['recordedData']))
        time = recordingTimes
        sourcePoints = np.load(str(datadir['sources']))
        X = X[rinterval, :, :]
        
    elif args.type == 'testfunc':
        if 'testFuncs' in datadir and not Path('VZTestFuncs.npz').exists():
            X = np.load(str(datadir['testFuncs']))
            time = recordingTimes[tinterval]
            sourcePoints = np.load(str(datadir['samplingPoints']))
            
            X = X[rinterval, :, :]
            X = X[:, tinterval, :]
            sourcePoints = sourcePoints[:, :-1]
        
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
            
            time = recordingTimes[tinterval]
            if samplingIsCurrent(TFDict, time, velocity, tau, x, y, z, peakFreq, peakTime):
                print('Moving forward to plot test functions...')
                X = TFDict['TFarray']
                sourcePoints = TFDict['samplingPoints']
                
                    
            else:
                print('Recomputing test functions...')
                X, sourcePoints = sampleSpaceTime(receiverPoints, time,
                                                  velocity, tau, x, y, z, pulse)
                    
                if z is None:
                    np.savez('VZTestFuncs.npz', TFarray=X, time=time,
                             peakFreq=peakFreq, peakTime=peakTime, velocity=velocity,
                             x=x, y=y, tau=tau, samplingPoints=sourcePoints)
                else:
                    np.savez('VZTestFuncs.npz', TFarray=X, time=time,
                             peakFreq=peakFreq, peakTime=peakTime, velocity=velocity,
                             x=x, y=y, z=z, tau=tau, samplingPoints=sourcePoints)
                    
            sourcePoints = sourcePoints[:, :-1]
            X = X[:, :, :, 0]
                    
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
                    time = recordingTimes[tinterval]
                    samplingPoints = np.load(str(datadir['samplingPoints']))
                    sourcePoints = samplingPoints[:, :-1]
                    X = X[rinterval, :, :]
                    X = X[:, tinterval, :]
                    userResponded = True
                    break
                
                elif answer == '2':
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
                    
                    time = recordingTimes[tinterval]
                    if samplingIsCurrent(TFDict, time, velocity, tau, x, y, z, peakFreq, peakTime):
                        print('Moving forward to plot test functions...')
                        X = TFDict['TFarray']
                        sourcePoints = TFDict['samplingPoints']
                    
                    else:
                        print('Recomputing test functions...')
                        X, sourcePoints = sampleSpaceTime(receiverPoints, time, velocity,
                                                          tau, x, y, z, pulse)
                    
                        if z is None:
                            np.savez('VZTestFuncs.npz', TFarray=X, time=time,
                                     peakFreq=peakFreq, peakTime=peakTime, velocity=velocity,
                                     x=x, y=y, tau=tau, samplingPoints=sourcePoints)
                        else:
                            np.savez('VZTestFuncs.npz', TFarray=X, time=time,
                                     peakFreq=peakFreq, peakTime=peakTime, velocity=velocity,
                                     x=x, y=y, z=z, tau=tau, samplingPoints=sourcePoints)
                    
                    sourcePoints = sourcePoints[:, :-1]
                    X = X[:, :, :, 0]
                    userResponded = True
                
                elif answer == 'q' or answer == 'quit':
                    sys.exit('Exiting program.')
                
                else:
                    print('Invalid response. Please enter \'1\', \'2\', or \'q/quit\'.')
        
        else:                
            print('\nComputing free-space test functions for the current sampling grid...')
            time = recordingTimes[tinterval]
            X, sourcePoints = sampleSpaceTime(receiverPoints, time, velocity,
                                              tau, x, y, z, pulse)
                    
            if z is None:
                np.savez('VZTestFuncs.npz', TFarray=X, time=time,
                         peakFreq=peakFreq, peakTime=peakTime, velocity=velocity,
                         x=x, y=y, tau=tau, samplingPoints=sourcePoints)
            else:
                np.savez('VZTestFuncs.npz', TFarray=X, time=time,
                         peakFreq=peakFreq, peakTime=peakTime, velocity=velocity,
                         x=x, y=y, z=z, tau=tau, samplingPoints=sourcePoints)
            
            sourcePoints = sourcePoints[:, :-1]
            X = X[:, :, :, 0]
        
    #==============================================================================
    if args.type == 'data':
        t0 = tstart
        tf = tstop
    else:
        t0 = recordingTimes[0]
        tf = recordingTimes[-1]
    
    if Path('window.npz').exists() and args.type == 'data':            
        # Apply the source window
        sstart = windowDict['sstart']
        sstop = windowDict['sstop']
        sstep = windowDict['sstep']
            
    else:
        sstart = 0
        sstop = X.shape[2]
        sstep = 1
            
    sinterval = np.arange(sstart, sstop, sstep)
        
    X = X[:, :, sinterval]
    sourcePoints = sourcePoints[sinterval, :]
    
    Ns = X.shape[2]
    
    remove_keymap_conflicts({'left', 'right', 'up', 'down', 'save'})
    if args.map:
        fig, ax1, ax2 = setFigure(num_axes=2, mode=plotParams['view_mode'],
                                  ax2_dim=receiverPoints.shape[1])
            
        ax1.volume = X
        ax1.index = Ns // 2
        title = wave_title(ax1.index, sinterval, sourcePoints, args.type, plotParams)
        plotWiggles(ax1, X[:, :, ax1.index], time, t0, tf, rstart, rinterval, receiverPoints, title, args.type, plotParams)
        
        ax2.index = ax1.index
        plotMap(ax2, ax2.index, receiverPoints, sourcePoints, scatterer, args.type, plotParams)
        plt.tight_layout()
        fig.canvas.mpl_connect('key_press_event', lambda event: process_key_waves(event, time, t0, tf, rstart, rinterval,
                                                                                  sinterval, receiverPoints, sourcePoints,
                                                                                  scatterer, args.map, args.type, plotParams))
    
    else:
        fig, ax = setFigure(num_axes=1, mode=plotParams['view_mode'])
            
        ax.volume = X
        ax.index = Ns // 2
        title = wave_title(ax.index, sinterval, sourcePoints, args.type, plotParams)
        plotWiggles(ax, X[:, :, ax.index], time, t0, tf, rstart, rinterval, receiverPoints, title, args.type, plotParams)
        plt.tight_layout()
        fig.canvas.mpl_connect('key_press_event', lambda event: process_key_waves(event, time, t0, tf, rstart, rinterval,
                                                                                  sinterval, receiverPoints, sourcePoints,
                                                                                  scatterer, args.map, args.type, plotParams))
    
    plt.show()
