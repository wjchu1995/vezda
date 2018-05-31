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
import pickle
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from pathlib import Path

def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, choices=['data', 'testfunc'], required=True,
                        help='''Specify whether to plot the scattered wave data or simulated test functions.
                        Type must be either \'data\' or \'testfunc\'.''')
    parser.add_argument('--method', type=str, choices=['lsm', 'telsm'],
                        help='''Specify whether to plot the test functions from the classical
                        linear sampling method 'lsm' or total-energy linear sampling method 'telsm'.''')
    parser.add_argument('--tu', type=str,
                        help='Specify the time units (e.g., \'s\' or \'ms\').')
    parser.add_argument('--au', type=str,
                        help='Specify the amplitude units (e.g., \'m\' or \'mm\').')
    parser.add_argument('--format', '-f', type=str, default='pdf', choices=['png', 'pdf', 'ps', 'eps', 'svg'],
                        help='''Specify the image format of the saved file. Accepted formats are png, pdf,
                        ps, eps, and svg. Default format is set to pdf.''')
    parser.add_argument('--map', type=str, default='no', choices=['n', 'no', 'false', 'y', 'yes', 'true'],
                        help='''Plot a map of the receiver and source/sampling point locations. The current
                        source/sampling point will be highlighted. The boundary of the scatterer will also
                        be shown if available.''')
    args = parser.parse_args()
    #==============================================================================
    
    datadir = np.load('datadir.npz')
    recordingTimes = np.load(str(datadir['recordingTimes']))
    receiverPoints = np.load(str(datadir['receivers']))
    if 'scatterer' in datadir:
        scatterer = np.load(str(datadir['scatterer']))
    else:
        scatterer = None
    
    # if a plotParams.pkl file already exists, load relevant parameters
    if Path('plotParams.pkl').exists():
        plotParams = pickle.load(open('plotParams.pkl', 'rb'))
        
        # for image/map plots
        xlabel = plotParams['xlabel']
        ylabel = plotParams['ylabel']
        zlabel = plotParams['zlabel']
        xu = plotParams['xu']
        yu = plotParams['yu']
        zu = plotParams['zu']
        invertX = plotParams['invert_xaxis']
        invertY = plotParams['invert_yaxis']
        invertZ = plotParams['invert_zaxis']
        showScatterer = plotParams['show_scatterer']
        
        # for both wiggle plots and image plots
        pltformat = plotParams['pltformat']
        
        # update parameters for wiggle plots based on passed arguments
        if args.tu is not None:
            tu = args.tu
            plotParams['tu'] = tu
        else:
            tu = plotParams['tu']
        
        if args.au is not None:
            au = args.au
            plotParams['au'] = au
        else:
            au = plotParams['au']
        
        pickle.dump(plotParams, open('plotParams.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)
    
    else: # create a plotParams dictionary file with default values
        plotParams = {}
        #for both image and wiggle plots
        pltformat = 'pdf'
        plotParams['pltformat'] = pltformat
        
        # for image/map plots
        isolevel = 0.7
        plotParams['isolevel'] = isolevel
        xlabel = ''
        plotParams['xlabel'] = xlabel
        ylabel = ''
        plotParams['ylabel'] = ylabel
        zlabel = ''
        plotParams['zlabel'] = zlabel
        xu = ''
        plotParams['xu'] = xu
        yu = ''
        plotParams['yu'] = yu
        zu = ''
        plotParams['zu'] = zu
        plotParams['colorbar'] = False
        invertX = False
        plotParams['invert_xaxis'] = invertX
        invertY = False
        plotParams['invert_yaxis'] = invertY
        invertZ = False
        plotParams['invert_zaxis'] = invertZ
        showScatterer = False
        plotParams['show_scatterer'] = showScatterer
        plotParams['show_sources'] = True
        plotParams['show_receivers'] = True
        
        # update parameters for wiggle plots based on passed arguments
        if args.tu is not None:
            tu = args.tu
        else:
            tu = ''
        plotParams['tu'] = tu
        
        if args.au is not None:
            au = args.au
        else:
            au = ''
        plotParams['au'] = au
        
        pickle.dump(plotParams, open('plotParams.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)
        
    #==============================================================================
    if args.type == 'data':
        # load the 3D data array into variable 'X'
        # X[receiver, time, source]
        X = np.load(str(datadir['scatteredData']))
        time = recordingTimes
        tau = None
        sourcePoints = np.load(str(datadir['sources']))
        
    elif args.type == 'testfunc' and args.method == 'lsm':
        TFType = 'lsm'
        testFuncs = np.load('VZTestFuncsLSM.npz')
        TFarray = testFuncs['TFshifted']
        tinterval = testFuncs['tinterval']
        X = TFarray[:, tinterval, :]
        time = testFuncs['time'] 
        tau = testFuncs['tau']
        sourcePoints = testFuncs['samplingPoints']
        
    elif args.type == 'testfunc' and args.method == 'telsm':
        TFType = 'telsm'
        testFuncs = np.load('VZTestFuncsTELSM.npz')
        TFarray = testFuncs['TFarray']
        X = TFarray[:, :, :, 0]
        time = testFuncs['time'] 
        tau = None
        samplingPoints = testFuncs['samplingPoints']
        # Extract all but last column of sampling points,
        # which corresponds to tau (sampling point in time)
        sourcePoints = samplingPoints[:, :-1]
        
    elif args.type == 'testfunc' and args.method is None:
        if Path('VZTestFuncsLSM.npz').exists() and Path('VZTestFuncsTELSM.npz').exists():
            userResponded = False
            print(textwrap.dedent(
                 '''
                 Detected two different simulated test functions available to plot.
                 
                 Enter '1' to plot the test functions for the classical linear sampling method. (Default)
                 Enter '2' to plot the test functions for the total-energy linear sampling method.
                 Enter 'q/quit' to exit.
                 '''))
            while userResponded == False:
                answer = input('Action: ')
                if answer == '' or answer == '1':
                    TFType = 'lsm'
                    testFuncs = np.load('VZTestFuncsLSM.npz')
                    X = testFuncs['TFshifted']
                    tinterval = testFuncs['tinterval']
                    X = X[:, tinterval, :]
                    time = testFuncs['time'] 
                    tau = testFuncs['tau']
                    sourcePoints = testFuncs['samplingPoints']
                    userResponded = True
                    break
                
                elif answer == '2':
                    TFType = 'telsm'
                    testFuncs = np.load('VZTestFuncsTELSM.npz')
                    TFarray = testFuncs['TFarray']
                    X = TFarray[:, :, :, 0]
                    time = testFuncs['time'] 
                    tau = None
                    samplingPoints = testFuncs['samplingPoints']
                    # Extract all but last column of sampling points,
                    # which corresponds to tau (sampling point in time)
                    sourcePoints = samplingPoints[:, :-1]
                    userResponded = True
                    break
                
                elif answer == 'q' or answer == 'quit':
                    sys.exit('Exiting program.')
                else:
                    print('Invalid response. Please enter \'1\', \'2\', or \'q/quit\'.')
        
        elif Path('VZTestFuncsLSM.npz').exists() and not Path('VZTestFuncsTELSM.npz').exists():
            TFType = 'lsm'
            testFuncs = np.load('VZTestFuncsLSM.npz')
            X = testFuncs['TFarray']
            time = testFuncs['time'] 
            tau = testFuncs['tau']
            sourcePoints = testFuncs['samplingPoints']
            userResponded = True
        
        elif not Path('VZTestFuncsLSM.npz').exists() and Path('VZTestFuncsTELSM.npz').exists():
            TFType = 'telsm'
            testFuncs = np.load('VZTestFuncsTELSM.npz')
            TFarray = testFuncs['TFarray']
            X = TFarray[:, :, :, 0]
            time = testFuncs['time'] 
            tau = None
            samplingPoints = testFuncs['samplingPoints']
            # Extract all but last column of sampling points,
            # which corresponds to tau (sampling point in time)
            sourcePoints = samplingPoints[:, :-1]
            userResponded = True
            
        elif not Path('VZTestFuncsLSM.npz').exists() and not Path('VZTestFuncsTELSM.npz').exists():
            sys.exit(textwrap.dedent(
                    '''
                    Error: No test functions exist to plot. Run the command
                    
                    vzsolve --method=<lsm/telsm> --medium=<constant/variable> 
                    
                    to make test functions available for plotting.
                    '''))
        
    if args.map == 'n' or args.map == 'no' or args.map == 'false':
        showMap = False
    
    elif args.map == 'y' or args.map == 'yes' or args.map == 'true':
        showMap = True
        
    #==============================================================================
    def remove_keymap_conflicts(new_keys_set):
        for prop in plt.rcParams:
            if prop.startswith('keymap.'):
                keys = plt.rcParams[prop]
                remove_list = set(keys) & new_keys_set
                for key in remove_list:
                    keys.remove(key)
    #==============================================================================
    #def save_key(event):
    #   if args.type == 'data':
    #       fig.savefig(args.type + '_src' + str(sinterval[ax.index]) + '.' + args.format,
    #                  format=pltformat, bbox_inches='tight', transparent=True)
    #   elif args.type == 'testfunc':
    #       fig.savefig(args.type + '_pnt' + str(sinterval[ax.index]) + '.' + args.format,
    #                  format=pltformat, bbox_inches='tight', transparent=True)
    
    def process_key(event, tstart, tstop, rinterval, sinterval,
                    receiverPoints, sourcePoints, scatterer,
                    args, tau, recordingTimes):
        if showMap:
            fig = event.canvas.figure
            ax1 = fig.axes[0]
            ax2 = fig.axes[1]
            if event.key == 'left' or event.key == 'down':
                previous_slice(ax1, tstart, tstop, rinterval, sinterval, args, tau,
                               recordingTimes, receiverPoints, sourcePoints)
                previous_source(ax2, args, rinterval, sinterval, receiverPoints, sourcePoints, scatterer)
            elif event.key == 'right' or event.key == 'up':
                next_slice(ax1, tstart, tstop, rinterval, sinterval, args, tau,
                           recordingTimes, receiverPoints, sourcePoints)
                next_source(ax2, args, rinterval, sinterval, receiverPoints, sourcePoints, scatterer)
        else:
            fig = event.canvas.figure
            ax = fig.axes[0]
            if event.key == 'left' or event.key == 'down':
                previous_slice(ax, tstart, tstop, rinterval, sinterval, args, tau,
                               recordingTimes, receiverPoints, sourcePoints)
            elif event.key == 'right' or event.key == 'up':
                next_slice(ax, tstart, tstop, rinterval, sinterval, args, tau,
                           recordingTimes, receiverPoints, sourcePoints)
        fig.canvas.draw()
    #==============================================================================    
    def previous_slice(ax, tstart, tstop, rinterval, sinterval, args, tau,
                       recordingTimes, receiverPoints, sourcePoints):
        volume = ax.volume
        ax.index = (ax.index - 1) % volume.shape[2]  # wrap around using %
        wiggle_plot(ax, volume[:, :, ax.index], tstart, tstop, rinterval, sinterval,
                    args, tau, recordingTimes, receiverPoints, sourcePoints)
    
    def next_slice(ax, tstart, tstop, rinterval, sinterval, args, tau,
                   recordingTimes, receiverPoints, sourcePoints):
        volume = ax.volume
        ax.index = (ax.index + 1) % volume.shape[2]
        wiggle_plot(ax, volume[:, :, ax.index], tstart, tstop, rinterval, sinterval,
                    args, tau, recordingTimes, receiverPoints, sourcePoints)
        
    def wiggle_plot(ax, X, tstart, tstop, rinterval, sinterval, args, tau,
                    recordingTimes, receiverPoints, sourcePoints):
        ax.clear()
        Nr, Nt = X.shape
        if Nr > 1:
            ax.set_ylabel('Receiver')
            ax.set_yticks(rinterval)                
            if args.type == 'testfunc':
                ax.set_yticklabels(pltrstart + rinterval)
            plt.setp(ax.get_yticklabels(), visible=True)
            plt.setp(ax.get_yticklines(),visible=True)
            # rescale all wiggle traces by largest displacement range
            scaleFactor = np.max(np.ptp(X, axis=1))
            if scaleFactor != 0:
                X /= scaleFactor
            
            for r in range(Nr):
                ax.plot(time, rinterval[r] + X[r, :], color='darkgray')
                ax.fill_between(time, rinterval[r], rinterval[r] + X[r, :],
                                where=(rinterval[r] + X[r, :] > rinterval[r]), color='m')
                ax.fill_between(time, rinterval[r], rinterval[r] + X[r, :],
                                where=(rinterval[r] + X[r, :] < rinterval[r]), color='c')
                
        else: # Nr == 1
            ax.yaxis.get_offset_text().set_x(-0.1)
            if receiverPoints.shape[1] == 2:
                if args.type == 'testfunc':
                    if au != '' and xu != '' and yu != '':
                        ax.set_ylabel('''Amplitude (%s) [Receiver %s @ (%0.2f %s, %0.2f %s)]'''
                                      %(au, pltrstart,
                                        receiverPoints[rinterval[0], 0], xu,
                                        receiverPoints[rinterval[0], 1], yu))
                    elif au == '' and xu != '' and yu != '':
                        ax.set_ylabel('''Amplitude [Receiver %s @ (%0.2f %s, %0.2f %s)]'''
                                      %(pltrstart,
                                        receiverPoints[rinterval[0], 0], xu,
                                        receiverPoints[rinterval[0], 1], yu))
                    elif au != '' and xu == '' and yu == '':
                        ax.set_ylabel('''Amplitude (%s) [Receiver %s @ (%0.2f, %0.2f)]'''
                                      %(au, pltrstart,
                                        receiverPoints[rinterval[0], 0],
                                        receiverPoints[rinterval[0], 1]))
                    elif au == '' and xu == '' and yu == '':
                        ax.set_ylabel('''Amplitude [Receiver %s @ (%0.2f, %0.2f)]'''
                                      %(pltrstart,
                                        receiverPoints[rinterval[0], 0],
                                        receiverPoints[rinterval[0], 1]))
                else: #args.type == 'data'
                    if au != '' and xu != '' and yu != '':
                        ax.set_ylabel('''Amplitude (%s) [Receiver %s @ (%0.2f %s, %0.2f %s)]'''
                                      %(au, rinterval[0],
                                        receiverPoints[rinterval[0], 0], xu,
                                        receiverPoints[rinterval[0], 1], yu))
                    elif au == '' and xu != '' and yu != '':
                        ax.set_ylabel('''Amplitude [Receiver %s @ (%0.2f %s, %0.2f %s)]'''
                                      %(rinterval[0],
                                        receiverPoints[rinterval[0], 0], xu,
                                        receiverPoints[rinterval[0], 1], yu))
                    elif au != '' and xu == '' and yu == '':
                        ax.set_ylabel('''Amplitude (%s) [Receiver %s @ (%0.2f, %0.2f)]'''
                                      %(au, rinterval[0],
                                        receiverPoints[rinterval[0], 0],
                                        receiverPoints[rinterval[0], 1]))
                    elif au == '' and xu == '' and yu == '':
                        ax.set_ylabel('''Amplitude [Receiver %s @ (%0.2f, %0.2f)]'''
                                      %(rinterval[0],
                                        receiverPoints[rinterval[0], 0],
                                        receiverPoints[rinterval[0], 1]))
            
            elif receiverPoints.shape[1] == 3:
                if args.type == 'testfunc':
                    if au != '' and xu != '' and yu != '' and zu != '':
                        ax.set_ylabel('''Amplitude (%s) [Receiver %s @ (%0.2f %s, %0.2f %s, %0.2f %s)]'''
                                      %(au, pltrstart,
                                        receiverPoints[rinterval[0], 0], xu,
                                        receiverPoints[rinterval[0], 1], yu,
                                        receiverPoints[rinterval[0], 2], zu))
                    elif au == '' and xu != '' and yu != '' and zu != '':
                        ax.set_ylabel('''Amplitude [Receiver %s @ (%0.2f %s, %0.2f %s, %0.2f %s)]'''
                                      %(pltrstart,
                                        receiverPoints[rinterval[0], 0], xu,
                                        receiverPoints[rinterval[0], 1], yu,
                                        receiverPoints[rinterval[0], 2], zu))
                    elif au != '' and xu == '' and yu == '' and zu == '':
                        ax.set_ylabel('''Amplitude (%s) [Receiver %s @ (%0.2f, %0.2f, %0.2f)]'''
                                      %(au, pltrstart,
                                        receiverPoints[rinterval[0], 0],
                                        receiverPoints[rinterval[0], 1],
                                        receiverPoints[rinterval[0], 2]))
                    elif au == '' and xu == '' and yu == '' and zu == '':
                        ax.set_ylabel('''Amplitude [Receiver %s @ (%0.2f, %0.2f, %0.2f)]'''
                                      %(pltrstart,
                                        receiverPoints[rinterval[0], 0],
                                        receiverPoints[rinterval[0], 1],
                                        receiverPoints[rinterval[0], 2]))
                else: #args.type == 'data'
                    if au != '' and xu != '' and yu != '' and zu != '':
                        ax.set_ylabel('''Amplitude (%s) [Receiver %s @ (%0.2f %s, %0.2f %s, %0.2f %s)]'''
                                      %(au, rinterval[0],
                                        receiverPoints[rinterval[0], 0], xu,
                                        receiverPoints[rinterval[0], 1], yu,
                                        receiverPoints[rinterval[0], 2], zu))
                    elif au == '' and xu != '' and yu != '' and zu != '':
                        ax.set_ylabel('''Amplitude [Receiver %s @ (%0.2f %s, %0.2f %s, %0.2f %s)]'''
                                      %(rinterval[0],
                                        receiverPoints[rinterval[0], 0], xu,
                                        receiverPoints[rinterval[0], 1], yu,
                                        receiverPoints[rinterval[0], 2], zu))
                    elif au != '' and xu == '' and yu == '' and zu == '':
                        ax.set_ylabel('''Amplitude (%s) [Receiver %s @ (%0.2f, %0.2f, %0.2f)]'''
                                      %(au, rinterval[0],
                                        receiverPoints[rinterval[0], 0],
                                        receiverPoints[rinterval[0], 1],
                                        receiverPoints[rinterval[0], 2]))
                    elif au == '' and xu == '' and yu == '' and zu == '':
                        ax.set_ylabel('''Amplitude [Receiver %s @ (%0.2f, %0.2f, %0.2f)]'''
                                      %(rinterval[0],
                                        receiverPoints[rinterval[0], 0],
                                        receiverPoints[rinterval[0], 1],
                                        receiverPoints[rinterval[0], 2]))
                       
            ax.plot(time, X[0, :], 'darkgray')
            ax.fill_between(time, 0, X[0, :], where=(X[0, :] > 0), color='m')
            ax.fill_between(time, 0, X[0, :], where=(X[0, :] < 0), color='c')
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            
        if sourcePoints.shape[1] == 2:
            if args.type == 'data':
                if  xu != '' and yu != '':
                    ax.set_title('''Scattered Wave [Source %s @ (%0.2f %s, %0.2f %s)]'''
                                 %(sinterval[ax.index],
                                   sourcePoints[sinterval[ax.index], 0], xu,
                                   sourcePoints[sinterval[ax.index], 1], yu))
                else:
                    ax.set_title('''Scattered Wave [Source %s @ (%0.2f, %0.2f)]'''
                                 %(sinterval[ax.index],
                                   sourcePoints[sinterval[ax.index], 0],
                                   sourcePoints[sinterval[ax.index], 1]))
                
            elif args.type == 'testfunc' and TFType == 'telsm':
                if xu != '' and yu != '':
                    ax.set_title('''Test Function [$\\bf{z}$ @ (%0.2f %s, %0.2f %s)]'''
                                 %(sourcePoints[sinterval[ax.index], 0], xu,
                                   sourcePoints[sinterval[ax.index], 1], yu))
                elif xu == '' and yu == '':
                    ax.set_title('''Test Function [$\\bf{z}$ @ (%0.2f, %0.2f)]'''
                                 %(sourcePoints[sinterval[ax.index], 0],
                                   sourcePoints[sinterval[ax.index], 1]))
            
            elif args.type == 'testfunc' and TFType == 'lsm':
                if tu != '' and xu != '' and yu != '':
                    ax.set_title('''Test Function [$\\tau$ = %s %s, $\\bf{z}$ @ (%0.2f %s, %0.2f %s)]'''
                                 %(tau, tu,
                                   sourcePoints[sinterval[ax.index], 0], xu,
                                   sourcePoints[sinterval[ax.index], 1], yu))
                elif tu != '' and xu == '' and yu == '':
                    ax.set_title('''Test Function [$\\tau$ = %s %s, $\\bf{z}$ @ (%0.2f, %0.2f)]'''
                                 %(tau, tu,
                                   sourcePoints[sinterval[ax.index], 0],
                                   sourcePoints[sinterval[ax.index], 1]))
                elif tu == '' and xu != '' and yu != '':
                    ax.set_title('''Test Function [$\\tau$ = %s, $\\bf{z}$ @ (%0.2f %s, %0.2f %s)]'''
                                 %(tau,
                                   sourcePoints[sinterval[ax.index], 0], xu,
                                   sourcePoints[sinterval[ax.index], 1], yu))
                elif tu == '' and xu == '' and yu == '':
                    ax.set_title('''Test Function [$\\tau$ = %s, $\\bf{z}$ @ (%0.2f, %0.2f)]'''
                                 %(tau,
                                   sourcePoints[sinterval[ax.index], 0],
                                   sourcePoints[sinterval[ax.index], 1]))
                    
        elif sourcePoints.shape[1] == 3:
            if args.type == 'data':
                if  xu != '' and yu != '' and zu != '':
                    ax.set_title('''Scattered Wave [Source %s @ (%0.2f %s, %0.2f %s, %0.2f %s)]'''
                                 %(sinterval[ax.index],
                                   sourcePoints[sinterval[ax.index], 0], xu,
                                   sourcePoints[sinterval[ax.index], 1], yu,
                                   sourcePoints[sinterval[ax.index], 2], zu))
                else:
                    ax.set_title('''Scattered Wave [Source %s @ (%0.2f, %0.2f, %0.2f)]'''
                                 %(sinterval[ax.index],
                                   sourcePoints[sinterval[ax.index], 0],
                                   sourcePoints[sinterval[ax.index], 1],
                                   sourcePoints[sinterval[ax.index], 2]))
                
            elif args.type == 'testfunc' and TFType == 'telsm':
                if xu != '' and yu != '' and zu != '':
                    ax.set_title('''Test Function [$\\bf{z}$ @ (%0.2f %s, %0.2f %s, %0.2f %s)]'''
                                 %(sourcePoints[sinterval[ax.index], 0], xu,
                                   sourcePoints[sinterval[ax.index], 1], yu,
                                   sourcePoints[sinterval[ax.index], 2], zu))
                elif xu == '' and yu == '' and zu == '':
                    ax.set_title('''Test Function [$\\bf{z}$ @ (%0.2f, %0.2f, %0.2f)]'''
                                 %(sourcePoints[sinterval[ax.index], 0],
                                   sourcePoints[sinterval[ax.index], 1],
                                   sourcePoints[sinterval[ax.index], 2]))
                    
            elif args.type == 'testfunc' and TFType == 'lsm':
                if tu != '' and xu != '' and yu != '' and zu != '':
                    ax.set_title('''Test Function [$\\tau$ = %s %s, $\\bf{z}$ @ (%0.2f %s, %0.2f %s, %0.2f %s)]'''
                                 %(tau, tu,
                                   sourcePoints[sinterval[ax.index], 0], xu,
                                   sourcePoints[sinterval[ax.index], 1], yu,
                                   sourcePoints[sinterval[ax.index], 2], zu))
                elif tu != '' and xu == '' and yu == '' and zu == '':
                    ax.set_title('''Test Function [$\\tau$ = %s %s, $\\bf{z}$ @ (%0.2f, %0.2f, %0.2f)]'''
                                 %(tau, tu,
                                   sourcePoints[sinterval[ax.index], 0],
                                   sourcePoints[sinterval[ax.index], 1],
                                   sourcePoints[sinterval[ax.index], 2]))
                elif tu == '' and xu != '' and yu != '' and zu != '':
                    ax.set_title('''Test Function [$\\tau$ = %s, $\\bf{z}$ @ (%0.2f %s, %0.2f %s, %0.2f %s)]'''
                                 %(tau,
                                   sourcePoints[sinterval[ax.index], 0], xu,
                                   sourcePoints[sinterval[ax.index], 1], yu,
                                   sourcePoints[sinterval[ax.index], 2], zu))
                elif tu == '' and xu == '' and yu == '' and zu == '':
                    ax.set_title('''Test Function [$\\tau$ = %s, $\\bf{z}$ @ (%0.2f, %0.2f, %0.2f)]'''
                                 %(tau,
                                   sourcePoints[sinterval[ax.index], 0],
                                   sourcePoints[sinterval[ax.index], 1],
                                   sourcePoints[sinterval[ax.index], 2]))
                       
        if tu != '':
            ax.set_xlabel('Time (%s)' %(tu))
        else:
            ax.set_xlabel('Time')
        
        if tstart != recordingTimes[0] or tstop != recordingTimes[-1]:
            ax.axvspan(tstart, tstop, alpha=0.25, color='silver')
        ax.set_xlim([recordingTimes[0], recordingTimes[-1]])
        
        return ax
    #==============================================================================
    def previous_source(ax, args, rinterval, sinterval, receiverPoints, sourcePoints, scatterer):
        ax.index = (ax.index - 1) % sourcePoints[sinterval].shape[0]  # wrap around using %
        map_plot(ax, ax.index, args, rinterval, sinterval, receiverPoints, sourcePoints, scatterer)
    
    def next_source(ax, args, rinterval, sinterval, receiverPoints, sourcePoints, scatterer):
        ax.index = (ax.index + 1) % sourcePoints[sinterval].shape[0]  # wrap around using %
        map_plot(ax, ax.index, args, rinterval, sinterval, receiverPoints, sourcePoints, scatterer)
        
    def map_plot(ax, index, args, rinterval, sinterval, receiverPoints, sourcePoints, scatterer):
        ax.clear()
        
        # delete the row corresponding to the current source (plot current source separately)        
        sources = sourcePoints[sinterval, :]
        sources = np.delete(sources, index, axis=0)
        currentSource = sourcePoints[sinterval[index], :]
        if receiverPoints.shape[1] == 2:
            if args.type == 'data':
                ax.plot(receiverPoints[rinterval, 0], receiverPoints[rinterval, 1], 'v', color='k')
                ax.plot(sources[:, 0], sources[:, 1], '*', color='silver')
                ax.plot(currentSource[0], currentSource[1], marker='*', markersize=12, color='darkcyan')
            elif args.type == 'testfunc':
                ax.plot(receiverPoints[rinterval + pltrstart, 0],
                        receiverPoints[rinterval + pltrstart, 1], 'v', color='k')
                ax.plot(sources[:, 0], sources[:, 1], '.', color='silver')
                ax.plot(currentSource[0], currentSource[1], marker='.', markersize=12, color='darkcyan')
            if scatterer is not None and showScatterer:
                ax.plot(scatterer[:, 0], scatterer[:, 1], '--', color='darkgray')
                
            if xu != '':
                ax.set_xlabel(xlabel + ' (%s)' %(xu))
            else:
                ax.set_xlabel(xlabel)

            if yu != '':
                ax.set_ylabel(ylabel + ' (%s)' %(yu))
            else:
                ax.set_ylabel(ylabel)
                
        elif receiverPoints.shape[1] == 3:
            if args.type == 'data':
                ax.plot(receiverPoints[rinterval, 0], receiverPoints[rinterval, 1], receiverPoints[rinterval, 2], 'v', color='k')
                ax.plot(sources[:, 0], sources[:, 1], sources[:, 2], '*', color='silver')
                ax.plot(currentSource[0], currentSource[1], currentSource[2], marker='*', markersize=12, color='darkcyan')
            elif args.type == 'testfunc':
                ax.plot(receiverPoints[rinterval + pltrstart, 0],
                        receiverPoints[rinterval + pltrstart, 1],
                        receiverPoints[rinterval + pltrstart, 2], 'v', color='k')
                ax.plot(sources[:, 0], sources[:, 1], sources[:, 2], '.', color='silver')
                ax.plot(currentSource[0], currentSource[1], currentSource[2], marker='.', markersize=12, color='darkcyan')
            if scatterer is not None and showScatterer:
                ax.plot(scatterer[:, 0], scatterer[:, 1], scatterer[:, 2], '--', color='darkgray')
                
            if xu != '':
                ax.set_xlabel(xlabel + ' (%s)' %(xu))
            else:
                ax.set_xlabel(xlabel)

            if yu != '':
                ax.set_ylabel(ylabel + ' (%s)' %(yu))
            else:
                ax.set_ylabel(ylabel)
                
            if zu != '':
                ax.set_zlabel(zlabel + ' (%s)' %(zu))
            else:
                ax.set_zlabel(zlabel)
        
        ax.set_title('Map')
        ax.set_aspect('equal')
        if invertX:
            ax.invert_xaxis()
        if invertY:
            ax.invert_yaxis()
        if invertZ:
            ax.invert_zaxis()
        
        return ax
    #==============================================================================
    if Path('window.npz').exists():
        windowDict = np.load('window.npz')
        tstart = windowDict['tstart']
        tstop = windowDict['tstop']
        
        if args.type == 'data':
            # Set the receiver window
            rstart = windowDict['rstart']
            rstop = windowDict['rstop']
            rstep = windowDict['rstep']
            
            # Set the source window
            sstart = windowDict['sstart']
            sstop = windowDict['sstop']
            sstep = windowDict['sstep']
            
        elif args.type == 'testfunc':
            # Set the receiver window
            rstart = 0
            rstop = X.shape[0]
            rstep = 1
            
            # pltrstart is used to plot the correct receivers for
            # the simulated test function computed by Vezda
            pltrstart = windowDict['rstart']
            
            # Set the source window
            sstart = 0
            sstop = X.shape[2]
            sstep = 1
        
    else:
        tstart = recordingTimes[0]
        tstop = recordingTimes[-1]
        
        rstart = 0
        pltrstart = rstart
        rstop = X.shape[0]
        rstep = 1
        
        sstart = 0
        sstop = X.shape[2]
        sstep = 1
        
    rinterval = np.arange(rstart, rstop, rstep)
    sinterval = np.arange(sstart, sstop, sstep)
    X = X[rinterval, :, :]
    X = X[:, :, sinterval]
    Ns = X.shape[2]
    
    remove_keymap_conflicts({'left', 'right', 'up', 'down', 'save'})
    if showMap:
        fig = plt.figure(figsize=plt.figaspect(0.48))
        ax1 = fig.add_subplot(121)
        ax1.volume = X
        ax1.index = Ns // 2
        wiggle_plot(ax1, X[:, :, ax1.index], tstart, tstop, rinterval, sinterval, args, tau,
                    recordingTimes, receiverPoints, sourcePoints)
        
        if receiverPoints.shape[1] == 2:
            ax2 = fig.add_subplot(122)
        elif receiverPoints.shape[1] == 3:
            ax2 = fig.add_subplot(122, projection='3d')   
        
        ax2.index = ax1.index
        map_plot(ax2, ax2.index, args, rinterval, sinterval, receiverPoints, sourcePoints, scatterer)
        plt.tight_layout()
        fig.canvas.mpl_connect('key_press_event', lambda event: process_key(event, tstart, tstop, rinterval, sinterval, 
                                                                       receiverPoints, sourcePoints, scatterer,
                                                                       args, tau, recordingTimes))
    
    else:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.volume = X
        ax.index = Ns // 2
        wiggle_plot(ax, X[:, :, ax.index], tstart, tstop, rinterval, sinterval, args, tau,
                    recordingTimes, receiverPoints, sourcePoints)
        plt.tight_layout()
        fig.canvas.mpl_connect('key_press_event', lambda event: process_key(event, tstart, tstop, rinterval, sinterval, 
                                                                       receiverPoints, sourcePoints, scatterer,
                                                                       args, tau, recordingTimes))
    
    plt.show()
