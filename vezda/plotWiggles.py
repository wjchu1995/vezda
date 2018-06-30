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
    args = parser.parse_args()
    #==============================================================================
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
            
        if args.title is not None:
            if args.type == 'data':
                data_title = args.title
                plotParams['data_title'] = data_title
            elif args.type == 'testfunc':
                tf_title = args.title
                plotParams['tf_title'] = tf_title
        else:
            if args.type == 'data':
                data_title = plotParams['data_title']
            elif args.type == 'testfunc':
                tf_title = plotParams['tf_title']
        
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
        plotParams['colormap'] = 'magma'
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
        
        if args.title is not None:
            if args.type == 'data':
                data_title = args.title
                tf_title = 'Test Function'
            elif args.type == 'testfunc':
                data_title = 'Data'
                tf_title = args.title
        else:
            data_title = 'Data'
            tf_title = 'Test Function'
        plotParams['data_title'] = data_title
        plotParams['tf_title'] = tf_title
        
        pickle.dump(plotParams, open('plotParams.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)
        
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
                    args, recordingTimes):
        if args.map:
            fig = event.canvas.figure
            ax1 = fig.axes[0]
            ax2 = fig.axes[1]
            if event.key == 'left' or event.key == 'down':
                previous_slice(ax1, tstart, tstop, rinterval, sinterval, args,
                               recordingTimes, receiverPoints, sourcePoints)
                previous_source(ax2, args, rinterval, receiverPoints, sourcePoints, scatterer)
            elif event.key == 'right' or event.key == 'up':
                next_slice(ax1, tstart, tstop, rinterval, sinterval, args,
                           recordingTimes, receiverPoints, sourcePoints)
                next_source(ax2, args, rinterval, receiverPoints, sourcePoints, scatterer)
        else:
            fig = event.canvas.figure
            ax = fig.axes[0]
            if event.key == 'left' or event.key == 'down':
                previous_slice(ax, tstart, tstop, rinterval, sinterval, args,
                               recordingTimes, receiverPoints, sourcePoints)
            elif event.key == 'right' or event.key == 'up':
                next_slice(ax, tstart, tstop, rinterval, sinterval, args,
                           recordingTimes, receiverPoints, sourcePoints)
        fig.canvas.draw()
    #==============================================================================    
    def previous_slice(ax, tstart, tstop, rinterval, sinterval, args,
                       recordingTimes, receiverPoints, sourcePoints):
        volume = ax.volume
        ax.index = (ax.index - 1) % volume.shape[2]  # wrap around using %
        wiggle_plot(ax, volume[:, :, ax.index], tstart, tstop, rinterval, sinterval,
                    args, recordingTimes, receiverPoints, sourcePoints)
    
    def next_slice(ax, tstart, tstop, rinterval, sinterval, args,
                   recordingTimes, receiverPoints, sourcePoints):
        volume = ax.volume
        ax.index = (ax.index + 1) % volume.shape[2]
        wiggle_plot(ax, volume[:, :, ax.index], tstart, tstop, rinterval, sinterval,
                    args, recordingTimes, receiverPoints, sourcePoints)
        
    def wiggle_plot(ax, X, tstart, tstop, rinterval, sinterval, args,
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
                                        receiverPoints[0, 0], xu,
                                        receiverPoints[0, 1], yu))
                    elif au == '' and xu != '' and yu != '':
                        ax.set_ylabel('''Amplitude [Receiver %s @ (%0.2f %s, %0.2f %s)]'''
                                      %(pltrstart,
                                        receiverPoints[0, 0], xu,
                                        receiverPoints[0, 1], yu))
                    elif au != '' and xu == '' and yu == '':
                        ax.set_ylabel('''Amplitude (%s) [Receiver %s @ (%0.2f, %0.2f)]'''
                                      %(au, pltrstart,
                                        receiverPoints[0, 0],
                                        receiverPoints[0, 1]))
                    elif au == '' and xu == '' and yu == '':
                        ax.set_ylabel('''Amplitude [Receiver %s @ (%0.2f, %0.2f)]'''
                                      %(pltrstart,
                                        receiverPoints[0, 0],
                                        receiverPoints[0, 1]))
                else: #args.type == 'data'
                    if au != '' and xu != '' and yu != '':
                        ax.set_ylabel('''Amplitude (%s) [Receiver %s @ (%0.2f %s, %0.2f %s)]'''
                                      %(au, rstart,
                                        receiverPoints[0, 0], xu,
                                        receiverPoints[0, 1], yu))
                    elif au == '' and xu != '' and yu != '':
                        ax.set_ylabel('''Amplitude [Receiver %s @ (%0.2f %s, %0.2f %s)]'''
                                      %(rstart,
                                        receiverPoints[0, 0], xu,
                                        receiverPoints[0, 1], yu))
                    elif au != '' and xu == '' and yu == '':
                        ax.set_ylabel('''Amplitude (%s) [Receiver %s @ (%0.2f, %0.2f)]'''
                                      %(au, rstart,
                                        receiverPoints[0, 0],
                                        receiverPoints[0, 1]))
                    elif au == '' and xu == '' and yu == '':
                        ax.set_ylabel('''Amplitude [Receiver %s @ (%0.2f, %0.2f)]'''
                                      %(rstart,
                                        receiverPoints[0, 0],
                                        receiverPoints[0, 1]))
            
            elif receiverPoints.shape[1] == 3:
                if args.type == 'testfunc':
                    if au != '' and xu != '' and yu != '' and zu != '':
                        ax.set_ylabel('''Amplitude (%s) [Receiver %s @ (%0.2f %s, %0.2f %s, %0.2f %s)]'''
                                      %(au, pltrstart,
                                        receiverPoints[0, 0], xu,
                                        receiverPoints[0, 1], yu,
                                        receiverPoints[0, 2], zu))
                    elif au == '' and xu != '' and yu != '' and zu != '':
                        ax.set_ylabel('''Amplitude [Receiver %s @ (%0.2f %s, %0.2f %s, %0.2f %s)]'''
                                      %(pltrstart,
                                        receiverPoints[0, 0], xu,
                                        receiverPoints[0, 1], yu,
                                        receiverPoints[0, 2], zu))
                    elif au != '' and xu == '' and yu == '' and zu == '':
                        ax.set_ylabel('''Amplitude (%s) [Receiver %s @ (%0.2f, %0.2f, %0.2f)]'''
                                      %(au, pltrstart,
                                        receiverPoints[0, 0],
                                        receiverPoints[0, 1],
                                        receiverPoints[0, 2]))
                    elif au == '' and xu == '' and yu == '' and zu == '':
                        ax.set_ylabel('''Amplitude [Receiver %s @ (%0.2f, %0.2f, %0.2f)]'''
                                      %(pltrstart,
                                        receiverPoints[0, 0],
                                        receiverPoints[0, 1],
                                        receiverPoints[0, 2]))
                else: #args.type == 'data'
                    if au != '' and xu != '' and yu != '' and zu != '':
                        ax.set_ylabel('''Amplitude (%s) [Receiver %s @ (%0.2f %s, %0.2f %s, %0.2f %s)]'''
                                      %(au, rinterval[0],
                                        receiverPoints[0, 0], xu,
                                        receiverPoints[0, 1], yu,
                                        receiverPoints[0, 2], zu))
                    elif au == '' and xu != '' and yu != '' and zu != '':
                        ax.set_ylabel('''Amplitude [Receiver %s @ (%0.2f %s, %0.2f %s, %0.2f %s)]'''
                                      %(rinterval[0],
                                        receiverPoints[0, 0], xu,
                                        receiverPoints[0, 1], yu,
                                        receiverPoints[0, 2], zu))
                    elif au != '' and xu == '' and yu == '' and zu == '':
                        ax.set_ylabel('''Amplitude (%s) [Receiver %s @ (%0.2f, %0.2f, %0.2f)]'''
                                      %(au, rinterval[0],
                                        receiverPoints[0, 0],
                                        receiverPoints[0, 1],
                                        receiverPoints[0, 2]))
                    elif au == '' and xu == '' and yu == '' and zu == '':
                        ax.set_ylabel('''Amplitude [Receiver %s @ (%0.2f, %0.2f, %0.2f)]'''
                                      %(rinterval[0],
                                        receiverPoints[0, 0],
                                        receiverPoints[0, 1],
                                        receiverPoints[0, 2]))
                       
            ax.plot(time, X[0, :], 'darkgray')
            ax.fill_between(time, 0, X[0, :], where=(X[0, :] > 0), color='m')
            ax.fill_between(time, 0, X[0, :], where=(X[0, :] < 0), color='c')
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            
        if sourcePoints.shape[1] == 2:
            if args.type == 'data':
                if  xu != '' and yu != '':
                    ax.set_title('''%s [Source %s @ (%0.2f %s, %0.2f %s)]'''
                                 %(data_title, sinterval[ax.index],
                                   sourcePoints[ax.index, 0], xu,
                                   sourcePoints[ax.index, 1], yu))
                else:
                    ax.set_title('''%s [Source %s @ (%0.2f, %0.2f)]'''
                                 %(data_title, sinterval[ax.index],
                                   sourcePoints[ax.index, 0],
                                   sourcePoints[ax.index, 1]))
                
            elif args.type == 'testfunc':
                if xu != '' and yu != '':
                    ax.set_title('''%s [$\\bf{z}$ @ (%0.2f %s, %0.2f %s)]'''
                                 %(tf_title, sourcePoints[ax.index, 0], xu,
                                   sourcePoints[ax.index, 1], yu))
                elif xu == '' and yu == '':
                    ax.set_title('''%s [$\\bf{z}$ @ (%0.2f, %0.2f)]'''
                                 %(tf_title, sourcePoints[ax.index, 0],
                                   sourcePoints[ax.index, 1]))
                    
        elif sourcePoints.shape[1] == 3:
            if args.type == 'data':
                if  xu != '' and yu != '' and zu != '':
                    ax.set_title('''%s [Source %s @ (%0.2f %s, %0.2f %s, %0.2f %s)]'''
                                 %(data_title, sinterval[ax.index],
                                   sourcePoints[ax.index, 0], xu,
                                   sourcePoints[ax.index, 1], yu,
                                   sourcePoints[ax.index, 2], zu))
                else:
                    ax.set_title('''%s [Source %s @ (%0.2f, %0.2f, %0.2f)]'''
                                 %(data_title, sinterval[ax.index],
                                   sourcePoints[ax.index, 0],
                                   sourcePoints[ax.index, 1],
                                   sourcePoints[ax.index, 2]))
                
            elif args.type == 'testfunc':
                if xu != '' and yu != '' and zu != '':
                    ax.set_title('''%s [$\\bf{z}$ @ (%0.2f %s, %0.2f %s, %0.2f %s)]'''
                                 %(tf_title, sourcePoints[ax.index, 0], xu,
                                   sourcePoints[ax.index, 1], yu,
                                   sourcePoints[ax.index, 2], zu))
                elif xu == '' and yu == '' and zu == '':
                    ax.set_title('''%s [$\\bf{z}$ @ (%0.2f, %0.2f, %0.2f)]'''
                                 %(tf_title, sourcePoints[ax.index, 0],
                                   sourcePoints[ax.index, 1],
                                   sourcePoints[ax.index, 2]))
                       
        if tu != '':
            ax.set_xlabel('Time (%s)' %(tu))
        else:
            ax.set_xlabel('Time')
        
        if tstart != recordingTimes[0] or tstop != recordingTimes[-1]:
            ax.axvspan(tstart, tstop, alpha=0.25, color='silver')
        ax.set_xlim([recordingTimes[0], recordingTimes[-1]])
        
        return ax
    #==============================================================================
    def previous_source(ax, args, rinterval, receiverPoints, sourcePoints, scatterer):
        ax.index = (ax.index - 1) % sourcePoints.shape[0]  # wrap around using %
        map_plot(ax, ax.index, args, rinterval, receiverPoints, sourcePoints, scatterer)
    
    def next_source(ax, args, rinterval, receiverPoints, sourcePoints, scatterer):
        ax.index = (ax.index + 1) % sourcePoints.shape[0]  # wrap around using %
        map_plot(ax, ax.index, args, rinterval, receiverPoints, sourcePoints, scatterer)
        
    def map_plot(ax, index, args, rinterval, receiverPoints, sourcePoints, scatterer):
        ax.clear()
        
        # delete the row corresponding to the current source (plot current source separately)
        sources = np.delete(sourcePoints, index, axis=0)
        currentSource = sourcePoints[index, :]
        if receiverPoints.shape[1] == 2:
            ax.plot(receiverPoints[:, 0], receiverPoints[:, 1], 'v', color='k')
            if args.type == 'data':
                ax.plot(sources[:, 0], sources[:, 1], '*', color='silver')
                ax.plot(currentSource[0], currentSource[1], marker='*', markersize=12, color='darkcyan')
            elif args.type == 'testfunc':
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
            ax.plot(receiverPoints[:, 0], receiverPoints[:, 1], receiverPoints[:, 2], 'v', color='k')
            if args.type == 'data':
                ax.plot(sources[:, 0], sources[:, 1], sources[:, 2], '*', color='silver')
                ax.plot(currentSource[0], currentSource[1], currentSource[2], marker='*', markersize=12, color='darkcyan')
            elif args.type == 'testfunc':
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
    datadir = np.load('datadir.npz')
    recordingTimes = np.load(str(datadir['recordingTimes']))
    receiverPoints = np.load(str(datadir['receivers']))
    if 'scatterer' in datadir:
        scatterer = np.load(str(datadir['scatterer']))
    else:
        scatterer = None
        
    if args.type == 'data':
        # load the 3D data array into variable 'X'
        # X[receiver, time, source]
        X = np.load(str(datadir['scatteredData']))
        time = recordingTimes
        sourcePoints = np.load(str(datadir['sources']))
        
    elif args.type == 'testfunc':
        if 'testFuncs' in datadir and not Path('VZTestFuncs.npz').exists():
            TFtype = 'user'
            X = np.load(str(datadir['testFuncs']))
            time = recordingTimes
            samplingPoints = np.load(str(datadir['samplingPoints']))
            sourcePoints = samplingPoints[:, :-1]
            
        elif not 'testFuncs' in datadir and Path('VZTestFuncs.npz').exists():
            TFtype = 'vezda'
            testFuncs = np.load('VZTestFuncs.npz')
            TFarray = testFuncs['TFarray']
            X = TFarray[:, :, :, 0]
            time = testFuncs['time'] 
            samplingPoints = testFuncs['samplingPoints']
            # Extract all but last column of sampling points,
            # which corresponds to sampling points in time
            sourcePoints = samplingPoints[:, :-1]
            
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
                    TFtype = 'user'
                    X = np.load(str(datadir['testFuncs']))
                    time = recordingTimes
                    samplingPoints = np.load(str(datadir['samplingPoints']))
                    sourcePoints = samplingPoints[:, :-1]
                    userResponded = True
                    break
                elif answer == '2':
                    TFtype = 'vezda'
                    testFuncs = np.load('VZTestFuncs.npz')
                    TFarray = testFuncs['TFarray']
                    X = TFarray[:, :, :, 0]
                    time = testFuncs['time'] 
                    samplingPoints = testFuncs['samplingPoints']
                    # Extract all but last column of sampling points,
                    # which corresponds to sampling points in time
                    sourcePoints = samplingPoints[:, :-1]
                    userResponded = True
                elif answer == 'q' or answer == 'quit':
                    sys.exit('Exiting program.')
                else:
                    print('Invalid response. Please enter \'1\', \'2\', or \'q/quit\'.')
        
        else:
            sys.exit(textwrap.dedent(
                    '''
                    Error: No test functions have been found to plot.
                    '''))
        
    if Path('window.npz').exists():
        windowDict = np.load('window.npz')
        tstart = windowDict['tstart']
        tstop = windowDict['tstop']
        
        # Set the receiver window for receiverPoints
        rstart = windowDict['rstart']
        rstop = windowDict['rstop']
        rstep = windowDict['rstep']
        
        if args.type == 'data':
            # Window the receiver axis in the data volume X
            Xrstart = rstart
            Xrstop = rstop
            Xrstep = rstep
            
            # Set the source window
            sstart = windowDict['sstart']
            sstop = windowDict['sstop']
            sstep = windowDict['sstep']
            
        elif args.type == 'testfunc':
            
            if TFtype == 'user':
                Xrstart = rstart
                Xrstop = rstop
                Xrstep = rstep
                
            elif TFtype == 'vezda':
                # Window the receiver axis in the data volume X
                Xrstart = 0
                Xrstop = X.shape[0]
                Xrstep = 1
            
            # pltrstart is used to plot the correct receivers for
            # the simulated test function computed by Vezda
            pltrstart = rstart
            
            # Set the source window
            sstart = 0
            sstop = X.shape[2]
            sstep = 1
        
    else:
        tstart = recordingTimes[0]
        tstop = recordingTimes[-1]
        
        rstart = 0
        rstop = X.shape[0]
        rstep = 1
        
        Xrstart = rstart
        pltrstart = rstart
        Xrstop = rstop
        Xrstep = rstep
        
        sstart = 0
        sstop = X.shape[2]
        sstep = 1
        
    rinterval = np.arange(rstart, rstop, rstep)
    receiverPoints = receiverPoints[rinterval, :]
    
    sinterval = np.arange(sstart, sstop, sstep)
    sourcePoints = sourcePoints[sinterval, :]
    
    Xrinterval = np.arange(Xrstart, Xrstop, Xrstep)
    X = X[Xrinterval, :, :]
    X = X[:, :, sinterval]
    Ns = X.shape[2]
    
    remove_keymap_conflicts({'left', 'right', 'up', 'down', 'save'})
    if args.map:
        fig = plt.figure(figsize=plt.figaspect(0.48))
        ax1 = fig.add_subplot(121)
        ax1.volume = X
        ax1.index = Ns // 2
        wiggle_plot(ax1, X[:, :, ax1.index], tstart, tstop, rinterval, sinterval, args,
                    recordingTimes, receiverPoints, sourcePoints)
        
        if receiverPoints.shape[1] == 2:
            ax2 = fig.add_subplot(122)
        elif receiverPoints.shape[1] == 3:
            ax2 = fig.add_subplot(122, projection='3d')   
        
        ax2.index = ax1.index
        map_plot(ax2, ax2.index, args, rinterval, receiverPoints, sourcePoints, scatterer)
        plt.tight_layout()
        fig.canvas.mpl_connect('key_press_event', lambda event: process_key(event, tstart, tstop, rinterval, sinterval, 
                                                                       receiverPoints, sourcePoints, scatterer,
                                                                       args, recordingTimes))
    
    else:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.volume = X
        ax.index = Ns // 2
        wiggle_plot(ax, X[:, :, ax.index], tstart, tstop, rinterval, sinterval, args,
                    recordingTimes, receiverPoints, sourcePoints)
        plt.tight_layout()
        fig.canvas.mpl_connect('key_press_event', lambda event: process_key(event, tstart, tstop, rinterval, sinterval, 
                                                                       receiverPoints, sourcePoints, scatterer,
                                                                       args, recordingTimes))
    
    plt.show()
