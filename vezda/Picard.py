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
from vezda.plot_utils import (remove_keymap_conflicts, default_params, setFigure)
#import matplotlib
#matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from pathlib import Path

def PicardPlot(ax, s, c, d):
    '''
    s : array of nonincreasing singular values
    c : array of coefficients |<u_n, b>|
    d : array of coefficients for the solution |<u_n, b>| / sigma_n
    '''
    
    ax.plot(s, '.', clip_on=False, markersize=9, label='$\sigma_n$', color=ax.pointcolor)
    ax.plot(c, 'dm', clip_on=False, alpha=ax.alpha,
            label=r'$\vert\langle \phi_n, u^i_{\mathbf{z},\tau} \rangle\vert$')
    ax.plot(d, 'Pc', clip_on=False, markersize=7, alpha=ax.alpha,
            label=r'$\vert\langle \phi_n, u^i_{\mathbf{z},\tau} \rangle\vert/\sigma_n$')
    
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.175),
          ncol=3, fancybox=True, shadow=True, fontsize='large')
    ax.set_xlabel('n', color=ax.labelcolor)
    ax.set_xlim([0, len(s)])
    ax.set_ylim(bottom=0)
    ax.locator_params(axis='y', nticks=6)
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    return ax

def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--format', '-f', type=str, default='pdf', choices=['png', 'pdf', 'ps', 'eps', 'svg'],
                        help='''specify the image format of the saved file. Accepted formats are png, pdf,
                        ps, eps, and svg. Default format is set to pdf.''')
    parser.add_argument('--mode', type=str, choices=['light', 'dark'], required=False,
                        help='''Specify whether to view plots in light mode for daytime viewing
                        or dark mode for nighttime viewing.
                        Mode must be either \'light\' or \'dark\'.''')
    args = parser.parse_args()
    #==============================================================================
                
    def process_key_picard(event, tstart, tstop, rinterval,
                           receiverPoints, scatterer,
                           args, recordingTimes):
        if args.map:
            fig = event.canvas.figure
            ax1 = fig.axes[0]
            ax2 = fig.axes[1]
            if event.key == 'left' or event.key == 'down':
                previous_slice(ax1, tstart, tstop, rinterval, args,
                               recordingTimes, receiverPoints)
                previous_source(ax2, tstart, tstop, rinterval, args,
                                recordingTimes, receiverPoints)
            elif event.key == 'right' or event.key == 'up':
                next_slice(ax1, tstart, tstop, rinterval, args,
                           recordingTimes, receiverPoints)
                next_source(ax1, tstart, tstop, rinterval, args,
                            recordingTimes, receiverPoints)
        else:
            fig = event.canvas.figure
            ax = fig.axes[0]
            if event.key == 'left' or event.key == 'down':
                previous_slice(ax, tstart, tstop, rinterval, args,
                               recordingTimes, receiverPoints)
            elif event.key == 'right' or event.key == 'up':
                next_slice(ax, tstart, tstop, rinterval, args,
                           recordingTimes, receiverPoints)
        fig.canvas.draw()
    
    #==============================================================================
    try:
        s = np.load('singularValues.npy')
        U = np.load('leftVectors.npy')
    except FileNotFoundError:
        s = None
        U = None
        
    if any(v is None for v in [s, U]):
        sys.exit(textwrap.dedent(
                '''
                A singular-value decomposition needs to be computed before any
                spectral analysis can be performed. Enter:
                    
                    vzsvd --help
                    
                from the command line for more information on how to compute
                a singular-value decomposition.
                '''))
            
    try:
        TFDict = np.load('VZTestFuncs.npz')
    except FileNotFoundError:
        TFDict = None
        
    if TFDict is None:
        sys.exit(textwrap.dedent(
                '''
                '''))
    
    k = s.size
    s = np.reshape(s, (k, 1))
    # Compute coefficients for Picard plot    
    TFarray = TFDict['TFarray']
    TF = TFarray[:, :, 0, 0]
    Nr, Nt = TF.shape
    b = np.reshape(TF, (Nt * Nr, 1))
    c = np.abs(U.T @ b)
    d = np.divide(c, s)
    
    #==============================================================================
    datadir = np.load('datadir.npz')
    receiverPoints = np.load(str(datadir['receivers']))
    sourcePoints = np.load(str(datadir['sources']))
    if 'scatterer' in datadir:
        scatterer = np.load(str(datadir['scatterer']))
    else:
        scatterer = None
        
    if Path('window.npz').exists():
        windowDict = np.load('window.npz')
        
        # Set the receiver window for receiverPoints
        rstart = windowDict['rstart']
        rstop = windowDict['rstop']
        rstep = windowDict['rstep']
        
        # Set the source window for sourcePoints
        sstart = windowDict['sstart']
        sstop = windowDict['sstop']
        sstep = windowDict['sstep']
        
    else:
        
        rstart = 0
        rstop = Nr
        rstep = 1
        
        sstart = 0
        sstop = sourcePoints.shape[0]
        sstep = 1
        
    # pltrstart is used to plot the correct receivers for
    # the simulated test function computed by Vezda
    pltrstart = rstart
    pltsstart = sstart
    
    rinterval = np.arange(rstart, rstop, rstep)
    receiverPoints = receiverPoints[rinterval, :]
    
    sinterval = np.arange(sstart, sstop, sstep)
    sourcePoints = sourcePoints[sinterval, :]
    
    #==============================================================================
    remove_keymap_conflicts({'left', 'right', 'up', 'down', 'save'})
    
    if Path('plotParams.pkl').exists():
        plotParams = pickle.load(open('plotParams.pkl', 'rb'))
    else:
        plotParams = default_params()
        
    if args.mode is not None:
        plotParams['view_mode'] = args.mode
        pickle.dump(plotParams, open('plotParams.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)
        
    
    fig, ax = setFigure(num_axes=1, mode=plotParams['view_mode'])
    
#        ax1.index = k // 2
#        PicardPlot(ax1, s, c, d)
        
#        if receiverPoints.shape[1] == 2:
#            ax2 = fig.add_subplot(122)
#        elif receiverPoints.shape[1] == 3:
#            ax2 = fig.add_subplot(122, projection='3d')   
        
#        ax2.index = ax1.index
#        map_plot(ax2, ax2.index, args, rinterval, receiverPoints, sourcePoints, scatterer)
#        plt.tight_layout()
#        #fig.canvas.mpl_connect('key_press_event', lambda event: process_key(event, tstart, tstop, rinterval, 
#        #                                                                    receiverPoints, sourcePoints, scatterer,
#        #                                                                    args, recordingTimes))
    ax.index = k // 2
    PicardPlot(ax, s, c, d)
    plt.tight_layout()
    fig.savefig('Picard.' + args.format, format=args.format, bbox_inches='tight')
    plt.show()
