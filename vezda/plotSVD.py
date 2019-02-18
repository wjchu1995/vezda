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
from pathlib import Path
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from vezda.data_utils import get_user_windows
from vezda.math_utils import nextPow2
from vezda.svd_utils import load_svd
from vezda.plot_utils import (vector_title, remove_keymap_conflicts, plotWiggles,
                              plotFreqVectors, process_key_vectors, default_params, setFigure)
import numpy as np
import pickle
from vezda.plot_utils import FontColor

def info():
    commandName = FontColor.BOLD + 'vzsvd:' + FontColor.END
    description = ' plot singular-value decompositions of linear operators'
    
    return commandName + description

#==============================================================================
def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nfo', action='store_true',
                        help='''Plot the singular-value decomposition of the
                        near-field operator (NFO).''')
    parser.add_argument('--lso', action='store_true',
                        help='''Plot the singular-value decomposition of the
                        Lippmann-Schwinger operator (LSO).''')
    parser.add_argument('--format', '-f', type=str, default='pdf', choices=['png', 'pdf', 'ps', 'eps', 'svg'],
                        help='''Specify the image format of the saved file. Accepted formats are png, pdf,
                        ps, eps, and svg. Default format is set to pdf.''')
    parser.add_argument('--mode', type=str, choices=['light', 'dark'], required=False,
                        help='''Specify whether to view plots in light mode for daytime viewing
                        or dark mode for nighttime viewing.
                        Mode must be either \'light\' or \'dark\'.''')
    args = parser.parse_args()
    
    # See if an SVD already exists. If so, attempt to load it...
    if args.nfo and not args.lso:
        operatorName = 'near-field operator'
        filename = 'NFO_SVD.npz'
    
    elif not args.nfo and args.lso:
        operatorName = 'Lippmann-Schwinger operator'
        filename = 'LSO_SVD.npz'
            
    elif args.nfo and args.lso:
        sys.exit(textwrap.dedent(
                '''
                UsageError: Please specify only one of the arguments \'--nfo\' or \'--lso\'.
                '''))
    
    else:
        sys.exit(textwrap.dedent(
                '''
                For which operator would you like to plot a singular-value decomposition?
                Enter:
                    
                    vzsvd --nfo
                
                for the near-field operator or
                
                    vzsvd --lso
                    
                for the Lippmann-Schwinger operator.
                '''))
            
    try:
        U, s, Vh = load_svd(filename)
    except IOError:
        sys.exit(textwrap.dedent(
                '''
                A singular-value decomposition of the {s} does not exist.
                '''.format(s=operatorName)))

    #==============================================================================
    # Read in data files 
    #==============================================================================
    datadir = np.load('datadir.npz')
    receiverPoints = np.load(str(datadir['receivers']))
    recordingTimes = np.load(str(datadir['recordingTimes']))
    
    # Apply user-specified windows
    rinterval, tinterval, tstep, dt, sinterval = get_user_windows()
    receiverPoints = receiverPoints[rinterval, :]
    recordingTimes = recordingTimes[tinterval]
    
    # Load appropriate source points and source window
    if args.nfo:    # Near-field operator                
        if 'sources' in datadir:
            sourcePoints = np.load(str(datadir['sources']))
            sourcePoints = sourcePoints[sinterval, :]
        else:
            sourcePoints = None
            
    else:
        # if args.lso (Lippmann-Schwinger operator)
            
        # in the case of the Lippmann-Schwinger operator, 'sourcePoints'
        # correspond to sampling points, which should always exist.
        if 'testFuncs' in datadir:
            sourcePoints = np.load(str(datadir['samplingPoints']))
                
        elif Path('VZTestFuncs.npz').exists():
            TFDict = np.load('VZTestFuncs.npz')
            sourcePoints = TFDict['samplingPoints']
        
        else:
            sys.exit(textwrap.dedent(
                    '''
                    Error: A sampling grid must exist and test functions computed
                    before a singular-value decomposition of the Lippmann-Schwinger
                    operator can be computed or plotted.
                    '''))
    
        # update sinterval for test functions
        sinterval = np.arange(0, sourcePoints.shape[0], 1)   
        
    # increment receiver/source intervals to be consistent with
    # one-based indexing (i.e., count from one instead of zero)
    rinterval += 1
    sinterval += 1
    
    #==============================================================================
    # Determine whether to plot SVD in time domain or frequency domain 
    #==============================================================================
    if np.issubdtype(U.dtype, np.complexfloating):
        domain = 'freq'
    else:
        domain = 'time'
    
    # Load plot parameters
    if Path('plotParams.pkl').exists():
        plotParams = pickle.load(open('plotParams.pkl', 'rb'))
    else:
        plotParams = default_params()
        
    Nr = receiverPoints.shape[0]
    Nt = len(recordingTimes)
    k = len(s)
                
    if domain == 'freq':
        # plot singular vectors in frequency domain 
        N = nextPow2(2 * Nt)
        freqs = np.fft.rfftfreq(N, tstep * dt)
            
        if plotParams['fmax'] is None:
            plotParams['fmax'] = np.max(freqs)
            
        # Apply the frequency window
        fmin = plotParams['fmin']
        fmax = plotParams['fmax']
        df = 1.0 / (N * tstep * dt)
            
        startIndex = int(round(fmin / df))
        stopIndex = int(round(fmax / df))
        finterval = np.arange(startIndex, stopIndex, 1)
        freqs = freqs[finterval]
        
        M = len(freqs)         
        Ns = int(Vh.shape[1] / M)
        U = U.toarray().reshape((Nr, M, k))
        V = Vh.getH().toarray().reshape((Ns, M, k))
            
    else: # domain == 'time'
        M = 2 * Nt - 1
        Ns = int(Vh.shape[1] / M)
        U = U.reshape((Nr, M, k))
        V = Vh.T.reshape((Ns, M, k))
        T = recordingTimes[-1] - recordingTimes[0]
        times = np.linspace(-T, T, M)
        
    if args.mode is not None:
        plotParams['view_mode'] = args.mode
        
    pickle.dump(plotParams, open('plotParams.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)
        
    remove_keymap_conflicts({'left', 'right', 'up', 'down', 'save'})
    if domain == 'freq':
            
        # plot the left singular vectors
        fig_lvec, ax_lvec_r, ax_lvec_i = setFigure(num_axes=2, mode=plotParams['view_mode'])
        ax_lvec_r.volume = U.real
        ax_lvec_i.volume = U.imag
        ax_lvec_r.index = 0
        ax_lvec_i.index = 0
        fig_lvec.suptitle('Left-Singular Vector', color=ax_lvec_r.titlecolor, fontsize=16)
        fig_lvec.subplots_adjust(bottom=0.27, top=0.86)
        leftTitle_r = vector_title('left', ax_lvec_r.index + 1, 'real')
        leftTitle_i = vector_title('left', ax_lvec_i.index + 1, 'imag')
        for ax, title in zip([ax_lvec_r, ax_lvec_i], [leftTitle_r, leftTitle_i]):
            left_im = plotFreqVectors(ax, ax.volume[:, :, ax.index], freqs, rinterval,
                                      receiverPoints, title, 'left', plotParams)
                
        lp0 = ax_lvec_r.get_position().get_points().flatten()
        lp1 = ax_lvec_i.get_position().get_points().flatten()
        left_cax = fig_lvec.add_axes([lp0[0], 0.12, lp1[2]-lp0[0], 0.03])
        lcbar = fig_lvec.colorbar(left_im, left_cax, orientation='horizontal')
        lcbar.outline.set_edgecolor(ax_lvec_r.cbaredgecolor)
        lcbar.ax.tick_params(axis='x', colors=ax_lvec_r.labelcolor)              
        lcbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        lcbar.set_label('Amplitude',
                        labelpad=5, rotation=0, fontsize=12, color=ax_lvec_r.labelcolor)
        fig_lvec.canvas.mpl_connect('key_press_event', lambda event: process_key_vectors(event, freqs, rinterval, sinterval,
                                                                                         receiverPoints, sourcePoints, plotParams,
                                                                                         'cmplx_left'))
            
        # plot the right singular vectors
        fig_rvec, ax_rvec_r, ax_rvec_i = setFigure(num_axes=2, mode=plotParams['view_mode'])
        ax_rvec_r.volume = V.real
        ax_rvec_i.volume = V.imag
        ax_rvec_r.index = 0
        ax_rvec_i.index = 0
        fig_rvec.suptitle('Right-Singular Vector', color=ax_rvec_r.titlecolor, fontsize=16)
        fig_rvec.subplots_adjust(bottom=0.27, top=0.86)
        rightTitle_r = vector_title('right', ax_rvec_r.index + 1, 'real')
        rightTitle_i = vector_title('right', ax_rvec_i.index + 1, 'imag')
        for ax, title in zip([ax_rvec_r, ax_rvec_i], [rightTitle_r, rightTitle_i]):
            right_im = plotFreqVectors(ax, ax.volume[:, :, ax.index], freqs, sinterval,
                                       sourcePoints, title, 'right', plotParams)
            
        rp0 = ax_rvec_r.get_position().get_points().flatten()
        rp1 = ax_rvec_i.get_position().get_points().flatten()
        right_cax = fig_rvec.add_axes([rp0[0], 0.12, rp1[2]-rp0[0], 0.03])
        rcbar = fig_rvec.colorbar(right_im, right_cax, orientation='horizontal')  
        rcbar.outline.set_edgecolor(ax_rvec_r.cbaredgecolor)
        rcbar.ax.tick_params(axis='x', colors=ax_rvec_r.labelcolor)
        rcbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        rcbar.set_label('Amplitude',
                        labelpad=5, rotation=0, fontsize=12, color=ax_lvec_r.labelcolor)
        fig_rvec.canvas.mpl_connect('key_press_event', lambda event: process_key_vectors(event, freqs, rinterval, sinterval,
                                                                                         receiverPoints, sourcePoints, plotParams,
                                                                                         'cmplx_right'))
            
    else:
        # domain == 'time'   
        fig_vec, ax_lvec, ax_rvec = setFigure(num_axes=2, mode=plotParams['view_mode'])
            
        ax_lvec.volume = U
        ax_lvec.index = 0
        leftTitle = vector_title('left', ax_lvec.index + 1)
        plotWiggles(ax_lvec, ax_lvec.volume[:, :, ax_lvec.index], times, rinterval,
                    receiverPoints, leftTitle, 'left', plotParams)
      
        ax_rvec.volume = V
        ax_rvec.index = 0
        rightTitle = vector_title('right', ax_rvec.index + 1)
        plotWiggles(ax_rvec, ax_rvec.volume[:, :, ax_rvec.index], times, sinterval,
                    sourcePoints, rightTitle, 'right', plotParams)
        fig_vec.tight_layout()
        fig_vec.canvas.mpl_connect('key_press_event', lambda event: process_key_vectors(event, times, rinterval, sinterval,
                                                                                        receiverPoints, sourcePoints, plotParams))
    #==============================================================================
    # plot the singular values
    # figure and axis for singular values
    fig_vals, ax_vals = setFigure(num_axes=1, mode=plotParams['view_mode'])
        
    n = np.arange(1, k + 1, 1)
    kappa = s[0] / s[-1]    # condition number = max(s) / min(s)
    ax_vals.plot(n, s, '.', clip_on=False, markersize=9, label=r'Condition Number: %0.1e' %(kappa), color=ax_vals.pointcolor)
    ax_vals.set_xlabel('n', color=ax_vals.labelcolor)
    ax_vals.set_ylabel('$\sigma_n$', color=ax_vals.labelcolor)
    legend = ax_vals.legend(title='Singular Values', loc='upper center', bbox_to_anchor=(0.5, 1.25),
                            markerscale=0, handlelength=0, handletextpad=0, fancybox=True, shadow=True,
                            fontsize='large')
    legend.get_title().set_fontsize('large')
    ax_vals.set_xlim([1, k])
    ax_vals.set_ylim(bottom=0)
    ax_vals.locator_params(axis='y', nticks=6)
    ax_vals.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    fig_vals.tight_layout()
    fig_vals.savefig('singularValues.' + args.format, format=args.format, bbox_inches='tight', facecolor=fig_vals.get_facecolor())
    
    plt.show()
