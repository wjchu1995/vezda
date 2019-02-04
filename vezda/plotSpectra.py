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
import numpy as np
from scipy.signal import tukey
from vezda.plot_utils import setFigure, default_params
from vezda.sampling_utils import samplingIsCurrent, sampleSpace
from vezda.signal_utils import compute_spectrum
import matplotlib.pyplot as plt
from pathlib import Path
import pickle

sys.path.append(os.getcwd())
import pulseFun
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
    # Load the recording times from the data directory
    datadir = np.load('datadir.npz')
    receiverPoints = np.load(str(datadir['receivers']))
    recordingTimes = np.load(str(datadir['recordingTimes']))
    dt = recordingTimes[1] - recordingTimes[0]
    
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
                
    # Apply the receiver window
    rinterval = np.arange(rstart, rstop, rstep)
    receiverPoints = receiverPoints[rinterval, :]

    # Apply the time window
    tinterval = np.arange(Tstart, Tstop, tstep)
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
        if Path('noisyData.npz').exists():
            userResponded = False
            print(textwrap.dedent(
                  '''
                  Detected that band-limited noise has been added to the data array.
                  Would you like to plot the amplitude/power spectrum of the noisy data? ([y]/n)
              
                  Enter 'q/quit' exit the program.
                  '''))
            while userResponded == False:
                answer = input('Action: ')
                if answer == '' or answer == 'y' or answer == 'yes':
                    print('Proceeding with noisy data...')
                    # read in the noisy data array
                    noisy = True
                    X = np.load('noisyData.npz')['noisyData']
                    userResponded = True
                elif answer == 'n' or answer == 'no':
                    print('Proceeding with noise-free data...')
                    # read in the recorded data array
                    noisy = False
                    X  = np.load(str(datadir['recordedData']))
                    userResponded = True
                elif answer == 'q' or answer == 'quit':
                    sys.exit('Exiting program.\n')
                else:
                    print('Invalid response. Please enter \'y/yes\', \'n\no\', or \'q/quit\'.')
                
        else:
            # read in the recorded data array
            noisy = False
            X = np.load(str(datadir['recordedData']))
    
        # Load the windowing parameters for the receiver and time axes of
        # the 3D data array
        if Path('window.npz').exists():
            print('Detected user-specified window:\n')
                
            # For display/printing purposes, count receivers with one-based
            # indexing. This amounts to incrementing the rstart parameter by 1
            print('window @ receivers : start =', rstart + 1)
            print('window @ receivers : stop =', rstop)
            print('window @ receivers : step =', rstep, '\n')
            
            tu = plotParams['tu']
            if tu != '':
                print('window @ time : start = %0.2f %s' %(tstart, tu))
                print('window @ time : stop = %0.2f %s' %(tstop, tu))
            else:
                print('window @ time : start =', tstart)
                print('window @ time : stop =', tstop)
            print('window @ time : step =', tstep, '\n')
                
            # Apply the source window
            slabel = windowDict['slabel']
            sstart = windowDict['sstart']
            sstop = windowDict['sstop']
            sstep = windowDict['sstep']
            sinterval = np.arange(sstart, sstop, sstep)
                
            # For display/printing purposes, count recordings/sources with one-based
            # indexing. This amounts to incrementing the sstart parameter by 1
            print('window @ %s : start = %s' %(slabel, sstart + 1))
            print('window @ %s : stop = %s' %(slabel, sstop))
            print('window @ %s : step = %s\n' %(slabel, sstep))                
                
            print('Applying window to data volume...')
            X = X[rinterval, :, :]
            X = X[:, tinterval, :]
            X = X[:, :, sinterval]
            
            # Apply tapered cosine (Tukey) window to time signals.
            # This ensures the fast fourier transform (FFT) used in
            # the definition of the matrix-vector product below is
            # acting on a function that is continuous at its edges.
                
            Nt = X.shape[1]
            peakFreq = pulseFun.peakFreq
            # Np : Number of samples in the dominant period T = 1 / peakFreq
            Np = int(round(1 / (tstep * dt * peakFreq)))
            # alpha is set to taper over 6 of the dominant period of the
            # pulse function (3 periods from each end of the signal)
            alpha = 6 * Np / Nt
            print('Tapering time signals with Tukey window: %d'
                  %(int(round(alpha * 100))) + '%')
            TukeyWindow = tukey(Nt, alpha)
            X *= TukeyWindow[None, :, None]
        
    elif not args.data and args.testfunc:
        
        if 'testFuncs' in datadir:
            print('Loading user-provided test functions...')
            X = np.load(str(datadir['testFuncs']))
            
            # Load the windowing parameters for the receiver and time axes of
            # the 3D data array
            if Path('window.npz').exists():
                print('Detected user-specified window:\n')
                
                # For display/printing purposes, count receivers with one-based
                # indexing. This amounts to incrementing the rstart parameter by 1
                print('window @ receivers : start =', rstart + 1)
                print('window @ receivers : stop =', rstop)
                print('window @ receivers : step =', rstep, '\n')
            
                tu = plotParams['tu']
                if tu != '':
                    print('window @ time : start = %0.2f %s' %(tstart, tu))
                    print('window @ time : stop = %0.2f %s' %(tstop, tu))
                else:
                    print('window @ time : start =', tstart)
                    print('window @ time : stop =', tstop)
                print('window @ time : step =', tstep, '\n')
                
                print('Applying window to data volume...')
                X = X[rinterval, :, :]
                X = X[:, tinterval, :]
            
                # Apply tapered cosine (Tukey) window to time signals.
                # This ensures the fast fourier transform (FFT) used in
                # the definition of the matrix-vector product below is
                # acting on a function that is continuous at its edges.
                
                Nt = X.shape[1]
                peakFreq = pulseFun.peakFreq
                # Np : Number of samples in the dominant period T = 1 / peakFreq
                Np = int(round(1 / (tstep * dt * peakFreq)))
                # alpha is set to taper over 6 of the dominant period of the
                # pulse function (3 periods from each end of the signal)
                alpha = 6 * Np / Nt
                print('Tapering time signals with Tukey window: %d'
                      %(int(round(alpha * 100))) + '%')
                TukeyWindow = tukey(Nt, alpha)
                X *= TukeyWindow[None, :, None]
            
        else:
            if Path('samplingGrid.npz').exists():
                samplingGrid = np.load('samplingGrid.npz')
                x = samplingGrid['x']
                y = samplingGrid['y']
                tau = samplingGrid['tau']
                if 'z' in samplingGrid:
                    z = samplingGrid['z']
                else:
                    z = None
                    
            else:
                sys.exit(textwrap.dedent(
                        '''
                        A sampling grid needs to be set up and test functions
                        computed before their Fourier spectrum can be plotted.
                        Enter:
                            
                            vzgrid --help
                            
                        from the command-line for more information on how to set up a
                        sampling grid.
                        '''))
            
            pulse = lambda t : pulseFun.pulse(t)
            velocity = pulseFun.velocity
            peakFreq = pulseFun.peakFreq
            peakTime = pulseFun.peakTime
        
            # set up the convolution times based on length of recording time interval
            T = recordingTimes[-1] - recordingTimes[0]
            convolutionTimes = np.linspace(-T, T, 2 * len(recordingTimes) - 1)    
            if Path('VZTestFuncs.npz').exists():
                print('\nDetected that free-space test functions have already been computed...')
                print('Checking consistency with current space-time sampling grid...')
                TFDict = np.load('VZTestFuncs.npz')
                
                if samplingIsCurrent(TFDict, receiverPoints, convolutionTimes, velocity, tau, x, y, z, peakFreq, peakTime):
                    X = TFDict['TFarray']
                    
                else:
                    print('Recomputing test functions...')
                    if tau[0] != 0:
                        tu = plotParams['tu']
                        if tu != '':
                            print('Recomputing test functions for focusing time %0.2f %s...' %(tau[0], tu))
                        else:
                            print('Recomputing test functions for focusing time %0.2f...' %(tau[0]))
                            X, sourcePoints = sampleSpace(receiverPoints, convolutionTimes - tau[0], velocity,
                                                          x, y, z, pulse)
                    else:
                        X, sourcePoints = sampleSpace(receiverPoints, convolutionTimes, velocity,
                                                      x, y, z, pulse)
                    
                    
                    if z is None:
                        np.savez('VZTestFuncs.npz', TFarray=X, time=convolutionTimes, receivers=receiverPoints,
                                 peakFreq=peakFreq, peakTime=peakTime, velocity=velocity,
                                 x=x, y=y, tau=tau, samplingPoints=sourcePoints)
                    else:
                        np.savez('VZTestFuncs.npz', TFarray=X, time=convolutionTimes, receivers=receiverPoints,
                                 peakFreq=peakFreq, peakTime=peakTime, velocity=velocity,
                                 x=x, y=y, z=z, tau=tau, samplingPoints=sourcePoints)
                    
            else:                
                print('\nComputing free-space test functions for the current space-time sampling grid...')
                if tau[0] != 0:
                    if tu != '':
                        print('Computing test functions for focusing time %0.2f %s...' %(tau[0], tu))
                    else:
                        print('Computing test functions for focusing time %0.2f...' %(tau[0]))
                        X, sourcePoints = sampleSpace(receiverPoints, convolutionTimes - tau[0], velocity,
                                                      x, y, z, pulse)
                else:
                    X, sourcePoints = sampleSpace(receiverPoints, convolutionTimes, velocity,
                                                  x, y, z, pulse)
                
                if z is None:
                    np.savez('VZTestFuncs.npz', TFarray=X, time=convolutionTimes, receivers=receiverPoints,
                             peakFreq=peakFreq, peakTime=peakTime, velocity=velocity,
                             x=x, y=y, tau=tau, samplingPoints=sourcePoints)
                else:
                    np.savez('VZTestFuncs.npz', TFarray=X, time=convolutionTimes, receivers=receiverPoints,
                             peakFreq=peakFreq, peakTime=peakTime, velocity=velocity,
                             x=x, y=y, z=z, tau=tau, samplingPoints=sourcePoints)
        
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
        if noisy:
            plotParams['freq_title'] += ' [Noisy ' + plotParams['data_title'] + ']'
        else:
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
    