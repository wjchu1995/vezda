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
import numpy as np
from pathlib import Path

sys.path.append(os.getcwd())
import pulseFun

def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--xaxis', type=str, default=None,
                        help='''Specify the endpoint values of the sampling
                        grid along the x-axis and the number of sampling points
                        in between. Syntax: --xaxis=start,stop,num.''')
    parser.add_argument('--yaxis', type=str, default=None,
                        help='''Specify the endpoint values of the sampling
                        grid along the y-axis and the number of sampling points
                        in between. Syntax: --yaxis=start,stop,num.''')
    parser.add_argument('--zaxis', type=str, default=None,
                        help='''Specify the endpoint values of the sampling
                        grid along the z-axis and the number of sampling points
                        in between. Syntax: --zaxis=start,stop,num.''')
    parser.add_argument('--taxis', type=str, default=None,
                        help='''Specify the endpoint values of the sampling
                        grid along the time axis and the number of sampling points
                        in between. Syntax: --taxis=start,stop,num.
                        
                        Choose only one of \'taxis, Ntau, tau\' command-line
                        arguments for sampling the time axis.''')
    parser.add_argument('--Ntau', type=str, default=None,
                        help='''Specify the number of evenly spaced time samples.
                        The endpoint values of the sampling grid along the time
                        axis are determined automatically. The lower endpoint is
                        set to zero. The upper endpoint is set to the last
                        recording time in the time window containing the 
                        measured data. If '--Ntau=auto', the
                        number of time samples is determined by 2 * pi * f * T,
                        where 'T' is the last recording time and 'f' is the 
                        dominant frequency of the pulse function in the 
                        'pulseFun.py' file.
                        Syntax: --Ntau=num.
                        
                        Choose only one of \'taxis, Ntau, tau\' command-line
                        arguments for sampling the time axis.''')
    parser.add_argument('--tau', type=float, default=None,
                        help='''Specify a single value to sample along the time axis
                        of the sampling grid.
                        Syntax: --tau=value.
                        
                        Choose only one of \'--taxis, --Ntau, --tau\' command-line
                        arguments for sampling the time axis.''')
    args = parser.parse_args()
    
    
    try:
        samplingGrid = np.load('samplingGrid.npz')
    except FileNotFoundError:
        samplingGrid = None
    
    #==============================================================================
    if not len(sys.argv) > 1:   # no arguments were passed...
        if samplingGrid is None:    # and a sampling grid doesn't exist...
            sys.exit('\nA space-time sampling grid has not been set up.\n')
        
        else:   # a sampling grid does exist...
            # Vezda will simply print the current sampling grid and exit
            x = samplingGrid['x']
            xstart = x[0]
            xstop = x[-1]
            xnum = len(x)
            
            y = samplingGrid['y']
            ystart = y[0]
            ystop = y[-1]
            ynum = len(y)
            
            # get sampling along time axis 'tau'
            tau = samplingGrid['tau']
            tnum = len(tau)
            if tnum == 1:
                tstart = tau[0]
                tstop = tstart
            else:
                tstart = tau[0]
                tstop = tau[-1]
            
            ndspace = int(samplingGrid['ndspace'])
            if ndspace == 2:
                print('\nCurrent sampling grid:\n')
                
                print('*** 3D space-time ***\n')
                
                print('grid @ x-axis : start =', xstart)
                print('grid @ x-axis : stop =', xstop)
                print('grid @ x-axis : num =', xnum, '\n')
                
                print('grid @ y-axis : start =', ystart)
                print('grid @ y-axis : stop =', ystop)
                print('grid @ y-axis : num =', ynum, '\n')
                
                print('grid @ t-axis : start =', tstart)
                print('grid @ t-axis : stop =', tstop)
                print('grid @ t-axis : num =', tnum, '\n')
                sys.exit()
        
            elif ndspace == 3:            
                z = samplingGrid['z']
                zstart = y[0]
                zstop = y[-1]
                znum = len(y)
                
                print('\nCurrent sampling grid:\n')
                
                print('*** 4D space-time ***\n')
                
                print('grid @ x-axis : start =', xstart)
                print('grid @ x-axis : stop =', xstop)
                print('grid @ x-axis : num =', xnum, '\n')
                
                print('grid @ y-axis : start =', ystart)
                print('grid @ y-axis : stop =', ystop)
                print('grid @ y-axis : num =', ynum, '\n')
                
                print('grid @ z-axis : start =', zstart)
                print('grid @ z-axis : stop =', zstop)
                print('grid @ z-axis : num =', znum, '\n')
                
                print('grid @ t-axis : start =', tstart)
                print('grid @ t-axis : stop =', tstop)
                print('grid @ t-axis : num =', tnum, '\n')
                sys.exit()
                  
    else:   # arguments were passed with 'vzgrid' call
        if samplingGrid is None and (args.xaxis is None or args.yaxis is None):
            sys.exit(textwrap.dedent(
                    '''
                    Error: Both of the command-line arguments \'--xaxis\' and \'--yaxis\'
                    must be specified (a minimum of two space dimenions is required).
                    Use the \'--zaxis\' argument to optionally specify a third space
                    dimension.
                    '''))
        #==============================================================================
        # set/update grid along t-axis
        if args.taxis is not None and args.Ntau is None and args.tau is None:
            # User is specifying the endpoint values of the sampling grid
            # along the time axis and the number of samples.
            taxis = args.taxis.split(',')
            tstart = float(taxis[0])
            tstop = float(taxis[1])
            tnum = int(taxis[2])
            if len(taxis) != 3:
                sys.exit(textwrap.dedent(
                        '''
                        Error: Must specify three values when using --taxis parameter.
                        Syntax: --taxis=start,stop,num
                        '''))
            if tstart < tstop and tnum > 0:
                tau = np.linspace(tstart, tstop, tnum)
            elif tstart >= tstop:
                sys.exit(textwrap.dedent(
                        '''
                        Error: Starting time value of the sampling grid must be less
                        than the ending time value of the sampling grid.
                        '''))
            elif tnum <= 0:
                sys.exit(textwrap.dedent(
                        '''
                        Error: Number of sampling points along the time axis
                        must be greater than zero.
                        '''))
                    
        elif args.taxis is None and args.Ntau is not None and args.tau is None:
            # User is specifying the number of samples along the time axis
            # while the endpoint values are determined automatically by the 
            # time window containing the measured data.
            datadir = np.load('datadir.npz')
            recordingTimes = np.load(str(datadir['recordingTimes']))
            
            tstart = 0
            if Path('window.npz').exists():
                windowDict = np.load('window.npz')
                tstop = windowDict['tstop']
            else:
                tstop = recordingTimes[-1]
                
            if args.Ntau == 'auto':
                tstep = 1 / (4 * pulseFun.peakFreq)    
                tau = np.arange(tstart, tstop, tstep)
                tnum = len(tau)
            else:
                tnum = int(args.Ntau)
            if tnum <= 0:
                sys.exit(textwrap.dedent(
                        '''
                        Error: Must specify a positive number of time samples.
                        Syntax: --Ntau=num
                        '''))
            else:
                tau = np.linspace(tstart, tstop, tnum)
        
        elif args.taxis is None and args.Ntau is None and args.tau is not None:
            tau = np.asarray([args.tau])
            tstart = args.tau
            tstop = tstart
            tnum = 1
                
        elif args.taxis is None and args.Ntau is None and args.tau is None:
            if samplingGrid is None:
                sys.exit(textwrap.dedent(
                        '''
                        Error: One of the command-line arguments \'--taxis, --Ntau, --tau\'
                        must be specified to determine the sampling of the time axis.
                        '''))
            else:
                # sampling along time axis is left unchanged
                tau = samplingGrid['tau']
                tnum = len(tau)
                if tnum == 1:
                    tstart = tau[0]
                    tstop = tstart
                else:
                    tstart = tau[0]
                    tstop = tau[-1]
                    
        #==============================================================================
        # set/update grid along x-axis
        if args.xaxis is not None:
            xaxis = args.xaxis.split(',')
            xstart = float(xaxis[0])
            xstop = float(xaxis[1])
            xnum = int(xaxis[2])
            
            if len(xaxis) != 3:
                sys.exit(textwrap.dedent(
                        '''
                        Error: Must specify three values when using --xaxis parameter.
                        Syntax: --xaxis=start,stop,num
                        '''))
            if xstart < xstop and xnum > 0:
                x = np.linspace(xstart, xstop, xnum)
            elif xstart >= xstop:
                sys.exit(textwrap.dedent(
                        '''
                        Error: Starting x-value of the sampling grid must be less
                        than the ending x-value of the sampling grid.
                        '''))
            elif xnum <= 0:
                sys.exit(textwrap.dedent(
                        '''
                        Error: Number of sampling points along the x-axis
                        must be greater than zero.
                        '''))
        else:
            x = samplingGrid['x']
            xstart = x[0]
            xstop = x[-1]
            xnum = len(x)

        #==============================================================================    
        # set/update grid along y-axis
        if args.yaxis is not None:
            yaxis = args.yaxis.split(',')
            ystart = float(yaxis[0])
            ystop = float(yaxis[1])
            ynum = int(yaxis[2])
            
            if len(yaxis) != 3:
                sys.exit(textwrap.dedent(
                        '''
                        Error: Must specify three values when using --yaxis parameter.
                        Syntax: --yaxis=start,stop,num
                        '''))
            if ystart < ystop and ynum > 0:
                y = np.linspace(ystart, ystop, ynum)
            elif ystart >= ystop:
                sys.exit(textwrap.dedent(
                        '''
                        Error: Starting y-value of the sampling grid must be less
                        than the ending y-value of the sampling grid.
                        '''))
            elif ynum <= 0:
                sys.exit(textwrap.dedent(
                        '''
                        Error: Number of sampling points along the y-axis
                        must be greater than zero.
                        '''))
        else:
            y = samplingGrid['y']
            ystart = y[0]
            ystop = y[-1]
            ynum = len(y)
            
        #==============================================================================
        # set/update grid along z-axis
        if args.zaxis is not None:
            ndspace = 3
            zaxis = args.zaxis.split(',')
            zstart = float(zaxis[0])
            zstop = float(zaxis[1])
            znum = int(zaxis[2])
            
            if len(zaxis) != 3:
                sys.exit(textwrap.dedent(
                        '''
                        Error: Must specify three values when using --zaxis parameter.
                        Syntax: --zaxis=start,stop,num
                        '''))
            if zstart < zstop and znum > 0:
                z = np.linspace(zstart, zstop, znum)
            elif zstart >= zstop:
                sys.exit(textwrap.dedent(
                        '''
                        Error: Starting z-value of the sampling grid must be less
                        than the ending z-value of the sampling grid.
                        '''))
            elif znum <= 0:
                sys.exit(textwrap.dedent(
                        '''
                        Error: Number of sampling points along the z-axis
                        must be greater than zero.
                        '''))
            
        else:
            if samplingGrid is None:
                ndspace = 2
            else:
                ndspace = int(samplingGrid['ndspace'])
                if ndspace == 3:
                    z = samplingGrid['z']
                    zstart = z[0]
                    zstop = z[-1]
                    znum = len(z)
                
    #==============================================================================
    if ndspace == 2:
        print('\nSetting up 3D space-time sampling grid:\n')
        print('grid @ x-axis : start =', xstart)
        print('grid @ x-axis : stop =', xstop)
        print('grid @ x-axis : num =', xnum, '\n')
        
        print('grid @ y-axis : start =', ystart)
        print('grid @ y-axis : stop =', ystop)
        print('grid @ y-axis : num =', ynum, '\n')
        
        print('grid @ t-axis : start =', tstart)
        print('grid @ t-axis : stop =', tstop)
        print('grid @ t-axis : num =', tnum, '\n')
        np.savez('samplingGrid.npz', ndspace=ndspace, x=x, y=y, tau=tau)
    
    elif ndspace == 3:                    
        print('\nSetting up 4D space-time sampling grid:\n')
        print('grid @ x-axis : start =', xstart)
        print('grid @ x-axis : stop =', xstop)
        print('grid @ x-axis : num =', xnum, '\n')
        
        print('grid @ y-axis : start =', ystart)
        print('grid @ y-axis : stop =', ystop)
        print('grid @ y-axis : num =', ynum, '\n')
        
        print('grid @ z-axis : start =', zstart)
        print('grid @ z-axis : stop =', zstop)
        print('grid @ z-axis : num =', znum, '\n')
        
        print('grid @ t-axis : start =', tstart)
        print('grid @ t-axis : stop =', tstop)
        print('grid @ t-axis : num =', tnum, '\n')
        np.savez('samplingGrid.npz', ndspace=ndspace, x=x, y=y, z=z, tau=tau)
