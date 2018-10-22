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
    parser.add_argument('--xstart', type=float,
                        help='Specify the beginning x-value of the sampling interval')
    parser.add_argument('--xstop', type=float,
                        help='Specify the ending x-value of the sampling interval')
    parser.add_argument('--nx', type=int,
                        help='Specify how the number of sampling points along the x-axis.')
    
    
    parser.add_argument('--yaxis', type=str, default=None,
                        help='''Specify the endpoint values of the sampling
                        grid along the y-axis and the number of sampling points
                        in between. Syntax: --yaxis=start,stop,num.''')
    parser.add_argument('--ystart', type=float,
                        help='Specify the beginning y-value of the sampling interval')
    parser.add_argument('--ystop', type=float,
                        help='Specify the ending y-value of the sampling interval')
    parser.add_argument('--ny', type=int,
                        help='Specify how the number of sampling points along the y-axis.')
    
    
    parser.add_argument('--zaxis', type=str, default=None,
                        help='''Specify the endpoint values of the sampling
                        grid along the z-axis and the number of sampling points
                        in between. Syntax: --zaxis=start,stop,num.''')
    parser.add_argument('--zstart', type=float,
                        help='Specify the beginning z-value of the sampling interval')
    parser.add_argument('--zstop', type=float,
                        help='Specify the ending z-value of the sampling interval')
    parser.add_argument('--nz', type=int,
                        help='Specify how the number of sampling points along the z-axis.')
    
    
    parser.add_argument('--taxis', type=str, default=None,
                        help='''Specify the endpoint values of the sampling
                        grid along the time axis and the number of sampling points
                        in between. Syntax: --taxis=start,stop,num.''')
    parser.add_argument('--tstart', type=float,
                        help='Specify the beginning t-value of the sampling interval')
    parser.add_argument('--tstop', type=float,
                        help='Specify the ending t-value of the sampling interval')
    parser.add_argument('--nt', type=int,
                        help='Specify how the number of sampling points along the time axis.')                        
    parser.add_argument('--tau', type=float, default=None,
                        help='''Specify a single value to sample along the time axis
                        of the sampling grid.
                        Syntax: --tau=value.''')
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
            
            if 'z' in samplingGrid:            
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
            
            else:
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
        
            
    else:   # arguments were passed with 'vzgrid' call
        if samplingGrid is None and any(v is None for v in [args.xaxis, args.yaxis]):
            sys.exit(textwrap.dedent(
                    '''
                    Error: Both of the command-line arguments \'--xaxis\' and \'--yaxis\'
                    must be specified (a minimum of two space dimenions is required).
                    Use the \'--zaxis\' argument to optionally specify a third space
                    dimension.
                    '''))
        #==============================================================================
        # set/update grid along t-axis
        if all(v is None for v in [args.taxis, args.tstart, args.tstop, args.nt, args.tau]):
            if samplingGrid is None:
                sys.exit(textwrap.dedent(
                        '''
                        Error: One of the command-line arguments \'--taxis, --tstart, --tstop, --tnum, --tau\'
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
                    
        elif args.tau is not None and all(v is None for v in [args.taxis, args.tstart, args.tstop, args.nt]):
            tau = np.asarray([args.tau])
            tstart = args.tau
            tstop = tstart
            tnum = 1
            
        elif args.taxis is not None and all(v is None for v in [args.tstart, args.tstop, args.nt, args.tau]):
            taxis = args.taxis.split(',')
            if len(taxis) != 3:
                sys.exit(textwrap.dedent(
                        '''
                        Error: Must specify three values when using --taxis parameter.
                        Syntax: --taxis=start,stop,step
                        '''))
            tstart = float(taxis[0])
            tstop = float(taxis[1])
            tnum = int(taxis[2])
            
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
            
        elif all(v is None for v in [args.taxis, args.tau]) and all(v is not None for v in [args.tstart, args.tstop, args.nt]):
            if args.tstart < args.tstop and args.nt > 0:
                tstart = args.tstart
                tstop = args.tstop
                tnum = args.nt
                tau = np.linspace(tstart, tstop, tnum)
            elif args.tstart >= args.tstop:
                sys.exit(textwrap.dedent(
                        '''
                        Error: Starting time value of the sampling grid must be less
                        than the ending time value of the sampling grid.
                        '''))
            elif args.nt <= 0:
                sys.exit(textwrap.dedent(
                        '''
                        Error: Number of sampling points along the time axis
                        must be greater than zero.
                        '''))
        
        elif all(v is None for v in[args.taxis, args.ystop, args.tau]) and all(v is not None for v in [args.tstart, args.nt]):
            tau = samplingGrid['tau']
            tstop = tau[-1]
            if args.tstart < tstop and args.nt > 0:
                tstart = args.tstart
                tnum = args.nt
                tau = np.linspace(tstart, tstop, tnum)
            elif args.tstart >= tstop:
                sys.exit(textwrap.dedent(
                        '''
                        Error: Starting time value of the sampling grid must be less
                        than the ending time value of the sampling grid.
                        '''))
            elif args.nt <= 0:
                sys.exit(textwrap.dedent(
                        '''
                        Error: Number of sampling points along the time axis
                        must be greater than zero.
                        '''))
                
        elif all(v is None for v in[args.taxis, args.tstart, args.tau]) and all(v is not None for v in [args.tstop, args.nt]):
            tau = samplingGrid['tau']
            tstart = tau[0]
            if tstart < args.tstop and args.nt > 0:
                tstop = args.tstop
                tnum = args.nt
                tau = np.linspace(tstart, tstop, tnum)
            elif tstart >= args.tstop:
                sys.exit(textwrap.dedent(
                        '''
                        Error: Starting time value of the sampling grid must be less
                        than the ending time value of the sampling grid.
                        '''))
            elif args.nt <= 0:
                sys.exit(textwrap.dedent(
                        '''
                        Error: Number of sampling points along the time axis
                        must be greater than zero.
                        '''))
        
        elif all(v is None for v in [args.taxis, args.tstart, args.tstop, args.tau]) and args.nt is not None:
            if samplingGrid is None:
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
                
                if args.nt == 'auto':
                    tstep = 1 / pulseFun.peakFreq    
                    tau = np.arange(tstart, tstop, tstep)
                    tnum = len(tau)
                else:
                    tnum = int(args.nt)
                
                    if tnum > 0:
                        tau = np.linspace(tstart, tstop, tnum)
                    else:
                        sys.exit(textwrap.dedent(
                                '''
                                Error: Must specify a positive number of time samples.
                                Syntax: --tnum=num
                                '''))                        
                    
            else:
                
                if args.nt > 0:
                    tau = samplingGrid['tau']
                    if len(tau) > 1:
                        tstart = tau[0]
                        tstop = tau[-1]
                        tnum = args.nt
                        tau = np.linspace(tstart, tstop, tnum)
                    elif len(tau) == 1 and args.nt == 1:
                        tstart = tau[0]
                        tstop = tstart
                        tnum = args.nt
                        tau = np.asarray(tstart)
                    else:
                        sys.exit(textwrap.dedent(
                                '''
                                Error: A single point in time cannot be sampled more than once.
                                '''))
                        
                else:
                    sys.exit(textwrap.dedent(
                            '''
                            Error: Number of sampling points along the time axis
                            must be greater than zero.
                            '''))
        
        elif all(v is None for v in [args.taxis, args.nt, args.tau]) and all(v is not None for v in[args.tstart, args.tstop]):
            tau = samplingGrid['tau']
            tnum = len(tau)
            if args.tstart < args.tstop:
                tstart = args.tstart
                tstop = args.tstop
                tau = np.linspace(tstart, tstop, tnum)
            else:
                sys.exit(textwrap.dedent(
                        '''
                        Error: Starting time value of the sampling grid must be less
                        than the ending time value of the sampling grid.
                        '''))

        elif all(v is None for v in [args.taxis, args.tstop, args.nt, args.tau]) and args.tstart is not None:
            tau = samplingGrid['tau']
            tstop = tau[-1]
            tnum = len(tau)
            if args.tstart < tstop:
                tstart = args.tstart
                tau = np.linspace(tstart, tstop, tnum)
            else:
                sys.exit(textwrap.dedent(
                        '''
                        Error: Starting time value of the sampling grid must be less
                        than the ending time value of the sampling grid.
                        '''))

        elif all(v is None for v in [args.taxis, args.tstart, args.nt, args.tau]) and args.tstop is not None:
            tau = samplingGrid['tau']
            tstart = tau[0]
            tnum = len(tau)
            if tstart < args.tstop:
                tstop = args.tstop
                tau = np.linspace(tstart, tstop, tnum)
            else:
                sys.exit(textwrap.dedent(
                        '''
                        Error: Starting time value of the sampling grid must be less
                        than the ending time value of the sampling grid.
                        '''))
        
        elif args.taxis is not None and any(v is not None for v in [args.tstart, args.tstop, args.nt, args.tau]):
            sys.exit(textwrap.dedent(
                    '''
                    Error: Cannot use --taxis together with any of --tstart/tstop/tnum/tau.
                    Must use --taxis or --tau by itself or a combination of --tstart/tstop/tnum.
                    '''))            
                    
        #==============================================================================
        # set/update grid along x-axis
        if all(v is None for v in [args.xaxis, args.xstart, args.xstop, args.nx]):
            x = samplingGrid['x']
            xstart = x[0]
            xstop = x[-1]
            xnum = len(x)
            
        elif args.xaxis is not None and all(v is None for v in [args.xstart, args.xstop, args.nx]):
            xaxis = args.xaxis.split(',')
            if len(xaxis) != 3:
                sys.exit(textwrap.dedent(
                        '''
                        Error: Must specify three values when using --xaxis parameter.
                        Syntax: --xaxis=start,stop,step
                        '''))
            xstart = float(xaxis[0])
            xstop = float(xaxis[1])
            xnum = int(xaxis[2])
            
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
            
        elif args.xaxis is None and all(v is not None for v in [args.xstart, args.xstop, args.nx]):
            if args.xstart < args.xstop and args.nx > 0:
                xstart = args.xstart
                xstop = args.xstop
                xnum = args.nx
                x = np.linspace(xstart, xstop, xnum)
            elif args.xstart >= args.xstop:
                sys.exit(textwrap.dedent(
                        '''
                        Error: Starting x-value of the sampling grid must be less
                        than the ending x-value of the sampling grid.
                        '''))
            elif args.nx <= 0:
                sys.exit(textwrap.dedent(
                        '''
                        Error: Number of sampling points along the x-axis
                        must be greater than zero.
                        '''))
        
        elif all(v is None for v in[args.xaxis, args.xstop]) and all(v is not None for v in [args.xstart, args.nx]):
            x = samplingGrid['x']
            xstop = x[-1]
            if args.xstart < xstop and args.nx > 0:
                xstart = args.xstart
                xnum = args.nx
                x = np.linspace(xstart, xstop, xnum)
            elif args.xstart >= xstop:
                sys.exit(textwrap.dedent(
                        '''
                        Error: Starting x-value of the sampling grid must be less
                        than the ending x-value of the sampling grid.
                        '''))
            elif args.nx <= 0:
                sys.exit(textwrap.dedent(
                        '''
                        Error: Number of sampling points along the x-axis
                        must be greater than zero.
                        '''))
                
        elif all(v is None for v in[args.xaxis, args.xstart]) and all(v is not None for v in [args.xstop, args.nx]):
            x = samplingGrid['x']
            xstart = x[0]
            if xstart < args.xstop and args.nx > 0:
                xstop = args.xstop
                xnum = args.nx
                x = np.linspace(xstart, xstop, xnum)
            elif xstart >= args.xstop:
                sys.exit(textwrap.dedent(
                        '''
                        Error: Starting x-value of the sampling grid must be less
                        than the ending x-value of the sampling grid.
                        '''))
            elif args.nx <= 0:
                sys.exit(textwrap.dedent(
                        '''
                        Error: Number of sampling points along the x-axis
                        must be greater than zero.
                        '''))
        
        elif all(v is None for v in [args.xaxis, args.xstart, args.xstop]) and args.nx is not None:
            x = samplingGrid['x']
            xstart = x[0]
            xstop = x[-1]
            if args.nx > 0:
                xnum = args.nx
                x = np.linspace(xstart, xstop, xnum)
            else:
                sys.exit(textwrap.dedent(
                        '''
                        Error: Number of sampling points along the x-axis
                        must be greater than zero.
                        '''))
        
        elif all(v is None for v in [args.xaxis, args.nx]) and all(v is not None for v in[args.xstart, args.xstop]):
            x = samplingGrid['x']
            xnum = len(x)
            if args.xstart < args.xstop:
                xstart = args.xstart
                xstop = args.xstop
                x = np.linspace(xstart, xstop, xnum)
            else:
                sys.exit(textwrap.dedent(
                        '''
                        Error: Starting x-value of the sampling grid must be less
                        than the ending x-value of the sampling grid.
                        '''))

        elif all(v is None for v in [args.xaxis, args.xstop, args.nx]) and args.xstart is not None:
            x = samplingGrid['x']
            xstop = x[-1]
            xnum = len(x)
            if args.xstart < xstop:
                xstart = args.xstart
                x = np.linspace(xstart, xstop, xnum)
            else:
                sys.exit(textwrap.dedent(
                        '''
                        Error: Starting x-value of the sampling grid must be less
                        than the ending x-value of the sampling grid.
                        '''))

        elif all(v is None for v in [args.xaxis, args.xstart, args.nx]) and args.xstop is not None:
            x = samplingGrid['x']
            xstart = x[0]
            xnum = len(x)
            if xstart < args.xstop:
                xstop = args.xstop
                x = np.linspace(xstart, xstop, xnum)
            else:
                sys.exit(textwrap.dedent(
                        '''
                        Error: Starting x-value of the sampling grid must be less
                        than the ending x-value of the sampling grid.
                        '''))
        
        elif args.xaxis is not None and any(v is not None for v in [args.xstart, args.xstop, args.nx]):
            sys.exit(textwrap.dedent(
                    '''
                    Error: Cannot use --xaxis together with any of --xstart/xstop/xnum.
                    Must use either --xaxis by itself or a combination of --xstart/xstop/xnum.
                    '''))

        #==============================================================================    
        # set/update grid along y-axis
        if all(v is None for v in [args.yaxis, args.ystart, args.ystop, args.ny]):
            y = samplingGrid['y']
            ystart = y[0]
            ystop = y[-1]
            ynum = len(y)
            
        elif args.yaxis is not None and all(v is None for v in [args.ystart, args.ystop, args.ny]):
            yaxis = args.yaxis.split(',')
            if len(yaxis) != 3:
                sys.exit(textwrap.dedent(
                        '''
                        Error: Must specify three values when using --yaxis parameter.
                        Syntax: --yaxis=start,stop,step
                        '''))
            ystart = float(yaxis[0])
            ystop = float(yaxis[1])
            ynum = int(yaxis[2])
            
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
            
        elif args.yaxis is None and all(v is not None for v in [args.ystart, args.ystop, args.ny]):
            if args.ystart < args.ystop and args.ny > 0:
                ystart = args.ystart
                ystop = args.ystop
                ynum = args.ny
                y = np.linspace(ystart, ystop, ynum)
            elif args.ystart >= args.ystop:
                sys.exit(textwrap.dedent(
                        '''
                        Error: Starting y-value of the sampling grid must be less
                        than the ending y-value of the sampling grid.
                        '''))
            elif args.ny <= 0:
                sys.exit(textwrap.dedent(
                        '''
                        Error: Number of sampling points along the y-axis
                        must be greater than zero.
                        '''))
        
        elif all(v is None for v in[args.yaxis, args.ystop]) and all(v is not None for v in [args.ystart, args.ny]):
            y = samplingGrid['y']
            ystop = x[-1]
            if args.ystart < ystop and args.ny > 0:
                ystart = args.ystart
                ynum = args.ny
                y = np.linspace(ystart, ystop, ynum)
            elif args.ystart >= ystop:
                sys.exit(textwrap.dedent(
                        '''
                        Error: Starting y-value of the sampling grid must be less
                        than the ending y-value of the sampling grid.
                        '''))
            elif args.ny <= 0:
                sys.exit(textwrap.dedent(
                        '''
                        Error: Number of sampling points along the y-axis
                        must be greater than zero.
                        '''))
                
        elif all(v is None for v in[args.yaxis, args.ystart]) and all(v is not None for v in [args.ystop, args.ny]):
            y = samplingGrid['y']
            ystart = y[0]
            if ystart < args.ystop and args.ny > 0:
                ystop = args.ystop
                ynum = args.ny
                y = np.linspace(ystart, ystop, ynum)
            elif ystart >= args.ystop:
                sys.exit(textwrap.dedent(
                        '''
                        Error: Starting y-value of the sampling grid must be less
                        than the ending y-value of the sampling grid.
                        '''))
            elif args.ny <= 0:
                sys.exit(textwrap.dedent(
                        '''
                        Error: Number of sampling points along the y-axis
                        must be greater than zero.
                        '''))
        
        elif all(v is None for v in [args.yaxis, args.ystart, args.ystop]) and args.ny is not None:
            y = samplingGrid['y']
            ystart = y[0]
            ystop = y[-1]
            if args.ny > 0:
                ynum = args.ny
                y = np.linspace(ystart, ystop, ynum)
            else:
                sys.exit(textwrap.dedent(
                        '''
                        Error: Number of sampling points along the y-axis
                        must be greater than zero.
                        '''))
        
        elif all(v is None for v in [args.yaxis, args.ny]) and all(v is not None for v in[args.ystart, args.ystop]):
            y = samplingGrid['y']
            ynum = len(y)
            if args.ystart < args.ystop:
                ystart = args.ystart
                ystop = args.ystop
                y = np.linspace(ystart, ystop, ynum)
            else:
                sys.exit(textwrap.dedent(
                        '''
                        Error: Starting y-value of the sampling grid must be less
                        than the ending y-value of the sampling grid.
                        '''))

        elif all(v is None for v in [args.yaxis, args.ystop, args.ny]) and args.ystart is not None:
            y = samplingGrid['y']
            ystop = y[-1]
            ynum = len(y)
            if args.ystart < ystop:
                ystart = args.ystart
                y = np.linspace(ystart, ystop, ynum)
            else:
                sys.exit(textwrap.dedent(
                        '''
                        Error: Starting y-value of the sampling grid must be less
                        than the ending y-value of the sampling grid.
                        '''))

        elif all(v is None for v in [args.yaxis, args.ystart, args.ny]) and args.ystop is not None:
            y = samplingGrid['y']
            ystart = y[0]
            ynum = len(y)
            if ystart < args.ystop:
                ystop = args.ystop
                y = np.linspace(ystart, ystop, ynum)
            else:
                sys.exit(textwrap.dedent(
                        '''
                        Error: Starting y-value of the sampling grid must be less
                        than the ending y-value of the sampling grid.
                        '''))
        
        elif args.yaxis is not None and any(v is not None for v in [args.ystart, args.ystop, args.ny]):
            sys.exit(textwrap.dedent(
                    '''
                    Error: Cannot use --yaxis together with any of --ystart/ystop/ynum.
                    Must use either --yaxis by itself or a combination of --ystart/ystop/ynum.
                    '''))
            
        #==============================================================================
        # set/update grid along z-axis
        if all(v is None for v in [args.zaxis, args.zstart, args.zstop, args.nz]):
            if samplingGrid is not None and 'z' in samplingGrid:
                z = samplingGrid['z']
                zstart = z[0]
                zstop = z[-1]
                znum = len(z)
            
        elif args.zaxis is not None and all(v is None for v in [args.zstart, args.zstop, args.nz]):
            zaxis = args.zaxis.split(',')
            if len(zaxis) != 3:
                sys.exit(textwrap.dedent(
                        '''
                        Error: Must specify three values when using --zaxis parameter.
                        Syntax: --zaxis=start,stop,step
                        '''))
            zstart = float(zaxis[0])
            zstop = float(zaxis[1])
            znum = int(zaxis[2])
            
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
            
        elif args.zaxis is None and all(v is not None for v in [args.zstart, args.zstop, args.nz]):
            if args.zstart < args.zstop and args.nz > 0:
                zstart = args.zstart
                zstop = args.zstop
                znum = args.nz
                z = np.linspace(zstart, zstop, znum)
            elif args.zstart >= args.zstop:
                sys.exit(textwrap.dedent(
                        '''
                        Error: Starting z-value of the sampling grid must be less
                        than the ending z-value of the sampling grid.
                        '''))
            elif args.nz <= 0:
                sys.exit(textwrap.dedent(
                        '''
                        Error: Number of sampling points along the z-axis
                        must be greater than zero.
                        '''))
        
        elif all(v is None for v in[args.zaxis, args.zstop]) and all(v is not None for v in [args.zstart, args.nz]):
            z = samplingGrid['z']
            zstop = z[-1]
            if args.zstart < zstop and args.nz > 0:
                zstart = args.zstart
                znum = args.nz
                z = np.linspace(zstart, zstop, znum)
            elif args.zstart >= zstop:
                sys.exit(textwrap.dedent(
                        '''
                        Error: Starting z-value of the sampling grid must be less
                        than the ending z-value of the sampling grid.
                        '''))
            elif args.nz <= 0:
                sys.exit(textwrap.dedent(
                        '''
                        Error: Number of sampling points along the z-axis
                        must be greater than zero.
                        '''))
                
        elif all(v is None for v in[args.zaxis, args.zstart]) and all(v is not None for v in [args.zstop, args.nz]):
            z = samplingGrid['z']
            zstart = z[0]
            if zstart < args.zstop and args.nz > 0:
                zstop = args.zstop
                znum = args.nz
                z = np.linspace(zstart, zstop, znum)
            elif zstart >= args.zstop:
                sys.exit(textwrap.dedent(
                        '''
                        Error: Starting z-value of the sampling grid must be less
                        than the ending z-value of the sampling grid.
                        '''))
            elif args.nz <= 0:
                sys.exit(textwrap.dedent(
                        '''
                        Error: Number of sampling points along the z-axis
                        must be greater than zero.
                        '''))
        
        elif all(v is None for v in [args.zaxis, args.zstart, args.zstop]) and args.nz is not None:
            z = samplingGrid['z']
            zstart = z[0]
            zstop = z[-1]
            if args.nz > 0:
                znum = args.nz
                z = np.linspace(zstart, zstop, znum)
            else:
                sys.exit(textwrap.dedent(
                        '''
                        Error: Number of sampling points along the z-axis
                        must be greater than zero.
                        '''))
        
        elif all(v is None for v in [args.zaxis, args.nz]) and all(v is not None for v in[args.zstart, args.zstop]):
            z = samplingGrid['z']
            znum = len(z)
            if args.zstart < args.zstop:
                zstart = args.zstart
                zstop = args.zstop
                z = np.linspace(zstart, zstop, znum)
            else:
                sys.exit(textwrap.dedent(
                        '''
                        Error: Starting z-value of the sampling grid must be less
                        than the ending z-value of the sampling grid.
                        '''))

        elif all(v is None for v in [args.zaxis, args.zstop, args.nz]) and args.zstart is not None:
            z = samplingGrid['z']
            zstop = z[-1]
            znum = len(z)
            if args.zstart < zstop:
                zstart = args.zstart
                z = np.linspace(zstart, zstop, znum)
            else:
                sys.exit(textwrap.dedent(
                        '''
                        Error: Starting z-value of the sampling grid must be less
                        than the ending z-value of the sampling grid.
                        '''))

        elif all(v is None for v in [args.zaxis, args.zstart, args.nz]) and args.zstop is not None:
            z = samplingGrid['z']
            zstart = z[0]
            znum = len(z)
            if zstart < args.zstop:
                zstop = args.zstop
                z = np.linspace(zstart, zstop, znum)
            else:
                sys.exit(textwrap.dedent(
                        '''
                        Error: Starting z-value of the sampling grid must be less
                        than the ending z-value of the sampling grid.
                        '''))
        
        elif args.zaxis is not None and any(v is not None for v in [args.zstart, args.zstop, args.nz]):
            sys.exit(textwrap.dedent(
                    '''
                    Error: Cannot use --zaxis together with any of --zstart/zstop/znum.
                    Must use either --zaxis by itself or a combination of --zstart/zstop/znum.
                    '''))
                
    #==============================================================================
    try:
        z
    except NameError:
        z = None
    
    if z is None:
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
        np.savez('samplingGrid.npz', x=x, y=y, tau=tau)
    
    else:                    
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
        np.savez('samplingGrid.npz', x=x, y=y, z=z, tau=tau)
