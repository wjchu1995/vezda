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
import numpy as np
from pathlib import Path

sys.path.append(os.getcwd())
import pulseFun

def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--xaxis', type=str, required=True,
                        help='''Specify the endpoint values of the sampling
                        grid along the x-axis and the number of sampling points
                        in between. Syntax: --xaxis=start,stop,num.''')
    parser.add_argument('--yaxis', type=str, required=True,
                        help='''Specify the endpoint values of the sampling
                        grid along the y-axis and the number of sampling points
                        in between. Syntax: --yaxis=start,stop,num.''')
    parser.add_argument('--zaxis', type=str, required=False, default=None,
                        help='''Specify the endpoint values of the sampling
                        grid along the z-axis and the number of sampling points
                        in between. Syntax: --zaxis=start,stop,num.''')
    parser.add_argument('--taxis', type=str, required=False, default=None,
                        help='''Specify the endpoint values of the sampling
                        grid along the time axis and the number of sampling points
                        in between. Syntax: --taxis=start,stop,num.
                        
                        Choose only one of \'taxis, Ntau, tau\' command-line
                        arguments for sampling the time axis.''')
    parser.add_argument('--Ntau', type=str, required=False, default=None,
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
    parser.add_argument('--tau', type=float, required=False, default=None,
                        help='''Specify a single value to sample along the time axis
                        of the sampling grid.
                        Syntax: --tau=value.
                        
                        Choose only one of \'--taxis, --Ntau, --tau\' command-line
                        arguments for sampling the time axis.''')
    args = parser.parse_args()
    #==============================================================================
    # set grid along t-axis
    if args.taxis is not None and args.Ntau is None and args.tau is None:
        # User is specifying the endpoint values of the sampling grid
        # along the time axis and the number of samples.
        taxis = args.taxis.split(',')
        tstart = float(taxis[0])
        tstop = float(taxis[1])
        tnum = int(taxis[2])
        if len(taxis) != 3:
            sys.exit('''
                     Error: Must specify three values when using --taxis parameter.
                     Syntax: --taxis=start,stop,num
                     ''')
        if tstart < tstop and tnum > 0:
            tau = np.linspace(tstart, tstop, tnum)
        elif tstart >= tstop:
            sys.exit('''
                     Error: Starting time value of the sampling grid must be less
                     than the ending time value of the sampling grid.
                     ''')
        elif tnum <= 0:
            sys.exit('''
                     Error: Number of sampling points along the time axis
                     must be greater than zero.
                     ''')
        
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
            tstep = 1 / (2 * np.pi * pulseFun.peakFreq)    
            tau = np.arange(tstart, tstop, tstep)
            tnum = len(tau)
        else:
            tnum = int(args.Ntau)
            if tnum <= 0:
                sys.exit('''
                         Error: Must specify a positive number of time samples.
                         Syntax: --Ntau=num
                         ''')
            else:
                tau = np.linspace(tstart, tstop, tnum)
        
    elif args.taxis is None and args.Ntau is None and args.tau is not None:
        tau = np.asarray([args.tau])
        tstart = args.tau
        tstop = tstart
        tnum = 1
        
    elif args.taxis is None and args.Ntau is None and args.tau is None:
        sys.exit('''
                 Error: One of the command-line arguments \'--taxis, --Ntau, --tau\'
                 must be specified to determine the sampling of the time axis.
                 ''')
    #==============================================================================
    # set grid along x-axis
    xaxis = args.xaxis.split(',')
    xstart = float(xaxis[0])
    xstop = float(xaxis[1])
    xnum = int(xaxis[2])
    
    if len(xaxis) != 3:
        sys.exit('''
                 Error: Must specify three values when using --xaxis parameter.
                 Syntax: --xaxis=start,stop,num
                 ''')
    if xstart < xstop and xnum > 0:
        x = np.linspace(xstart, xstop, xnum)
    elif xstart >= xstop:
        sys.exit('''
                 Error: Starting x-value of the sampling grid must be less
                 than the ending x-value of the sampling grid.
                 ''')
    elif xnum <= 0:
        sys.exit('''
                 Error: Number of sampling points along the x-axis
                 must be greater than zero.
                 ''')
    #==============================================================================    
    # set grid along y-axis
    yaxis = args.yaxis.split(',')
    ystart = float(yaxis[0])
    ystop = float(yaxis[1])
    ynum = int(yaxis[2])
    
    if len(yaxis) != 3:
        sys.exit('''
                 Error: Must specify three values when using --yaxis parameter.
                 Syntax: --yaxis=start,stop,num
                 ''')
    if ystart < ystop and ynum > 0:
        y = np.linspace(ystart, ystop, ynum)
    elif ystart >= ystop:
        sys.exit('''
                 Error: Starting y-value of the sampling grid must be less
                 than the ending y-value of the sampling grid.
                 ''')
    elif ynum <= 0:
        sys.exit('''
                 Error: Number of sampling points along the y-axis
                 must be greater than zero.
                 ''')
    #==============================================================================
    if args.zaxis is None:
        print('Setting up three-dimensional space-time sampling grid...')
        print('grid @ x-axis : start = ', xstart)
        print('grid @ x-axis : stop = ', xstop)
        print('grid @ x-axis : num = ', xnum)
        
        print('grid @ y-axis : start = ', ystart)
        print('grid @ y-axis : stop = ', ystop)
        print('grid @ y-axis : num = ', ynum)
        
        print('grid @ t-axis : start = ', tstart)
        print('grid @ t-axis : stop = ', tstop)
        print('grid @ t-axis : num = ', tnum)
        np.savez('samplingGrid.npz', ndspace=2, x=x, y=y, tau=tau)
    
    else:            
        # set grid along z-axis
        zaxis = args.zaxis.split(',')
        zstart = float(zaxis[0])
        zstop = float(zaxis[1])
        znum = int(zaxis[2])
        
        if len(zaxis) != 3:
            sys.exit('''
                     Error: Must specify three values when using --zaxis parameter.
                     Syntax: --zaxis=start,stop,num
                     ''')
        if zstart < zstop and znum > 0:
            z = np.linspace(zstart, zstop, znum)
        elif zstart >= zstop:
            sys.exit('''
                     Error: Starting z-value of the sampling grid must be less
                     than the ending z-value of the sampling grid.
                     ''')
        elif znum <= 0:
            sys.exit('''
                     Error: Number of sampling points along the z-axis
                     must be greater than zero.
                     ''')
        
        print('Setting up four-dimensional space-time sampling grid...')
        print('grid @ x-axis : start = ', xstart)
        print('grid @ x-axis : stop = ', xstop)
        print('grid @ x-axis : num = ', xnum)
        
        print('grid @ y-axis : start = ', ystart)
        print('grid @ y-axis : stop = ', ystop)
        print('grid @ y-axis : num = ', ynum)
        
        print('grid @ z-axis : start = ', zstart)
        print('grid @ z-axis : stop = ', zstop)
        print('grid @ z-axis : num = ', znum)
        
        print('grid @ t-axis : start = ', tstart)
        print('grid @ t-axis : stop = ', tstop)
        print('grid @ t-axis : num = ', tnum)
        np.savez('samplingGrid.npz', ndspace=3, x=x, y=y, z=z, tau=tau)