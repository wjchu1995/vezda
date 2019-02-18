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
import numpy as np
from vezda.plot_utils import FontColor

def info():
    commandName = FontColor.BOLD + 'vzgrid:' + FontColor.END
    description = ' specify a sampling grid for imaging'
    
    return commandName + description

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
    
    
    parser.add_argument('--tau', type=float, default=None,
                        help='''Specify the focusing time. (Default is zero.)
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
            y = samplingGrid['y']
                                  
            xstart = x[0]
            xstop = x[-1]
            xnum = len(x)
            
            ystart = y[0]
            ystop = y[-1]
            ynum = len(y)
            
            # get the focusing time 'tau'
            tau = samplingGrid['tau']
            
            if 'z' in samplingGrid:
                z = samplingGrid['z']
                zstart = z[0]
                zstop = z[-1]
                znum = len(z)
                
                print('\nCurrent sampling grid:\n')
                
                print('*** 3D space ***\n')
                
                print('grid @ x-axis : start =', xstart)
                print('grid @ x-axis : stop =', xstop)
                print('grid @ x-axis : num =', xnum, '\n')
                
                print('grid @ y-axis : start =', ystart)
                print('grid @ y-axis : stop =', ystop)
                print('grid @ y-axis : num =', ynum, '\n')
                
                print('grid @ z-axis : start =', zstart)
                print('grid @ z-axis : stop =', zstop)
                print('grid @ z-axis : num =', znum, '\n')
                
                print('focusing time : tau =', tau, '\n')
                sys.exit()
            
            else:
                print('\nCurrent sampling grid:\n')
                
                print('*** 2D space ***\n')
                
                print('grid @ x-axis : start =', xstart)
                print('grid @ x-axis : stop =', xstop)
                print('grid @ x-axis : num =', xnum, '\n')
                
                print('grid @ y-axis : start =', ystart)
                print('grid @ y-axis : stop =', ystop)
                print('grid @ y-axis : num =', ynum, '\n')
                
                print('focusing time : tau =', tau, '\n')
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
        # set/update the focusing time
        if args.tau is None:
            if samplingGrid is None:
                tau = 0
            else:
                tau = samplingGrid['tau']
        else:
            tau = args.tau   
                    
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
        print('\nSetting up 2D sampling grid:\n')
        print('grid @ x-axis : start =', xstart)
        print('grid @ x-axis : stop =', xstop)
        print('grid @ x-axis : num =', xnum, '\n')
        
        print('grid @ y-axis : start =', ystart)
        print('grid @ y-axis : stop =', ystop)
        print('grid @ y-axis : num =', ynum, '\n')
        
        print('focusing time : tau =', tau, '\n')
        np.savez('samplingGrid.npz', x=x, y=y, tau=tau)
    
    else:                    
        print('\nSetting up 3D sampling grid:\n')
        print('grid @ x-axis : start =', xstart)
        print('grid @ x-axis : stop =', xstop)
        print('grid @ x-axis : num =', xnum, '\n')
        
        print('grid @ y-axis : start =', ystart)
        print('grid @ y-axis : stop =', ystop)
        print('grid @ y-axis : num =', ynum, '\n')
        
        print('grid @ z-axis : start =', zstart)
        print('grid @ z-axis : stop =', zstop)
        print('grid @ z-axis : num =', znum, '\n')
        
        print('focusing time : tau =', tau, '\n')
        np.savez('samplingGrid.npz', x=x, y=y, z=z, tau=tau)