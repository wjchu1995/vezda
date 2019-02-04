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
from pathlib import Path
from vezda.plot_utils import FontColor

def info():
    commandName = FontColor.BOLD + 'vzwindow:' + FontColor.END
    description = ' window a subset of the data volume to use for imaging'
    
    return commandName + description


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--time', type=str,
                        help='''specify the window applied to the time axis
                        and the step size for array sampling. Step size must be
                        a postive integer greater than or equal to 1.
                        Syntax: --time=start,stop,step.''')
    parser.add_argument('--sources', type=str,
                        help='''specify the window applied to the sources axis
                        and the step size for array sampling. Step size must be
                        a positive integer greater than or equal to 1.
                        Syntax: --sources=start,stop,step.''')
    parser.add_argument('--receivers', type=str,
                        help='''specify the window applied to the receiver axis
                        and the step size for array sampling. Step size must be
                        a positive integer greater than or equal to 1.
                        Syntax: --receivers=start,stop,step.''')
    
    parser.add_argument('--tstart', type=float,
                        help='specify the beginning of the time window')
    parser.add_argument('--tstop', type=float,
                        help='specify the end of the time window')
    parser.add_argument('--tstep', type=int,
                        help='''specify how to sample the windowed time signal array.
                        Must be a positive integer greater than or equal to 1 (default).
                        (e.g., --step=2 will sample every other value of the signal in the
                        time window.''')
                        
    parser.add_argument('--rstart', type=int,
                        help='specify the beginning of the receiver window')
    parser.add_argument('--rstop', type=int,
                        help='specify the end of the receiver window')
    parser.add_argument('--rstep', type=int,
                        help='''specify how to sample the windowed receiver array.
                        Must be a positive integer greater than or equal to 1 (default).
                        (e.g., --step=2 will use every other receiver in the window.''')
                        
    parser.add_argument('--sstart', type=int,
                        help='specify the beginning of the source window')
    parser.add_argument('--sstop', type=int,
                        help='specify the end of the source window')
    parser.add_argument('--sstep', type=int,
                        help='''specify how to sample the windowed source array.
                        Must be a positive integer greater than or equal to 1 (default).
                        (e.g., --step=2 will use every other source in the window.''')
                        
    args = parser.parse_args()    
        
    if Path('datadir.npz').exists():
        datadir = np.load('datadir.npz')
        
        # Load 3D data array
        # axis 0: receiver axis
        # axis 1: time axis
        # axis 2: source/recordings axis
        data = np.load(str(datadir['recordedData']))
        
        # Check that number of receivers in receiver array equals length of
        # receiver axis in data array
        receivers = np.load(str(datadir['receivers']))
        if receivers.shape[0] == data.shape[0]:
            Nr = receivers.shape[0]
        else:
            sys.exit(textwrap.dedent(
                    '''
                    Error: Inconsistent array length. Number of receivers
                    in receiverPoints array does not equal number of receivers
                    in recordedData array.
                    '''))
        
        # Check that length of recordingTimes array equals length of
        # time axis in data array
        recordingTimes = np.load(str(datadir['recordingTimes']))
        if len(recordingTimes) != data.shape[1]:
            sys.exit(textwrap.dedent(
                    '''
                    Error: Inconsistent array length. Length of recordingTimes array
                    does not equal length of time axis in recordedData array.
                    '''))
        
        if 'sources' in datadir:
            # Check that number of sources in source array equals length of
            # source in data array 
            sources = np.load(str(datadir['sources']))
            if sources.shape[0] == data.shape[2]:
                slabel = 'sources'
                Ns = sources.shape[0]
            else:
                sys.exit(textwrap.dedent(
                        '''
                        Error: Inconsistent array length. Number of sources
                        in sourcePoints array does not eqaul number of sources
                        in recordedData array.
                        '''))
        else:
            slabel = 'recordings'
            Ns = data.shape[2]
    
    else:
        sys.exit(textwrap.dedent(
                '''
                Error: A relative path to the data directory from the current 
                directory has not been specified. To access the data from this
                location, enter:
                    
                    vzdata --path=<path/to/data/directory>
                    
                from the command line.
                '''))
            
    try:
        windowDict = np.load('window.npz')
    except FileNotFoundError:
        windowDict = None
    
    #==============================================================================
    if not len(sys.argv) > 1:   # no arguments were passed...
        if windowDict is None:  # and a window dictionary doesn't exist...
            sys.exit(textwrap.dedent(
                    '''
                    A window to apply to the data volume has not been specified. Run the
                    command
                    
                        vzwindow --help
                            
                    from the command line to see windowing options.
                    '''))
        else:   # a window dictionary already exists...
            # Vezda will simply print the current windowing parameters and exit
            windowDict = np.load('window.npz')
            
            rstart = windowDict['rstart']
            rstop = windowDict['rstop']
            rstep = windowDict['rstep']
            
            tstart = windowDict['tstart']
            tstop = windowDict['tstop']
            tstep = windowDict['tstep']
            
            slabel = windowDict['slabel']
            sstart = windowDict['sstart']
            sstop = windowDict['sstop']
            sstep = windowDict['sstep']
            
            print('\nCurrent windowing parameters for data volume:\n')
            
            # For display/printing purposes, count receivers with one-based
            # indexing. This amounts to incrementing the rstart parameter by 1
            print('window @ receivers : start =', rstart + 1)
            print('window @ receivers : stop =', rstop)
            print('window @ receivers : step =', rstep, '\n')
            
            print('window @ time : start =', tstart)
            print('window @ time : stop =', tstop)
            print('window @ time : step =', tstep, '\n')
             
            # For display/printing purposes, count recordings/sources with one-based
            # indexing. This amounts to incrementing the sstart parameter by 1
            print('window @ %s : start = %s' %(slabel, sstart + 1))
            print('window @ %s : stop = %s' %(slabel, sstop))
            print('window @ %s : step = %s\n' %(slabel, sstep))
            sys.exit()
    
    else:   # arguments were passed with 'vzwindow' call
            
        #==============================================================================
        # set/update the time window
        if all(v is None for v in [args.time, args.tstart, args.tstop, args.tstep]):
            if windowDict is None:
                tstart = recordingTimes[0]
                tstop = recordingTimes[-1]
                tstep = 1
            else:
                tstart = windowDict['tstart']
                tstop = windowDict['tstop']
                tstep = windowDict['tstep']
            
        elif args.time is not None and all(v is None for v in [args.tstart, args.tstop, args.tstep]):
            time = args.time.split(',')
            if len(time) != 3:
                sys.exit(textwrap.dedent(
                        '''
                        Error: Must specify three values when using --time parameter.
                        Syntax: --time=start,stop,step
                        '''))
            if recordingTimes[0] <= float(time[0]) and float(time[0]) < float(time[1]) and float(time[1]) <= recordingTimes[-1] and int(time[2]) > 0:
                tstart = float(time[0])
                tstop = float(time[1])
                tstep = int(time[2])
            elif float(time[0]) >= float(time[1]):
                sys.exit(textwrap.dedent(
                        '''
                        Error: Starting value of the time window must be less
                        than the ending value of the time window.
                        
                        User entered:
                        window @ time : start = %s
                        window @ time : stop = %s
                        ''' %(float(time[0]), float(time[1]))))
            elif recordingTimes[0] > float(time[0]) or float(time[0]) > recordingTimes[-1]:
                sys.exit(textwrap.dedent(
                        '''
                        Error: Starting value of the time window must lie within
                        the recorded time interval [%s, %s].
                        
                        User entered:
                        window @ time : start = %s
                        ''' %(recordingTimes[0], recordingTimes[-1], float(time[0]))))
            elif recordingTimes[0] > float(time[1]) or float(time[1]) > recordingTimes[-1]:
                sys.exit(textwrap.dedent(
                        '''
                        Error: Ending value of the time window must lie within
                        the recorded time interval [%s, %s].
                        
                        User entered:
                        window @ time : stop = %s
                        ''' %(recordingTimes[0], recordingTimes[-1], float(time[1]))))
            elif int(time[2]) <= 0:
                sys.exit(textwrap.dedent(
                        '''
                        Error: Step size or spacing between array indices must
                        be a positive integer greater than or equal to 1.
                        
                        User entered:
                        window @ time : step = %
                        ''' %(int(time[2]))))
            
        elif args.time is None and all(v is not None for v in [args.tstart, args.tstop, args.tstep]):
            if recordingTimes[0] <= args.tstart and args.tstart < args.tstop and args.tstop <= recordingTimes[-1] and args.tstep > 0:
                tstart = args.tstart
                tstop = args.tstop
                tstep = args.tstep
            elif args.tstart >= args.tstop:
                sys.exit(textwrap.dedent(
                        '''
                        Error: Starting value of the time window must be less
                        than the ending value of the time window.
                        
                        User entered:
                        window @ time : start = %s
                        window @ time : stop = %s
                        ''' %(args.tstart, args.tstop)))
            elif recordingTimes[0] > args.tstart or args.tstart > recordingTimes[-1]:
                sys.exit(textwrap.dedent(
                        '''
                        Error: Starting value of the time window must lie within
                        the recorded time interval [%s, %s].
                        
                        User entered:
                        window @ time : start = %s
                        ''' %(recordingTimes[0], recordingTimes[-1], args.tstart)))
            elif recordingTimes[0] > args.tstop or args.tstop > recordingTimes[-1]:
                sys.exit(textwrap.dedent(
                        '''
                        Error: Ending value of the time window must lie within
                        the recorded time interval [%s, %s].
                        
                        User entered:
                        window @ time : stop = %s
                        ''' %(recordingTimes[0], recordingTimes[-1], args.tstop)))
            elif args.tstep <= 0:
                sys.exit(textwrap.dedent(
                        '''
                        Error: Step size or spacing between array indices must
                        be a positive integer greater than or equal to 1.
                        
                        User entered:
                        window @ time : step = %s
                        ''' %(args.tstep)))
        
        elif all(v is None for v in [args.time, args.tstart]) and all(v is not None for v in [args.tstop, args.tstep]):
            if recordingTimes[0] < args.tstop and args.tstop <= recordingTimes[-1] and args.tstep > 0:
                tstop = args.tstop
                tstep = args.tstep
                if windowDict is None:
                    tstart = recordingTimes[0]
                else:
                    tstart = windowDict['tstart']
            elif recordingTimes[0] >= args.tstop or args.tstop > recordingTimes[-1]:
                sys.exit(textwrap.dedent(
                        '''
                        Error: Ending value of the time window must lie within
                        the recorded time interval [%s, %s].
                        
                        User entered:
                        window @ time : stop = %s
                        ''' %(recordingTimes[0], recordingTimes[-1], args.tstop)))
            elif args.tstep <= 0:
                sys.exit(textwrap.dedent(
                        '''
                        Error: Step size or spacing between array indices must
                        be a positive integer greater than or equal to 1.
                        
                        User entered:
                        window @ time : step = %s
                        ''' %(args.tstep)))
        
        elif all(v is None for v in [args.time, args.tstop]) and all(v is not None for v in [args.tstart, args.tstep]):
            if recordingTimes[0] <= args.tstart and args.tstart < recordingTimes[-1] and args.tstep > 0:
                tstart = args.tstart
                tstep = args.tstep
                if windowDict is None:
                    tstop = recordingTimes[-1]
                else:
                    tstop = windowDict['tstop']
            elif recordingTimes[0] > args.tstart or args.tstart >= recordingTimes[-1]:
                sys.exit(textwrap.dedent(
                        '''
                        Error: Starting value of the time window must lie within
                        the recorded time interval [%s, %s].
                        
                        User entered:
                        window @ time : start = %s
                        ''' %(recordingTimes[0], recordingTimes[-1], args.tstart)))
            elif args.tstep <= 0:
                sys.exit(textwrap.dedent(
                        '''
                        Error: Step size or spacing between array indices must
                        be a positive integer greater than or equal to 1.
                        
                        User entered:
                        window @ time : step = %s
                        ''' %(args.tstep)))
        
        elif all(v is None for v in [args.time, args.tstep]) and all(v is not None for v in [args.tstart, args.tstop]):
            if recordingTimes[0] <= args.tstart and args.tstart < args.tstop and args.tstop <= recordingTimes[-1]:
                tstart = args.tstart
                tstop = args.tstop
                if windowDict is None:
                    tstep = 1
                else:
                    tstep = windowDict['tstep']
            elif args.tstart >= args.tstop:
                sys.exit(textwrap.dedent(
                        '''
                        Error: Starting value of the time window must be less
                        than the ending value of the time window.
                        
                        User entered:
                        window @ time : start = %s
                        window @ time : stop = %s
                        ''' %(args.tstart, args.tstop)))
            elif recordingTimes[0] > args.tstart or args.tstart >= recordingTimes[-1]:
                sys.exit(textwrap.dedent(
                        '''
                        Error: Starting value of the time window must lie within
                        the recorded time interval [%s, %s].
                        
                        User entered:
                        window @ time : start = %s
                        ''' %(recordingTimes[0], recordingTimes[-1], args.tstart)))
            elif recordingTimes[0] > args.tstop or args.tstop >= recordingTimes[-1]:
                sys.exit(textwrap.dedent(
                        '''
                        Error: Ending value of the time window must lie within
                        the recorded time interval [%s, %s].
                        
                        User entered:
                        window @ time : stop = %s
                        ''' %(recordingTimes[0], recordingTimes[-1], args.tstop)))

        elif all(v is None for v in [args.time, args.tstop, args.tstep]) and args.tstart is not None:
            if recordingTimes[0] <= args.tstart and args.tstart < recordingTimes[-1]:
                tstart = args.tstart
                if windowDict is None:
                    tstop = recordingTimes[-1]
                    tstep = 1
                else:
                    tstop = windowDict['tstop']
                    tstep = windowDict['tstep']
            elif recordingTimes[0] > args.tstart or args.tstart >= recordingTimes[-1]:
                sys.exit(textwrap.dedent(
                        '''
                        Error: Starting value of the time window must lie within
                        the recorded time interval (%s, %s).
                        
                        User entered:
                        window @ time : start = %s
                        ''' %(recordingTimes[0], recordingTimes[-1], args.tstart)))

        elif all(v is None for v in [args.time, args.tstart, args.tstep]) and args.tstop is not None:
            if recordingTimes[0] < args.tstop and args.tstop <= recordingTimes[-1]:
                tstop = args.tstop
                if windowDict is None:
                    tstart = recordingTimes[0]
                    tstep = 1
                else:
                    tstart = windowDict['tstart']
                    tstep = windowDict['tstep']
            elif recordingTimes[0] >= args.tstop or args.tstop > recordingTimes[-1]:
                sys.exit(textwrap.dedent(
                        '''
                        Error: Ending value of the time window must lie within
                        the recorded time interval [%s, %s].
                        
                        User entered:
                        window @ time : stop = %s
                        ''' %(recordingTimes[0], recordingTimes[-1], args.tstop)))
    
        elif all(v is None for v in [args.time, args.tstart, args.tstop]) and args.tstep is not None:
            if args.tstep > 0:
                tstep = args.tstep
                if windowDict is None:
                    tstart = recordingTimes[0]
                    tstop = recordingTimes[-1]
                else:
                    tstart = windowDict['tstart']
                    tstop = windowDict['tstop']  
            elif args.tstep <= 0:
                sys.exit(textwrap.dedent(
                        '''
                        Error: Step size or spacing between array indices must
                        be a positive integer greater than or equal to 1.
                        
                        User entered:
                        window @ time : step = %s
                        ''' %(args.tstep)))
        
        elif args.time is not None and any(v is not None for v in [args.tstart, args.tstop, args.tstep]):
            sys.exit(textwrap.dedent(
                    '''
                    Error: Cannot use --time together with any of --tstart/tstop/tstep.
                    Must use either --time by itself or a combination of --tstart/tstop/tstep.
                    '''))
    
        #==============================================================================
        # set/update the source/recording window
        if all(v is None for v in [args.sources, args.sstart, args.sstop, args.sstep]):
            if windowDict is None:
                sstart = 0
                sstop = Ns
                sstep = 1
            else:
                sstart = windowDict['sstart']
                sstop = windowDict['sstop']
                sstep = windowDict['sstep']
                
        elif args.sources is not None and all(v is None for v in [args.sstart, args.sstop, args.sstep]):
            userSources = args.sources.split(',')
            if len(userSources) != 3:
                sys.exit(textwrap.dedent(
                        '''
                        Error: Must specify three values when using --sources parameter.
                        Syntax: --sources=start,stop,step
                        '''))
            if 1 <= int(userSources[0]) and int(userSources[0]) <= int(userSources[1]) and int(userSources[1]) <= Ns and int(userSources[2]) > 0:
                sstart = int(userSources[0]) - 1
                sstop = int(userSources[1])
                sstep = int(userSources[2])
            elif int(userSources[0]) > int(userSources[1]):
                sys.exit(textwrap.dedent(
                        '''
                        Error: Starting value of the source/recording window must be less
                        than or equal to the ending value of the source/recording window.
                                
                        User entered:
                        window @ %s : start = %s
                        window @ %s : stop = %s
                        ''' %(slabel, int(userSources[0]), slabel, int(userSources[1]))))
            elif 1 > int(userSources[0]) or int(userSources[0]) > Ns:
                sys.exit(textwrap.dedent(
                        '''
                        Error: Starting value of the source/recording window must lie within
                        the range of sources/recordings [%s, %s].
                        
                        User entered:
                        window @ %s : start = %s
                        ''' %(1, Ns, slabel, int(userSources[0]))))
            elif 1 > int(userSources[1]) or int(userSources[1]) > Ns:
                sys.exit(textwrap.dedent(
                        '''
                        Error: Ending value of the source/recording window must lie within
                        the range of sources/recordings [%s, %s].
                        
                        User entered:
                        window @ %s : stop = %s
                        ''' %(1, Ns, slabel, int(userSources[1]))))
            elif int(userSources[2]) <= 0:
                sys.exit(textwrap.dedent(
                        '''
                        Error: Step size or spacing between array indices must
                        be a positive integer greater than or equal to 1.
                        
                        User entered:
                        window @ %s : step = %s
                        ''' %(slabel, int(userSources[2]))))
        
        elif args.sources is None and all(v is not None for v in [args.sstart, args.sstop, args.sstep]):
            if 1 <= args.sstart and args.sstart <= args.sstop and args.sstop <= Ns and args.sstep > 0:
                sstart = args.sstart - 1
                sstop = args.sstop
                sstep = args.sstep
            elif args.sstart > args.sstop:
                sys.exit(textwrap.dedent(
                                '''
                                Error: Starting value of the source/recording window must be less
                                than or equal to the ending value of the source/recording window.
                                
                                User entered:
                                window @ %s : start = %s
                                window @ %s : stop = %s
                                ''' %(slabel, args.sstart, slabel, args.sstop)))
            elif 1 > args.sstart or args.sstart > Ns:
                sys.exit(textwrap.dedent(
                        '''
                        Error: Starting value of the source/recording window must lie within
                        the range of sources/recordings [%s, %s].
                        
                        User entered:
                        window @ %s : start = %s
                        ''' %(1, Ns, slabel, args.sstart)))
            elif 1 > args.sstop or args.sstop > Ns:
                sys.exit(textwrap.dedent(
                        '''
                        Error: Ending value of the source/recording window must lie within
                        the range of sources/recordings [%s, %s].
                        
                        User entered:
                        window @ %s : stop = %s
                        ''' %(1, Ns, slabel, args.sstop)))
            elif args.sstep <= 0:
                sys.exit(textwrap.dedent(
                        '''
                        Error: Step size or spacing between array indices must
                        be a positive integer greater than or equal to 1.
                        
                        User entered:
                        window @ %s : step = %s
                        ''' %(slabel, args.sstep)))
            
        elif all(v is None for v in [args.sources, args.sstart]) and all(v is not None for v in [args.sstop, args.sstep]):
            if 1 <= args.sstop and args.sstop <= Ns and args.sstep > 0:
                sstep = args.sstep
                if windowDict is None:
                    sstart = 0
                    sstop = args.sstop
                else:
                    sstart = windowDict['sstart']
                    if sstart <= args.sstop:
                        sstop = args.sstop
                    else:
                        sys.exit(textwrap.dedent(
                                '''
                                Error: Starting value of the source/recording window must be less
                                than or equal to the ending value of the source/recording window.
                                
                                Current starting value:
                                window @ %s : start = %s
                                
                                User entered:
                                window @ %s : stop = %s
                                ''' %(slabel, sstart, slabel, args.sstop)))
                        
            elif 1 > args.sstop or args.sstop > Ns:
                sys.exit(textwrap.dedent(
                        '''
                        Error: Ending value of the source/recording window must lie within
                        the range of sources/recordings [%s, %s].
                        
                        User entered:
                        window @ %s : stop = %s
                        ''' %(1, Ns, slabel, args.sstop)))
            elif args.sstep <= 0:
                sys.exit(textwrap.dedent(
                        '''
                        Error: Step size or spacing between array indices must
                        be a positive integer greater than or equal to 1.
                        
                        User entered:
                        window @ %s : step = %s
                        ''' %(slabel, args.sstep)))
    
        elif all(v is None for v in [args.sources, args.sstop]) and all(v is not None for v in [args.sstart, args.sstep]):
            if 1 <= args.sstart and args.sstart <= Ns and args.sstep > 0:
                sstep = args.sstep
                if windowDict is None:
                    sstop = Ns
                    if args.sstart <= sstop:
                        sstart = args.sstart - 1
                    else:
                        sys.exit(textwrap.dedent(
                                '''
                                Error: Starting value of the source/recording window must be less
                                than or equal to the ending value of the source/recording window.
                                
                                User entered:
                                window @ %s : start = %s
                                
                                Current ending value:
                                window @ %s : stop = %s
                                ''' %(slabel, args.sstart, slabel, sstop)))
                else:
                    sstop = windowDict['sstop']
                    if args.sstart <= sstop:
                        sstart = args.sstart - 1
                    else:
                        sys.exit(textwrap.dedent(
                                '''
                                Error: Starting value of the source/recording window must be less
                                than or equal to the ending value of the source/recording window.
                                
                                User entered:
                                window @ %s : start = %s
                                
                                Current ending value:
                                window @ %s : stop = %s
                                ''' %(slabel, args.sstart, slabel, sstop)))
            elif 1 > args.sstart or args.sstart > Ns:
                sys.exit(textwrap.dedent(
                        '''
                        Error: Starting value of the source/recording window must lie within
                        the range of sources/recordings [%s, %s].
                        
                        User entered:
                        window @ %s : start = %s
                        ''' %(1, Ns, slabel, args.sstart)))
            elif args.sstep <= 0:
                sys.exit(textwrap.dedent(
                        '''
                        Error: Step size or spacing between array indices must
                        be a positive integer greater than or equal to 1.
                        
                        User entered:
                        window @ %s : step = %s
                        ''' %(slabel, args.sstep)))
            
        elif all(v is None for v in [args.sources, args.sstep]) and all(v is not None for v in [args.sstart, args.sstop]):
            if 1 <= args.sstart and args.sstart <= args.sstop and args.sstop <= Ns:
                sstart = args.sstart - 1
                sstop = args.sstop
                if windowDict is None:
                    sstep = 1
                else:
                    sstep = windowDict['sstep']
            elif args.sstart > args.sstop:
                sys.exit(textwrap.dedent(
                        '''
                        Error: Starting value of the source/recording window must be less
                        than or equal to the ending value of the source/recording window.
                        
                        User entered:
                        window @ %s : start = %s
                        window @ %s : stop = %s
                        ''' %(slabel, args.sstart, slabel, args.sstop)))
            elif 1 > args.sstart or args.sstart > Ns:
                sys.exit(textwrap.dedent(
                        '''
                        Error: Starting value of the source window must lie within
                        the range of sources [%s, %s].
                        
                        User entered:
                        window @ %s : start = %s
                        ''' %(1, Ns, slabel, args.sstart)))
            elif 1 > args.sstop or args.sstop > Ns:
                sys.exit(textwrap.dedent(
                        '''
                        Error: Ending value of the source/recording window must lie within
                        the range of sources/recordings [%s, %s].
                        
                        User entered:
                        window @ %s : stop = %s
                        ''' %(1, Ns, slabel, args.sstop)))

        elif all(v is None for v in [args.sources, args.sstop, args.sstep]) and args.sstart is not None:
            if 1 <= args.sstart and args.sstart <= Ns:
                if windowDict is None:
                    sstop = Ns
                    sstep = 1
                    if args.sstart <= sstop:
                        sstart = args.sstart - 1
                    else:
                        sys.exit(textwrap.dedent(
                                '''
                                Error: Starting value of the source/recording window must be less
                                than or equal to the ending value of the source/recording window.
                                
                                User entered:
                                window @ %s : start = %s
                                
                                Current ending value:
                                window @ %s : stop = %s
                                ''' %(slabel, args.sstart, slabel, sstop)))
                        
                else:
                    sstop = windowDict['sstop']
                    sstep = windowDict['sstep']
                    if args.sstart <= sstop:
                        sstart = args.sstart - 1
                    else:
                        sys.exit(textwrap.dedent(
                                '''
                                Error: Starting value of the source/recording window must be less
                                than or equal to the ending value of the source/recording window.
                                
                                User entered:
                                window @ %s : start = %s
                                
                                Current ending value:
                                window @ %s : stop = %s
                                ''' %(slabel, args.sstart, slabel, sstop)))
                    
            elif 1 > args.sstart or args.sstart > Ns:
                sys.exit(textwrap.dedent(
                        '''
                        Error: Beginning value of the source/recording window must lie within
                        the range of sources/recordings [%s, %s].
                        
                        User entered:
                        window @ %s : start = %s
                        ''' %(1, Ns, slabel, args.sstart)))
    
        elif all(v is None for v in [args.sources, args.sstart, args.sstep]) and args.sstop is not None:
            if 1 <= args.sstop and args.sstop <= Ns:
                if windowDict is None:
                    sstart = 0
                    sstep = 1
                    if sstart <= args.sstop:
                        sstop = args.sstop
                    else:
                        sys.exit(textwrap.dedent(
                                '''
                                Error: Starting value of the source/recording window must be less
                                than or equal to the ending value of the source/recording window.
                                
                                Current starting value:
                                window @ %s : start = %s
                                
                                User entered:
                                window @ %s : stop = %s
                                ''' %(slabel, sstart, slabel, args.sstop)))
                else:
                    sstart = windowDict['sstart']
                    sstep = windowDict['sstep']
                    if sstart <= args.sstop:
                        sstop = args.sstop
                    else:
                        sys.exit(textwrap.dedent(
                                '''
                                Error: Starting value of the source/recording window must be less
                                than or equal to the ending value of the source/recording window.
                                
                                Current starting value:
                                window @ %s : start = %s
                                
                                User entered:
                                window @ %s : stop = %s
                                ''' %(slabel, sstart, slabel, args.sstop)))
            elif 1 > args.sstop or args.sstop > Ns:
                sys.exit(textwrap.dedent(
                        '''
                        Error: Ending value of the source/recording window must lie within
                        the range of sources/recordings [%s, %s].
                        
                        User entered:
                        window @ %s : stop = %s
                        ''' %(1, Ns, slabel, args.sstop)))

        elif all(v is None for v in [args.sources, args.sstart, args.sstop]) and args.sstep is not None:
            if args.sstep > 0:
                sstep = args.sstep
                if windowDict is None:
                    sstart = 0
                    sstop = Ns
                else:
                    sstart = windowDict['sstart']
                    sstop = windowDict['sstop']
            elif args.sstep <= 0:
                sys.exit(textwrap.dedent(
                        '''
                        Error: Step size or spacing between array indices must
                        be a positive integer greater than or equal to 1.
                        
                        User entered:
                        window @ %s : step = %s
                        ''' %(slabel, args.sstep)))
        
        elif args.sources is not None and any(v is not None for v in [args.sstart, args.sstop, args.sstep]):
            sys.exit(textwrap.dedent(
                    '''
                    Error: Cannot use --sources together with any of --sstart/sstop/sstep.
                    Must use either --sources by itself or a combination of --sstart/sstop/sstep.
                    '''))
    
        #==============================================================================
        # set/update the receiver window
        if all(v is None for v in [args.receivers, args.rstart, args.rstop, args.rstep]):
            if windowDict is None:
                rstart = 0
                rstop = Nr
                rstep = 1
            else:
                rstart = windowDict['rstart']
                rstop = windowDict['rstop']
                rstep = windowDict['rstep']
                
        elif args.receivers is not None and all(v is None for v in [args.rstart, args.rstop, args.rstep]):
            userReceivers = args.receivers.split(',')
            if len(userReceivers) != 3:
                sys.exit(textwrap.dedent(
                        '''
                        Error: Must specify three values when using --receivers parameter.
                        Syntax: --receivers=start,stop,step
                        '''))
            if 1 <= int(userReceivers[0]) and int(userReceivers[0]) <= int(userReceivers[1]) and int(userReceivers[1]) <= Nr and int(userReceivers[2]) > 0:
                rstart = int(userReceivers[0]) - 1
                rstop = int(userReceivers[1])
                rstep = int(userReceivers[2])
            elif int(userReceivers[0]) > int(userReceivers[1]):
                sys.exit(textwrap.dedent(
                        '''
                        Error: Starting value of the receiver window must be less
                        than or equal to the ending value of the receiver window.
                                
                        User entered:
                        window @ receivers : start = %s
                        window @ receivers : stop = %s
                        ''' %(int(userReceivers[0]), int(userReceivers[1]))))
            elif 1 > int(userReceivers[0]) or int(userReceivers[0]) > Nr:
                sys.exit(textwrap.dedent(
                        '''
                        Error: Starting value of the receiver window must lie within
                        the range of receivers [%s, %s].
                        
                        User entered:
                        window @ receivers : start = %s
                        ''' %(1, Nr, int(userReceivers[0]))))
            elif 1 > int(userReceivers[1]) or int(userReceivers[1]) > Nr:
                sys.exit(textwrap.dedent(
                        '''
                        Error: Ending value of the receiver window must lie within
                        the range of receivers [%s, %s].
                        
                        User entered:
                        window @ receivers : stop = %s
                        ''' %(1, Nr, int(userReceivers[1]))))
            elif int(userReceivers[2]) <= 0:
                sys.exit(textwrap.dedent(
                        '''
                        Error: Step size or spacing between array indices must
                        be a positive integer greater than or equal to 1.
                        
                        User entered:
                        window @ receivers : step = %s
                        ''' %(int(userReceivers[2]))))
            
        elif args.receivers is None and all(v is not None for v in [args.rstart, args.rstop, args.rstep]):
            if 1 <= args.rstart and args.rstart <= args.rstop and args.rstop <= Nr and args.rstep > 0:
                rstart = args.rstart - 1
                rstop = args.rstop
                rstep = args.rstep
            elif args.rstart > args.rstop:
                sys.exit(textwrap.dedent(
                        '''
                        Error: Starting value of the receiver window must be less
                        than or equal to the ending value of the receiver window.
                                
                        User entered:
                        window @ receivers : start = %s
                        window @ receivers : stop = %s
                        ''' %(args.rstart, args.rstop)))
            elif 1 > args.rstart or args.rstart > Nr:
                sys.exit(textwrap.dedent(
                        '''
                        Error: Starting value of the receiver window must lie within
                        the range of receivers [%s, %s].
                        
                        User entered:
                        window @ receivers : start = %s
                        ''' %(1, Nr, args.rstart)))
            elif 1 > args.rstop or args.rstop > Nr:
                sys.exit(textwrap.dedent(
                        '''
                        Error: Ending value of the receiver window must lie within
                        the range of receivers [%s, %s].
                        
                        User entered:
                        window @ receivers : stop = %s
                        ''' %(1, Nr, args.rstop)))
            elif args.rstep <= 0:
                sys.exit(textwrap.dedent(
                        '''
                        Error: Step size or spacing between array indices must
                        be a positive integer greater than or equal to 1.
                        
                        User entered:
                        window @ receivers : step = %s
                        ''' %(args.rstep)))
        
        elif all(v is None for v in [args.receivers, args.rstart]) and all(v is not None for v in [args.rstop, args.rstep]):
            if 1 <= args.rstop and args.rstop <= Nr and args.rstep > 0:
                rstep = args.rstep
                if windowDict is None:
                    rstart = 0
                    if rstart <= args.rstop:
                        rstop = args.rstop
                    else:
                        sys.exit(textwrap.dedent(
                                '''
                                Error: Starting value of the receiver window must be less
                                than or equal to the ending value of the receiver window.
                                
                                Current starting value:
                                window @ receivers : start = %s
                                
                                User entered:
                                window @ receivers : stop = %s
                                ''' %(rstart, args.rstop)))
                else:
                    rstart = windowDict['rstart']
                    if rstart <= args.rstop:
                        rstop = args.rstop
                    else:
                        sys.exit(textwrap.dedent(
                                '''
                                Error: Starting value of the receiver window must be less
                                than or equal to the ending value of the receiver window.
                                
                                Current starting value:
                                window @ receivers : start = %s
                                
                                User entered:
                                window @ receivers : stop = %s
                                ''' %(rstart, args.rstop)))
            elif 1 > args.rstop or args.rstop > Nr:
                sys.exit(textwrap.dedent(
                        '''
                        Error: Ending value of the receiver window must lie within
                        the range of receivers [%s, %s].
                        
                        User entered:
                        window @ receivers : stop = %s
                        ''' %(1, Nr, args.rstop)))
            elif args.rstep <= 0:
                sys.exit(textwrap.dedent(
                        '''
                        Error: Step size or spacing between array indices must
                        be a positive integer greater than or equal to 1.
                        
                        User entered:
                        window @ receivers : step = %s
                        ''' %(args.rstep)))
    
        elif all(v is None for v in [args.receivers, args.rstop]) and all(v is not None for v in [args.rstart, args.rstep]):
            if 1 <= args.rstart and args.rstart <= Nr and args.rstep > 0:
                rstep = args.rstep
                if windowDict is None:
                    rstop = Nr
                    if args.rstart <= rstop:
                        rstart = args.rstart - 1
                    else:
                        sys.exit(textwrap.dedent(
                                '''
                                Error: Starting value of the receiver window must be less
                                than or equal to the ending value of the receiver window.
                                
                                User entered:
                                window @ receivers : start = %s
                                
                                Current ending value:
                                window @ receivers : stop = %s
                                ''' %(args.rstart, rstop)))
                        
                else:
                    rstop = windowDict['rstop']
                    if args.rstart <= rstop:
                        rstart = args.rstart - 1
                    else:
                        sys.exit(textwrap.dedent(
                                '''
                                Error: Starting value of the receiver window must be less
                                than or equal to the ending value of the receiver window.
                                
                                User entered:
                                window @ receivers : start = %s
                                
                                Current ending value:
                                window @ receivers : stop = %s
                                ''' %(args.rstart, rstop)))
            elif 1 > args.rstart or args.rstart > Nr:
                sys.exit(textwrap.dedent(
                        '''
                        Error: Starting value of the receiver window must lie within
                        the range of receivers [%s, %s].
                        
                        User entered:
                        window @ receivers : start = %s
                        ''' %(1, Nr, args.rstart)))
            elif args.rstep <= 0:
                sys.exit(textwrap.dedent(
                        '''
                        Error: Step size or spacing between array indices must
                        be a positive integer greater than or equal to 1.
                        
                        User entered:
                        window @ receivers : step = %s
                        ''' %(args.rstep)))
            
        elif all(v is None for v in [args.receivers, args.rstep]) and all(v is not None for v in [args.rstart, args.rstop]):
            if 1 <= args.rstart and args.rstart <= args.rstop and args.rstop <= Nr:
                rstart = args.rstart - 1
                rstop = args.rstop
                if windowDict is None:
                    rstep = 1
                else:
                    rstep = windowDict['rstep']
            elif args.rstart > args.rstop:
                sys.exit(textwrap.dedent(
                                '''
                                Error: Starting value of the receiver window must be less
                                than or equal to the ending value of the receiver window:
                                
                                User entered:
                                window @ receivers : start = %s
                                window @ receivers : stop = %s
                                ''' %(args.rstart, args.rstop)))
            elif 1 > args.rstart or args.rstart > Nr:
                sys.exit(textwrap.dedent(
                        '''
                        Error: Starting value of the receiver window must lie within
                        the range of receivers [%s, %s].
                        
                        User entered:
                        window @ receivers : start = %s
                        ''' %(1, Nr, args.rstart)))
            elif 1 > args.rstop or args.rstop > Nr:
                sys.exit(textwrap.dedent(
                        '''
                        Error: Ending value of the receiver window must lie within
                        the range of receivers [%s, %s].
                        
                        User entered:
                        window @ receivers : stop = %s
                        ''' %(1, Nr, args.rstop)))
    
        elif all(v is None for v in [args.receivers, args.rstop, args.rstep]) and args.rstart is not None:
            if 1 <= args.rstart and args.rstart <= Nr:
                if windowDict is None:
                    rstop = Nr
                    rstep = 1
                    if args.rstart <= rstop:
                        rstart = args.rstart - 1
                    else:
                        sys.exit(textwrap.dedent(
                                '''
                                Error: Starting value of the receiver window must be less
                                than or equal to the ending value of the receiver window:
                                
                                User entered:
                                window @ receivers : start = %s
                                
                                Current ending value:
                                window @ receivers : stop = %s
                                ''' %(args.rstart, rstop)))
                else:
                    rstop = windowDict['rstop']
                    rstep = windowDict['rstep']
                    if args.rstart <= rstop:
                        rstart = args.rstart - 1
                    else:
                        sys.exit(textwrap.dedent(
                                '''
                                Error: Starting value of the receiver window must be less
                                than or equal to the ending value of the receiver window:
                                
                                User entered:
                                window @ receivers : start = %s
                                
                                Current ending value:
                                window @ receivers : stop = %s
                                ''' %(args.rstart, rstop)))
            elif 1 > args.rstart or args.rstart > Nr:
                sys.exit(textwrap.dedent(
                        '''
                        Error: Starting value of the receiver window must lie within
                        the range of receivers [%s, %s].
                        
                        User entered:
                        window @ receivers : start = %s
                        ''' %(1, Nr, args.rstart)))
    
        elif all(v is None for v in [args.receivers, args.rstart, args.rstep]) and args.rstop is not None:
            if 1 <= args.rstop and args.rstop <= Nr:
                if windowDict is None:
                    rstart = 0
                    rstep = 1
                    if rstart <= args.rstop:
                        rstop = args.rstop
                    else:
                        sys.exit(textwrap.dedent(
                                '''
                                Error: Starting value of the receiver window must be less
                                than or equal to the ending value of the receiver window:
                                
                                Current starting value:
                                window @ receivers : start = %s
                                
                                User entered:
                                window @ receivers : stop = %s
                                ''' %(rstart, args.rstop)))
                else:
                    rstart = windowDict['rstart']
                    rstep = windowDict['rstep']
                    if rstart <= args.rstop:
                        rstop = args.rstop
                    else:
                        sys.exit(textwrap.dedent(
                                '''
                                Error: Starting value of the receiver window must be less
                                than or equal to the ending value of the receiver window:
                                
                                Current starting value:
                                window @ receivers : start = %s
                                
                                User entered:
                                window @ receivers : stop = %s
                                ''' %(rstart, args.rstop)))
            elif 1 > args.rstop or args.rstop > Nr:
                sys.exit(textwrap.dedent(
                        '''
                        Error: Ending value of the receiver window must lie within
                        the range of receivers [%s, %s].
                        
                        User entered:
                        window @ receivers : stop = %s
                        ''' %(1, Nr, args.rstop)))
    
        elif all(v is None for v in [args.receivers, args.rstart, args.rstop]) and args.rstep is not None:
            if args.rstep > 0:
                rstep = args.rstep
                if windowDict is None:
                    rstart = 0
                    rstop = Nr
                else:
                    rstart = windowDict['rstart']
                    rstop = windowDict['rstop']
            elif args.rstep <= 0:
                sys.exit(textwrap.dedent(
                        '''
                        Error: Step size or spacing between array indices must
                        be a positive integer greater than or equal to 1.
                        
                        User entered:
                        window @ receivers : step = %s
                        ''' %(args.rstep)))
        
        elif args.receivers is not None and any(v is not None for v in [args.rstart, args.rstop, args.rstep]):
            sys.exit(textwrap.dedent(
                    '''
                    Error: Cannot use --receivers together with any of --rstart/rstop/rstep.
                    Must use either --receivers by itself or a combination of --rstart/rstop/rstep.
                    '''))
        
        #==============================================================================    
        print('\nSetting up window parameters for data volume:\n')
        
        # For display/printing purposes, count receivers with one-based
        # indexing. This amounts to incrementing the rstart parameter by 1
        print('window @ receivers : start =', rstart + 1)
        print('window @ receivers : stop =', rstop)
        print('window @ receivers : step =', rstep, '\n')
            
        print('window @ time : start =', tstart)
        print('window @ time : stop =', tstop)
        print('window @ time : step =', tstep, '\n')
             
        # For display/printing purposes, count recordings/sources with one-based
        # indexing. This amounts to incrementing the sstart parameter by 1
        print('window @ %s : start = %s' %(slabel, sstart + 1))
        print('window @ %s : stop = %s' %(slabel, sstop))
        print('window @ %s : step = %s\n' %(slabel, sstep))
            
        np.savez('window.npz',
                 tstart=tstart,
                 tstop=tstop,
                 tstep=tstep,
                 slabel=slabel,
                 sstart=sstart,
                 sstop=sstop,
                 sstep=sstep,
                 rstart=rstart,
                 rstop=rstop,
                 rstep=rstep)
