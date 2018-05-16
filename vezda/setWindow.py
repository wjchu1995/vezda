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
import numpy as np

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
    
    datadir = np.load('datadir.npz')
    recordingTimes = np.load(str(datadir['recordingTimes']))
    receivers = np.load(str(datadir['receivers']))
    sources = np.load(str(datadir['sources']))
    
    #==============================================================================
    # set time window
    if args.time != None and args.tstart == None and args.tstop == None and args.tstep == None:
        time = args.time.split(',')
        if len(time) != 3:
            sys.exit('''
                     Error: Must specify three values when using --time parameter.
                     Syntax: --time=start,stop,step
                     ''')
        if recordingTimes[0] <= float(time[0]) and float(time[0]) < float(time[1]) and float(time[1]) <= recordingTimes[-1] and int(time[2]) > 0:
            tstart = float(time[0])
            tstop = float(time[1])
            tstep = int(time[2])
        elif float(time[0]) >= float(time[1]):
            sys.exit('''
                     Error: Starting value of the time window must be less
                     than the ending value of the time window.
                     ''')
        elif recordingTimes[0] > float(time[0]) or float(time[0]) > recordingTimes[-1]:
            sys.exit('''
                     Error: Starting value of the time window must lie within
                     the recorded time interval (%s, %s).
                     ''' %(recordingTimes[0], recordingTimes[-1]))
        elif recordingTimes[0] > float(time[1]) or float(time[1]) > recordingTimes[-1]:
            sys.exit('''
                     Error: Ending value of the time window must lie within
                     the recorded time interval (%s, %s).
                     ''' %(recordingTimes[0], recordingTimes[-1]))
        elif int(time[2]) <= 0:
            sys.exit('''
                     Error: Step size or spacing between array indices must
                     be a positive integer greater than or equal to 1.
                     ''')
            
    elif args.time == None and args.tstart != None and args.tstop != None and args.tstep != None:
        if recordingTimes[0] <= args.tstart and args.tstart < args.tstop and args.tstop <= recordingTimes[-1] and args.tstep > 0:
            tstart = args.tstart
            tstop = args.tstop
            tstep = args.tstep
        elif args.tstart >= args.tstop:
            sys.exit('''
                     Error: Starting value of the time window must be less
                     than the ending value of the time window.
                     ''')
        elif recordingTimes[0] > args.tstart or args.tstart > recordingTimes[-1]:
            sys.exit('''
                     Error: Starting value of the time window must lie within
                     the recorded time interval (%s, %s).
                     ''' %(recordingTimes[0], recordingTimes[-1]))
        elif recordingTimes[0] > args.tstop or args.tstop > recordingTimes[-1]:
            sys.exit('''
                     Error: Ending value of the time window must lie within
                     the recorded time interval (%s, %s).
                     ''' %(recordingTimes[0], recordingTimes[-1]))
        elif args.tstep <= 0:
            sys.exit('''
                     Error: Step size or spacing between array indices must
                     be a positive integer greater than or equal to 1.
                     ''')
            
    elif args.time == None and args.tstart == None and args.tstop != None and args.tstep != None:
        if recordingTimes[0] < args.tstop and args.tstop <= recordingTimes[-1] and args.tstep > 0:
            tstart = recordingTimes[0]
            tstop = args.tstop
            tstep = args.tstep
        elif recordingTimes[0] >= args.tstop or args.tstop > recordingTimes[-1]:
            sys.exit('''
                     Error: Ending value of the time window must lie within
                     the recorded time interval (%s, %s).
                     ''' %(recordingTimes[0], recordingTimes[-1]))
        elif args.tstep <= 0:
            sys.exit('''
                     Error: Step size or spacing between array indices must
                     be a positive integer greater than or equal to 1.
                     ''')
            
    
    elif args.time == None and args.tstart != None and args.tstop == None and args.tstep != None:
        if recordingTimes[0] <= args.tstart and args.tstart < recordingTimes[-1] and args.tstep > 0:
            tstart = args.tstart
            tstop = recordingTimes[-1]
            tstep = args.tstep
        elif recordingTimes[0] > args.tstart or args.tstart >= recordingTimes[-1]:
            sys.exit('''
                     Error: Beginning value of the time window must lie within
                     the recorded time interval (%s, %s).
                     ''' %(recordingTimes[0], recordingTimes[-1]))
        elif args.tstep <= 0:
            sys.exit('''
                     Error: Step size or spacing between array indices must
                     be a positive integer greater than or equal to 1.
                     ''')
            
    elif args.time == None and args.tstart != None and args.tstop != None and args.tstep == None:
        if recordingTimes[0] <= args.tstart and args.tstart < args.tstop and args.tstop <= recordingTimes[-1]:
            tstart = args.tstart
            tstop = args.tstop
            tstep = 1
        elif args.tstart >= args.tstop:
            sys.exit('''
                     Error: Starting value of the time window must be less
                     than the ending value of the time window.
                     ''')
        elif recordingTimes[0] > args.tstart or args.tstart >= recordingTimes[-1]:
            sys.exit('''
                     Error: Beginning value of the time window must lie within
                     the recorded time interval (%s, %s).
                     ''' %(recordingTimes[0], recordingTimes[-1]))
        elif recordingTimes[0] > args.tstop or args.tstop >= recordingTimes[-1]:
            sys.exit('''
                     Error: Beginning value of the time window must lie within
                     the recorded time interval (%s, %s).
                     ''' %(recordingTimes[0], recordingTimes[-1]))
    
    elif args.time == None and args.tstart != None and args.tstop == None and args.tstep == None:
        if recordingTimes[0] <= args.tstart and args.tstart < recordingTimes[-1]:
            tstart = args.tstart
            tstop = recordingTimes[-1]
            tstep = 1
        elif recordingTimes[0] > args.tstart or args.tstart >= recordingTimes[-1]:
            sys.exit('''
                     Error: Beginning value of the time window must lie within
                     the recorded time interval (%s, %s).
                     ''' %(recordingTimes[0], recordingTimes[-1]))
    
    elif args.time == None and args.tstart == None and args.tstop != None and args.tstep == None:
        if recordingTimes[0] < args.tstop and args.tstop <= recordingTimes[-1]:
            tstart = recordingTimes[0]
            tstop = args.tstop
            tstep = 1
        elif recordingTimes[0] >= args.tstop or args.tstop > recordingTimes[-1]:
            sys.exit('''
                     Error: Ending value of the time window must lie within
                     the recorded time interval (%s, %s).
                     ''' %(recordingTimes[0], recordingTimes[-1]))
    
    elif args.time == None and args.tstart == None and args.tstop == None and args.tstep != None:
        if args.tstep > 0:
            tstart = recordingTimes[0]
            tstop = recordingTimes[-1]
            tstep = args.tstep
        elif args.tstep <= 0:
            sys.exit('''
                     Error: Step size or spacing between array indices must
                     be a positive integer greater than or equal to 1.
                     ''')
        
    elif args.time == None and args.tstart == None and args.tstop == None and args.tstep == None:
        tstart = recordingTimes[0]
        tstop = recordingTimes[-1]
        tstep = 1
        
    elif args.time != None and args.tstart != None:
        sys.exit('''
                 Error: Cannot use --time together with any of --tstart/tstop/tstep.
                 Must use either --time by itself or a combination of --tstart/tstop/tstep.
                 ''')
    
    elif args.time != None and args.tstop != None:
        sys.exit('''
                 Error: Cannot use --time together with any of --tstart/tstop/tstep.
                 Must use either --time by itself or a combination of --tstart/tstop/tstep.
                 ''')
    
    elif args.time != None and args.tstep != None:
        sys.exit('''
                 Error: Cannot use --time together with any of --tstart/tstop/tstep.
                 Must use either --time by itself or a combination of --tstart/tstop/tstep.
                 ''')
    
    #==============================================================================
    # set source window
    if args.sources != None and args.sstart == None and args.sstop == None and args.sstep == None:
        userSources = args.sources.split(',')
        if len(userSources) != 3:
            sys.exit('''
                     Error: Must specify three values when using --sources parameter.
                     Syntax: --sources=start,stop,step
                     ''')
        if 0 <= int(userSources[0]) and int(userSources[0]) < int(userSources[1]) and int(userSources[1]) <= sources.shape[0] and int(userSources[2]) > 0:
            sstart = int(userSources[0])
            sstop = int(userSources[1])
            sstep = int(userSources[2])
        elif int(userSources[0]) >= int(userSources[1]):
            sys.exit('''
                     Error: Starting value of the source window must be less
                     than the ending value of the source window.
                     ''')
        elif 0 > int(userSources[0]) or int(userSources[0]) > sources.shape[0]:
            sys.exit('''
                     Error: Starting value of the source window must lie within
                     the range of sources (%s, %s).
                     ''' %(0, sources.shape[0]))
        elif 0 > int(userSources[1]) or int(userSources[1]) > sources.shape[0]:
            sys.exit('''
                     Error: Ending value of the source window must lie within
                     the range of sources (%s, %s).
                     ''' %(0, sources.shape[0]))
        elif int(userSources[2]) <= 0:
            sys.exit('''
                     Error: Step size or spacing between array indices must
                     be a positive integer greater than or equal to 1.
                     ''')
            
    elif args.sources == None and args.sstart != None and args.sstop != None and args.sstep != None:
        if 0 <= args.sstart and args.sstart < args.sstop and args.sstop <= sources.shape[0] and args.sstep > 0:
            sstart = args.sstart
            sstop = args.sstop
            sstep = args.sstep
        elif args.sstart >= args.sstop:
            sys.exit('''
                     Error: Starting value of the source window must be less
                     than the ending value of the source window.
                     ''')
        elif 0 > args.sstart or args.sstart > sources.shape[0]:
            sys.exit('''
                     Error: Starting value of the source window must lie within
                     the range of sources (%s, %s).
                     ''' %(0, sources.shape[0]))
        elif 0 > args.sstop or args.sstop > sources.shape[0]:
            sys.exit('''
                     Error: Ending value of the source window must lie within
                     the range of sources (%s, %s).
                     ''' %(0, sources.shape[0]))
        elif args.sstep <= 0:
            sys.exit('''
                     Error: Step size or spacing between array indices must
                     be a positive integer greater than or equal to 1.
                     ''')
            
    elif args.sources == None and args.sstart == None and args.sstop != None and args.sstep != None:
        if 0 < args.sstop and args.sstop <= sources.shape[0] and args.sstep > 0:
            sstart = 0
            sstop = args.sstop
            sstep = args.sstep
        elif 0 >= args.sstop or args.sstop > sources.shape[0]:
            sys.exit('''
                     Error: Ending value of the source window must lie within
                     the range of sources (%s, %s).
                     ''' %(0, sources.shape[0]))
        elif args.sstep <= 0:
            sys.exit('''
                     Error: Step size or spacing between array indices must
                     be a positive integer greater than or equal to 1.
                     ''')
    
    elif args.sources == None and args.sstart != None and args.sstop == None and args.sstep != None:
        if 0 <= args.sstart and args.sstart < sources.shape[0] and args.sstep > 0:
            sstart = args.sstart
            sstop = sources.shape[0]
            sstep = args.sstep
        elif 0 > args.sstart or args.sstart >= sources.shape[0]:
            sys.exit('''
                     Error: Beginning value of the source window must lie within
                     the range of sources (%s, %s).
                     ''' %(0, sources.shape[0]))
        elif args.sstep <= 0:
            sys.exit('''
                     Error: Step size or spacing between array indices must
                     be a positive integer greater than or equal to 1.
                     ''')
            
    elif args.sources == None and args.sstart != None and args.sstop != None and args.sstep == None:
        if 0 <= args.sstart and args.sstart < args.sstop and args.sstop <= sources.shape[0]:
            sstart = args.sstart
            sstop = args.sstop
            sstep = 1
        elif args.sstart >= args.sstop:
            sys.exit('''
                     Error: Starting value of the source window must be less
                     than the ending value of the source window.
                     ''')
        elif 0 > args.sstart or args.sstart >= sources.shape[0]:
            sys.exit('''
                     Error: Beginning value of the source window must lie within
                     the range of sources (%s, %s).
                     ''' %(0, sources.shape[0]))
        elif 0 > args.sstop or args.sstop >= sources.shape[0]:
            sys.exit('''
                     Error: Beginning value of the source window must lie within
                     the range of sources (%s, %s).
                     ''' %(0, sources.shape[0]))
    
    elif args.sources == None and args.sstart != None and args.sstop == None and args.sstep == None:
        if 0 <= args.sstart and args.sstart < sources.shape[0]:
            sstart = args.sstart
            sstop = sources.shape[0]
            sstep = 1
        elif 0 > args.sstart or args.sstart >= sources.shape[0]:
            sys.exit('''
                     Error: Beginning value of the source window must lie within
                     the range of sources (%s, %s).
                     ''' %(0, sources.shape[0]))
    
    elif args.sources == None and args.sstart == None and args.sstop != None and args.sstep == None:
        if 0 <= args.sstop and args.sstop < sources.shape[0]:
            sstart = 0
            sstop = args.sstop
            sstep = 1
        elif 0 > args.sstart or args.sstart >= sources.shape[0]:
            sys.exit('''
                     Error: Ending value of the source window must lie within
                     the range of sources (%s, %s).
                     ''' %(0, sources.shape[0]))
    
    elif args.sources == None and args.sstart == None and args.sstop == None and args.sstep != None:
        if args.sstep > 0:
            sstart = 0
            sstop = sources.shape[0]
            sstep = args.sstep
        elif args.sstep <= 0:
            sys.exit('''
                     Error: Step size or spacing between array indices must
                     be a positive integer greater than or equal to 1.
                     ''')
        
    elif args.sources == None and args.sstart == None and args.sstop == None and args.sstep == None:
        sstart = 0
        sstop = sources.shape[0]
        sstep = 1
        
    elif args.sources != None and args.sstart != None:
        sys.exit('''
                 Error: Cannot use --sources together with any of --sstart/sstop/sstep.
                 Must use either --sources by itself or a combination of --sstart/sstop/sstep.
                 ''')
    
    elif args.sources != None and args.sstop != None:
        sys.exit('''
                 Error: Cannot use --sources together with any of --sstart/sstop/sstep.
                 Must use either --sources by itself or a combination of --sstart/sstop/sstep.
                 ''')
    
    elif args.sources != None and args.sstep != None:
        sys.exit('''
                 Error: Cannot use --sources together with any of --sstart/sstop/sstep.
                 Must use either --sourcces by itself or a combination of --sstart/sstop/sstep.
                 ''')
    
    #==============================================================================
    # set receiver window
    if args.receivers != None and args.rstart == None and args.rstop == None and args.rstep == None:
        userReceivers = args.receivers.split(',')
        if len(userReceivers) != 3:
            sys.exit('''
                     Error: Must specify three values when using --receivers parameter.
                     Syntax: --receivers=start,stop,step
                     ''')
        if 0 <= int(userReceivers[0]) and int(userReceivers[0]) < int(userReceivers[1]) and int(userReceivers[1]) <= receivers.shape[0] and int(userReceivers[2]) > 0:
            rstart = int(userReceivers[0])
            rstop = int(userReceivers[1])
            rstep = int(userReceivers[2])
        elif int(userReceivers[0]) >= int(userReceivers[1]):
            sys.exit('''
                     Error: Starting value of the receiver window must be less
                     than the ending value of the receiver window.
                     ''')
        elif 0 > int(userReceivers[0]) or int(userReceivers[0]) > receivers.shape[0]:
            sys.exit('''
                     Error: Starting value of the receiver window must lie within
                     the range of receivers (%s, %s).
                     ''' %(0, receivers.shape[0]))
        elif 0 > int(userReceivers[1]) or int(userReceivers[1]) > receivers.shape[0]:
            sys.exit('''
                     Error: Ending value of the receiver window must lie within
                     the range of receivers (%s, %s).
                     ''' %(0, receivers.shape[0]))
        elif int(userReceivers[2]) <= 0:
            sys.exit('''
                     Error: Step size or spacing between array indices must
                     be a positive integer greater than or equal to 1.
                     ''')
            
    elif args.receivers == None and args.rstart != None and args.rstop != None and args.rstep != None:
        if 0 <= args.rstart and args.rstart < args.rstop and args.rstop <= receivers.shape[0] and args.rstep > 0:
            rstart = args.rstart
            rstop = args.rstop
            rstep = args.rstep
        elif args.rstart >= args.rstop:
            sys.exit('''
                     Error: Starting value of the receiver window must be less
                     than the ending value of the receiver window.
                     ''')
        elif 0 > args.rstart or args.rstart > receivers.shape[0]:
            sys.exit('''
                     Error: Starting value of the receiver window must lie within
                     the range of receivers (%s, %s).
                     ''' %(0, receivers.shape[0]))
        elif 0 > args.rstop or args.rstop > receivers.shape[0]:
            sys.exit('''
                     Error: Ending value of the receiver window must lie within
                     the range of receivers (%s, %s).
                     ''' %(0, receivers.shape[0]))
        elif args.rstep <= 0:
            sys.exit('''
                     Error: Step size or spacing between array indices must
                     be a positive integer greater than or equal to 1.
                     ''')
            
    elif args.receivers == None and args.rstart == None and args.rstop != None and args.rstep != None:
        if 0 < args.rstop and args.rstop <= receivers.shape[0] and args.rstep > 0:
            rstart = 0
            rstop = args.rstop
            rstep = args.rstep
        elif 0 >= args.rstop or args.rstop > receivers.shape[0]:
            sys.exit('''
                     Error: Ending value of the receiver window must lie within
                     the range of receivers (%s, %s).
                     ''' %(0, receivers.shape[0]))
        elif args.rstep <= 0:
            sys.exit('''
                     Error: Step size or spacing between array indices must
                     be a positive integer greater than or equal to 1.
                     ''')
    
    elif args.receivers == None and args.rstart != None and args.rstop == None and args.rstep != None:
        if 0 <= args.rstart and args.rstart < receivers.shape[0] and args.rstep > 0:
            rstart = args.rstart
            rstop = receivers.shape[0]
            rstep = args.rstep
        elif 0 > args.rstart or args.rstart >= receivers.shape[0]:
            sys.exit('''
                     Error: Beginning value of the receiver window must lie within
                     the range of receivers (%s, %s).
                     ''' %(0, receivers.shape[0]))
        elif args.rstep <= 0:
            sys.exit('''
                     Error: Step size or spacing between array indices must
                     be a positive integer greater than or equal to 1.
                     ''')
            
    elif args.receivers == None and args.rstart != None and args.rstop != None and args.rstep == None:
        if 0 <= args.rstart and args.rstart < args.rstop and args.rstop <= receivers.shape[0]:
            rstart = args.rstart
            rstop = args.rstop
            rstep = 1
        elif args.rstart >= args.rstop:
            sys.exit('''
                     Error: Starting value of the receiver window must be less
                     than the ending value of the receiver window.
                     ''')
        elif 0 > args.rstart or args.rstart >= receivers.shape[0]:
            sys.exit('''
                     Error: Beginning value of the receiver window must lie within
                     the range of receivers (%s, %s).
                     ''' %(0, receivers.shape[0]))
        elif 0 > args.rstop or args.rstop >= receivers.shape[0]:
            sys.exit('''
                     Error: Beginning value of the receiver window must lie within
                     the range of receivers (%s, %s).
                     ''' %(0, receivers.shape[0]))
    
    elif args.receivers == None and args.rstart != None and args.rstop == None and args.rstep == None:
        if 0 <= args.rstart and args.rstart < receivers.shape[0]:
            rstart = args.rstart
            rstop = receivers.shape[0]
            rstep = 1
        elif 0 > args.rstart or args.rstart >= receivers.shape[0]:
            sys.exit('''
                     Error: Beginning value of the receiver window must lie within
                     the range of receivers (%s, %s).
                     ''' %(0, receivers.shape[0]))
    
    elif args.receivers == None and args.rstart == None and args.rstop != None and args.rstep == None:
        if 0 <= args.rstop and args.rstop < receivers.shape[0]:
            rstart = 0
            rstop = args.rstop
            rstep = 1
        elif 0 > args.rstart or args.rstart >= receivers.shape[0]:
            sys.exit('''
                     Error: Beginning value of the receiver window must lie within
                     the range of receivers (%s, %s).
                     ''' %(0, receivers.shape[0]))
    
    elif args.receivers == None and args.rstart == None and args.rstop == None and args.rstep != None:
        if args.rstep > 0:
            rstart = 0
            rstop = receivers.shape[0]
            rstep = args.rstep
        elif args.rstep <= 0:
            sys.exit('''
                     Error: Step size or spacing between array indices must
                     be a positive integer greater than or equal to 1.
                     ''')
        
    elif args.receivers == None and args.rstart == None and args.rstop == None and args.rstep == None:
        rstart = 0
        rstop = receivers.shape[0]
        rstep = 1
        
    elif args.receivers != None and args.rstart != None:
        sys.exit('''
                 Error: Cannot use --receivers together with any of --rstart/rstop/rstep.
                 Must use either --receivers by itself or a combination of --rstart/rstop/rstep.
                 ''')
    
    elif args.receivers != None and args.rstop != None:
        sys.exit('''
                 Error: Cannot use --receivers together with any of --rstart/rstop/rstep.
                 Must use either --receivers by itself or a combination of --rstart/rstop/rstep.
                 ''')
    
    elif args.receivers != None and args.rstep != None:
        sys.exit('''
                 Error: Cannot use --receivers together with any of --rstart/rstop/rstep.
                 Must use either --receivers by itself or a combination of --rstart/rstop/rstep.
                 ''')
    #==============================================================================
    
    print('window @ receivers : start = ', rstart)
    print('window @ receivers : stop = ', rstop)
    print('window @ receivers : step = ', rstep)
    
    print('window @ time : start = ', tstart)
    print('window @ time : stop = ', tstop)
    print('window @ time : step = ', tstep)
    
    print('window @ sources : start = ', sstart)
    print('window @ sources : stop = ', sstop)
    print('window @ sources : step = ', sstep)
    
    np.savez('window.npz',
             tstart=tstart,
             tstop=tstop,
             tstep=tstep,
             sstart=sstart,
             sstop=sstop,
             sstep=sstep,
             rstart=rstart,
             rstop=rstop,
             rstep=rstep)
