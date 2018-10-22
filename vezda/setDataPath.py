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

def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default=None,
                        help='''specify the path to the directory containing the
                        experimental data.''')
    args = parser.parse_args()
    
    if args.path is None:
        if not Path('datadir.npz').exists():
            sys.exit(textwrap.dedent(
                    '''
                    A relative path to the data directory has not been specified for
                    the current working directory. To specify a relative path to the
                    data directory, enter:
                        
                        vzdata --path=<path/to/data/files>
                        
                    from the command line.
                    '''))
        
        elif Path('datadir.npz').exists():
            datadir = np.load('datadir.npz')
            dataPath = str(datadir['path'])
            files = list(datadir['files'])
            for f in range(len(files)):
                files[f] = files[f].split('/')[-1]
                
            print('\nCurrent data directory:\n')
            print(dataPath, '\n')
            print('Known files:\n')
            print(*files, sep='\n')
            sys.exit('')
    
    elif args.path is not None:
        dataPath = os.path.abspath(args.path)
        #==============================================================================
        if Path(os.path.join(dataPath, 'receiverPoints.npy')).exists():
            receivers = os.path.join(dataPath, 'receiverPoints.npy')
        else:
            userResponded = False
            print(textwrap.dedent(
                 '''
                 Error: Expected file \'receiverPoints.npy\' not found. Does a file
                 exist containing the receiver coordinates? (This is a required file.)
                 
                 Enter 'y/yes' to specify the filename containing the receiver coordinates
                 (must be binary NumPy '.npy' format). (Default)
                 Enter 'n/no' or 'q/quit to exit this program.
                 '''))
            while userResponded == False:
                answer = input('Action: ')
                if answer == '' or answer == 'y' or answer == 'yes':
                    receiverFile = input('Please specify the filename containing the receiver coordinates: ')
                    if '.npy' in receiverFile and Path(os.path.join(dataPath, receiverFile)).exists():
                        receivers = os.path.join(dataPath, receiverFile)
                        userResponded = True
                    elif '.npy' in receiverFile and not Path(os.path.join(dataPath, receiverFile)).exists():
                        print(textwrap.dedent(
                              '''
                              Error: File \'%s\' does not exist within the specified data directory.
                              
                              Enter 'y/yes' to specify another filename containing the receiver coordinates
                              (must be binary NumPy '.npy' format). (Default)
                              Enter 'n/no' or 'q/quit to exit this program.
                              ''' %(receiverFile)))
                    elif '.npy' not in receiverFile:
                        print(textwrap.dedent(
                              '''
                              Error: File \'%s\' is not NumPy '.npy' format.
                              
                              Enter 'y/yes' to specify another filename containing the receiver coordinates
                              (must be binary NumPy '.npy' format). (Default)
                              Enter 'n/no' or 'q/quit to exit this program.
                              ''' %(receiverFile)))
                elif answer == 'n' or answer == 'no' or answer == 'q' or answer == 'quit':
                    sys.exit('Exiting program.\n')                
                else:
                    print('Invalid response. Please enter \'y/yes\', \'n/no\', or \'q/quit\'.')
        
        #==============================================================================
        if Path(os.path.join(dataPath, 'sourcePoints.npy')).exists():
            sources = os.path.join(dataPath, 'sourcePoints.npy')
            noSources = False
        else:
            userResponded = False
            print(textwrap.dedent(
                 '''
                 Warning: Expected file \'sourcesPoints.npy\' not found. Does a file
                 exist containing the source coordinates? (This is NOT a required file.)
                 
                 Enter 'y/yes' to specify the filename containing the source coordinates
                 (must be binary NumPy '.npy' format).
                 Enter 'n/no' to proceed without specifying the source coordinates. (Default)
                 Enter 'q/quit to exit this program.
                 '''))
            while userResponded == False:
                answer = input('Action: ')
                if answer == 'y' or answer == 'yes':
                    sourceFile = input('Please specify the filename containing the source coordinates: ')
                    if '.npy' in sourceFile and Path(os.path.join(dataPath, sourceFile)).exists():
                        sources = os.path.join(dataPath, sourceFile)
                        noSources = False
                        userResponded = True
                    elif '.npy' in sourceFile and not Path(os.path.join(dataPath, sourceFile)).exists():
                        print(textwrap.dedent(
                              '''
                              Error: File \'%s\' does not exist within the specified data directory.
                                
                              Enter 'y/yes' to specify another filename containing the source coordinates
                              (must be binary NumPy '.npy' format).
                              Enter 'n/no' to proceed without specifying the source coordinates. (Default)
                              Enter 'q/quit to exit this program.
                              ''' %(sourceFile)))
                    elif '.npy' not in sourceFile:
                        print(textwrap.dedent(
                              '''
                              Error: File \'%s\' is not NumPy '.npy' format.
                              
                              Enter 'y/yes' to specify another filename containing the source coordinates
                              (must be binary NumPy '.npy' format).
                              Enter 'n/no' to proceed without specifying the source coordinates. (Default)
                              Enter 'q/quit to exit this program.
                              ''' %(sourceFile)))
                elif answer == '' or answer == 'n' or answer == 'no':
                    print('Proceeding without specifying the source coordinates.')
                    noSources = True
                    userResponded = True
                elif answer == 'q' or answer == 'quit':
                    sys.exit('Exiting program.\n')                
                else:
                    print('Invalid response. Please enter \'y/yes\', \'n/no\', or \'q/quit\'.')
    
        #==============================================================================
        if Path(os.path.join(dataPath, 'scattererPoints.npy')).exists():
            scatterer = os.path.join(dataPath, 'scattererPoints.npy')
            noScatterer = False
        else:
            userResponded = False
            print(textwrap.dedent(
                 '''
                 Warning: Expected file \'scattererPoints.npy\' not found. Does a file
                 exist containing the scatterer coordinates? (This is NOT a required file.)
                 
                 Enter 'y/yes' to specify the filename containing the scatterer coordinates
                 (must be binary NumPy '.npy' format).
                 Enter 'n/no' to proceed without specifying the scatterer coordinates. (Default)
                 Enter 'q/quit' to exit this program.
                 '''))
            while userResponded == False:
                answer = input('Action: ')
                if answer == 'y' or answer == 'yes':
                    scattererFile = input('Please specify the filename containing the scatterer coordinates: ')
                    if '.npy' in scattererFile and Path(os.path.join(dataPath, scattererFile)).exists():
                        scatterer = os.path.join(dataPath, scattererFile)
                        noScatterer = False
                        userResponded = True
                    elif '.npy' in scattererFile and not Path(os.path.join(dataPath, scattererFile)).exists():
                        print(textwrap.dedent(
                              '''
                              Error: File \'%s\' does not exist within the specified data directory.
                                
                              Enter 'y/yes' to specify another filename containing the scatterer coordinates
                              (must be binary NumPy '.npy' format).
                              Enter 'n/no' to proceed without specifying the scatterer coordinates. (Default)
                              Enter 'q/quit' to exit this program.
                              ''' %(scattererFile)))
                    elif '.npy' not in scattererFile:
                        print(textwrap.dedent(
                              '''
                              Error: File \'%s\' is not NumPy '.npy' format.
                              
                              Enter 'y/yes' to specify another filename containing the scatterer coordinates
                              (must be binary NumPy '.npy' format).
                              Enter 'n/no' to proceed without specifying the scatterer coordinates. (Default)
                              Enter 'q/quit' to exit this program.
                              ''' %(scattererFile)))
                elif answer == '' or answer == 'n' or answer == 'no':
                    print('Proceeding without specifying the scatterer coordinates.')
                    noScatterer = True
                    userResponded = True
                elif answer == 'q' or answer == 'quit':
                    sys.exit('Exiting program.\n')
                else:
                    print('Invalid response. Please enter \'y/yes\', \'n/no\', or \'q/quit\'.')
    
        #==============================================================================    
        if Path(os.path.join(dataPath, 'recordingTimes.npy')).exists():
            recordingTimes = os.path.join(dataPath, 'recordingTimes.npy')
        else:
            userResponded = False
            print(textwrap.dedent(
                 '''
                 Error: Expected file \'recordingTimes.npy\' not found. Does a file
                 exist containing the recording times? (This is a required file.)
                 
                 Enter 'y/yes' to specify the filename containing the recording
                 times (must be binary NumPy '.npy' format). (Default)
                 Enter 'n/no' or 'q/quit to exit this program.
                 '''))
            while userResponded == False:
                answer = input('Action: ')
                if answer == '' or answer == 'y' or answer == 'yes':
                    timeFile = input('Please specify the filename containing the recording times: ')
                    if '.npy' in timeFile and Path(os.path.join(dataPath, timeFile)).exists():
                        recordingTimes = os.path.join(dataPath, timeFile)
                        userResponded = True
                    elif '.npy' in timeFile and not Path(os.path.join(dataPath, timeFile)).exists():
                        print(textwrap.dedent(
                              '''
                              Error: File \'%s\' does not exist within the specified data directory.
                              
                              Enter 'y/yes' to specify another filename containing the recording times
                              (must be binary NumPy '.npy' format). (Default)
                              Enter 'n/no' or 'q/quit to exit this program.
                              ''' %(timeFile)))
                    elif '.npy' not in timeFile:
                        print(textwrap.dedent(
                              '''
                              Error: File \'%s\' is not NumPy '.npy' format.
                              
                              Enter 'y/yes' to specify another filename containing the recording times
                              (must be binary NumPy '.npy' format). (Default)
                              Enter 'n/no' or 'q/quit to exit this program.
                              ''' %(timeFile)))
                elif answer == 'n' or answer == 'no' or answer == 'q' or answer == 'quit':
                    sys.exit('Exiting program.\n')                
                else:
                    print('Invalid response. Please enter \'y/yes\', \'n/no\', or \'q/quit\'.')
    
        #==============================================================================    
        if Path(os.path.join(dataPath, 'recordedData.npy')).exists():
            recordedData = os.path.join(dataPath, 'recordedData.npy')
        else:
            userResponded = False
            print(textwrap.dedent(
                 '''
                 Error: Expected file \'recordedData.npy\' not found. Does a file
                 exist containing the recorded waves? (This is a required file.)
                 
                 Enter 'y/yes' to specify the filename containing the recorded
                 waves (must be binary NumPy '.npy' format). (Default)
                 Enter 'n/no' or 'q/quit to exit this program.
                 '''))
            while userResponded == False:
                answer = input('Action: ')
                if answer == '' or answer == 'y' or answer == 'yes':
                    dataFile = input('Please specify the filename containing the recorded waves: ')
                    if '.npy' in dataFile and Path(os.path.join(dataPath, dataFile)).exists():
                        recordedData = os.path.join(dataPath, dataFile)
                        userResponded = True
                    elif '.npy' in dataFile and not Path(os.path.join(dataPath, dataFile)).exists():
                        print(textwrap.dedent(
                              '''
                              Error: File \'%s\' does not exist within the specified data directory.
                              
                              Enter 'y/yes' to specify another filename containing the recorded waves
                              (must be binary NumPy '.npy' format). (Default)
                              Enter 'n/no' or 'q/quit to exit this program.
                              ''' %(dataFile)))
                    elif '.npy' not in dataFile:
                        print(textwrap.dedent(
                              '''
                              Error: File \'%s\' is not NumPy '.npy' format.
                              
                              Enter 'y/yes' to specify another filename containing the recorded waves
                              (must be binary NumPy '.npy' format). (Default)
                              Enter 'n/no' or 'q/quit to exit this program.
                              ''' %(dataFile)))
                elif answer == 'n' or answer == 'no' or answer == 'q' or answer == 'quit':
                    sys.exit('Exiting program.\n')                
                else:
                    print('Invalid response. Please enter \'y/yes\', \'n/no\', or \'q/quit\'.')
    
        #==============================================================================
        if Path(os.path.join(dataPath, 'testFunctions.npy')).exists():
            testFuncs = os.path.join(dataPath, 'testFunctions.npy')
            noTestFuncs = False
        else:
            userResponded = False
            print(textwrap.dedent(
                 '''
                 Warning: Expected file \'testFunctions.npy\' not found. Does a file
                 exist containing the simulated test functions? (This is NOT a required file.)
                 
                 Enter 'y/yes' to specify the filename containing the simulated test functions
                 (must be binary NumPy '.npy' format).
                 Enter 'n/no' to proceed without specifying the test functions. (Default)
                 Enter 'q/quit' to exit this program.
                 '''))
            while userResponded == False:
                answer = input('Action: ')
                if answer == 'y' or answer == 'yes':
                    testFuncsFile = input('Please specify the filename containing the simulated test functions: ')
                    if '.npy' in testFuncsFile and Path(os.path.join(dataPath, testFuncsFile)).exists():
                        testFuncs = os.path.join(dataPath, testFuncsFile)
                        noTestFuncs = False
                        userResponded = True
                    elif '.npy' in testFuncsFile and not Path(os.path.join(dataPath, testFuncsFile)).exists():
                        print(textwrap.dedent(
                              '''
                              Error: File \'%s\' does not exist within the specified data directory.
                              
                              Enter 'y/yes' to specify another filename containing the simulated test functions
                              (must be binary NumPy '.npy' format).
                              Enter 'n/no' to proceed without specifying the test functions. (Default)
                              Enter 'q/quit' to exit this program.
                              ''' %(testFuncsFile)))
                    elif '.npy' not in testFuncsFile:
                        print(textwrap.dedent(
                              '''
                              Error: File \'%s\' is not NumPy '.npy' format.
                              
                              Enter 'y/yes' to specify another filename containing the simulated test functions
                              (must be binary NumPy '.npy' format).
                              Enter 'n/no' to proceed without specifying the test functions. (Default)
                              Enter 'q/quit' to exit this program.
                              ''' %(testFuncsFile)))
                elif answer == '' or answer == 'n' or answer == 'no':
                    print('Proceeding without specifying the simulated test functions.')
                    noTestFuncs = True
                    userResponded = True
                elif answer == 'q' or answer == 'quit':
                    sys.exit('Exiting program.\n')
                else:
                    print('Invalid response. Please enter \'y/yes\', \'n/no\', or \'q/quit\'.')
    
        #==============================================================================
        # If a user is supplying test functions, the sampling points from which those
        # test functions were generated must also be supplied.
        if noTestFuncs == False:
            if Path(os.path.join(dataPath, 'samplingPoints.npy')).exists():
                samplingPoints = os.path.join(dataPath, 'samplingPoints.npy')
            else:
                userResponded = False
                print(textwrap.dedent(
                     '''
                     Error: Expected file \'samplingPoints.npy\' not found. Does a file
                     exist containing the space-time sampling points? (This file is
                     required only if test functions are provided.)
                         
                     Enter 'y/yes' to specify the filename containing the sampling points
                     (must be binary NumPy '.npy' format). (Default)
                     Enter 'n/no' or 'q/quit' to exit this program.
                     '''))
                while userResponded == False:
                    answer = input('Action: ')
                    if answer == '' or answer == 'y' or answer == 'yes':
                        samplingPointsFile = input('Please specify the filename containing the sampling points: ')
                        if '.npy' in samplingPointsFile and Path(os.path.join(dataPath, samplingPointsFile)).exists():
                            samplingPoints = os.path.join(dataPath, samplingPointsFile)
                            userResponded = True
                        elif '.npy' in samplingPointsFile and not Path(os.path.join(dataPath, samplingPointsFile)).exists():
                            print(textwrap.dedent(
                                  '''
                                  Error: File \'%s\' does not exist within the specified data directory.
                                  
                                  Enter 'y/yes' to specify another filename containing the sampling points
                                  (must be binary NumPy '.npy' format). (Default)
                                  Enter 'n/no' or 'q/quit' to exit this program.
                                  ''' %(samplingPointsFile)))
                        elif '.npy' not in samplingPointsFile:
                            print(textwrap.dedent(
                                  '''
                                  Error: File \'%s\' is not NumPy '.npy' format.
                                  
                                  Enter 'y/yes' to specify another filename containing the sampling points
                                  (must be binary NumPy '.npy' format). (Default)
                                  Enter 'n/no' or 'q/quit' to exit this program.
                                  ''' %(samplingPointsFile)))
                    elif answer == 'n' or answer == 'no' or answer == 'q' or answer == 'quit':
                        sys.exit('Exiting program.\n')
                    else:
                        print('Invalid response. Please enter \'y/yes\', \'n/no\', or \'q/quit\'.')
        
        #==============================================================================  
        if noSources and noScatterer and noTestFuncs:
            files = [receivers, recordingTimes, recordedData]
            np.savez('datadir.npz',
                     path = dataPath,
                     files = files,
                     receivers = receivers,
                     recordingTimes = recordingTimes,
                     recordedData = recordedData)
        elif noSources and noScatterer and not noTestFuncs:
            files = [receivers, recordingTimes, recordedData, testFuncs, samplingPoints]
            np.savez('datadir.npz',
                     path = dataPath,
                     files = files,
                     receivers = receivers,
                     recordingTimes = recordingTimes,
                     recordedData = recordedData,
                     testFuncs = testFuncs,
                     samplingPoints = samplingPoints)
        elif noSources and not noScatterer and noTestFuncs:
            files = [receivers, scatterer, recordingTimes, recordedData]
            np.savez('datadir.npz',
                     path = dataPath,
                     files = files,
                     receivers = receivers,
                     scatterer = scatterer,
                     recordingTimes = recordingTimes,
                     recordedData = recordedData)
        elif not noSources and noScatterer and noTestFuncs:
            files = [receivers, sources, recordingTimes, recordedData]
            np.savez('datadir.npz',
                     path = dataPath,
                     files = files,
                     receivers = receivers,
                     sources = sources,
                     recordingTimes = recordingTimes,
                     recordedData = recordedData)
        elif noSources and not noScatterer and not noTestFuncs:
            files = [receivers, scatterer, recordingTimes, recordedData, testFuncs, samplingPoints]
            np.savez('datadir.npz',
                     path = dataPath,
                     files = files,
                     receivers = receivers,
                     scatterer = scatterer,
                     recordingTimes = recordingTimes,
                     recordedData = recordedData,
                     testFuncs = testFuncs,
                     samplingPoints = samplingPoints)
        elif not noSources and noScatterer and not noTestFuncs:
            files = [receivers, sources, recordingTimes, recordedData, testFuncs, samplingPoints]
            np.savez('datadir.npz',
                     path = dataPath,
                     files = files,
                     receivers = receivers,
                     sources = sources,
                     recordingTimes = recordingTimes,
                     recordedData = recordedData,
                     testFuncs = testFuncs,
                     samplingPoints = samplingPoints)
        elif not noSources and not noScatterer and noTestFuncs:
            files = [receivers, sources, scatterer, recordingTimes, recordedData]
            np.savez('datadir.npz',
                     path = dataPath,
                     files = files,
                     receivers = receivers,
                     sources = sources,
                     scatterer = scatterer,
                     recordingTimes = recordingTimes,
                     recordedData = recordedData)
        else:
            files = [receivers, sources, scatterer, recordingTimes, recordedData, testFuncs, samplingPoints]
            np.savez('datadir.npz',
                     path = dataPath,
                     files = files,
                     receivers = receivers,
                     sources = sources,
                     scatterer = scatterer,
                     recordingTimes = recordingTimes,
                     recordedData = recordedData,
                     testFuncs = testFuncs,
                     samplingPoints = samplingPoints)