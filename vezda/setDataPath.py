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

def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True,
                        help='''specify the path to the directory containing the
                        experimental data.''')
    args = parser.parse_args()
    
    datadir = os.path.abspath(args.path)
    #==============================================================================
    if Path(os.path.join(datadir, 'receiverPoints.npy')).exists():
        receivers = os.path.join(datadir, 'receiverPoints.npy')
    else:
        userResponded = False
        print('''
              Error: Expected file \'receiverPoints.npy\' not found. Does a file
              exist containing the receiver coordinates? (This is a required file.)
              
              Enter 'y/yes' to specify the filename containing the coordinates of the
              receivers (must be binary NumPy '.npy' format). (Default)
              Enter 'n/no' or 'q/quit to exit this program.
              ''')
        while userResponded == False:
            answer = input('Action: ')
            if answer == '' or answer == 'y' or answer == 'yes':
                receiverFile = input('Please specify the filename containing the receiver coordinates: ')
                if '.npy' in receiverFile and Path(os.path.join(datadir, receiverFile)).exists():
                    receivers = os.path.abspath(receiverFile)
                    userResponded = True
                    break
                elif '.npy' in receiverFile and not Path(os.path.join(datadir, receiverFile)).exists():
                    print('Error: file \'%s\' does not exist within the current directory.' %(receiverFile))
                    break
                elif '.npy' not in receiverFile:
                    print('''Error: file \'%s\' is not NumPy '.npy' format.''' %(receiverFile))
                    break  
            elif answer == 'n' or answer == 'no' or answer == 'q' or answer == 'quit':
                sys.exit('Exiting program.')                
            else:
                print('Invalid response. Please enter \'y/yes\', \'n/no\', or \'q/quit\'.')
        
    #==============================================================================
    if Path(os.path.join(datadir, 'sourcePoints.npy')).exists():
        sources = os.path.join(datadir, 'sourcePoints.npy')
    else:
        userResponded = False
        print('''
              Error: Expected file \'sourcesPoints.npy\' not found. Does a file
              exist containing the source coordinates? (This is a required file.)
              
              Enter 'y/yes' to specify the filename containing the coordinates of the
              sources (must be binary NumPy '.npy' format). (Default)
              Enter 'n/no' or 'q/quit to exit this program.
              ''')
        while userResponded == False:
            answer = input('Action: ')
            if answer == '' or answer == 'y' or answer == 'yes':
                sourceFile = input('Please specify the filename containing the source coordinates: ')
                if '.npy' in sourceFile and Path(os.path.join(datadir, sourceFile)).exists():
                    sources = os.path.abspath(sourceFile)
                    userResponded = True
                    break
                elif '.npy' in sourceFile and not Path(os.path.join(datadir, sourceFile)).exists():
                    print('Error: file \'%s\' does not exist within the current directory.' %(sourceFile))
                    break
                elif '.npy' not in sourceFile:
                    print('''Error: file \'%s\' is not NumPy '.npy' format.''' %(sourceFile))
                    break  
            elif answer == 'n' or answer == 'no' or answer == 'q' or answer == 'quit':
                sys.exit('Exiting program.')                
            else:
                print('Invalid response. Please enter \'y/yes\', \'n/no\', or \'q/quit\'.')
    #==============================================================================
    if Path(os.path.join(datadir, 'scattererPoints.npy')).exists():
        scatterer = os.path.join(datadir, 'scattererPoints.npy')
        noScatterer = False
    else:
        userResponded = False
        print('''
              Warning: Expected file \'scattererPoints.npy\' not found. Does a file
              exist containing the scatterer coordinates? (This is NOT a required file.)
              
              Enter 'y/yes' to specify the filename containing the coordinates of the
              scatterer (must be binary NumPy '.npy' format).
              Enter 'n/no' to proceed without specifying the scatterer coordinates. (Default)
              Enter 'q/quit' to exit this program.
              ''')
        while userResponded == False:
            answer = input('Action: ')
            if answer == 'y' or answer == 'yes':
                scattererFile = input('Please specify the filename containing the scatterer coordinates: ')
                if '.npy' in scattererFile and Path(os.path.join(datadir, scattererFile)).exists():
                    scatterer = os.path.abspath(scattererFile)
                    noScatterer = False
                    userResponded = True
                    break
                elif '.npy' in scattererFile and not Path(os.path.join(datadir, scattererFile)).exists():
                    print('Error: file \'%s\' does not exist within the current directory.' %(scattererFile))
                    break
                elif '.npy' not in scattererFile:
                    print('''Error: file \'%s\' is not NumPy '.npy' format.''' %(scattererFile))
                    break
            elif answer == '' or answer == 'n' or answer == 'no':
                print('Proceeding without specifying the scatterer coordinates.')
                noScatterer = True
                userResponded = True
                break
            elif answer == 'q' or answer == 'quit':
                sys.exit('Exiting program.')
            else:
                print('Invalid response. Please enter \'y/yes\', \'n/no\', or \'q/quit\'.')
    #==============================================================================    
    if Path(os.path.join(datadir, 'recordingTimes.npy')).exists():
        recordingTimes = os.path.join(datadir, 'recordingTimes.npy')
    else:
        userResponded = False
        print('''
              Error: Expected file \'recordingTimes.npy\' not found. Does a file
              exist containing the recording times? (This is a required file.)
              
              Enter 'y/yes' to specify the filename containing the recording
              times (must be binary NumPy '.npy' format). (Default)
              Enter 'n/no' or 'q/quit to exit this program.
              ''')
        while userResponded == False:
            answer = input('Action: ')
            if answer == '' or answer == 'y' or answer == 'yes':
                timeFile = input('Please specify the filename containing the recording times: ')
                if '.npy' in timeFile and Path(os.path.join(datadir, timeFile)).exists():
                    recordingTimes = os.path.abspath(timeFile)
                    userResponded = True
                    break
                elif '.npy' in timeFile and not Path(os.path.join(datadir, timeFile)).exists():
                    print('Error: file \'%s\' does not exist within the current directory.' %(timeFile))
                    break
                elif '.npy' not in timeFile:
                    print('''Error: file \'%s\' is not NumPy '.npy' format.''' %(timeFile))
                    break  
            elif answer == 'n' or answer == 'no' or answer == 'q' or answer == 'quit':
                sys.exit('Exiting program.')                
            else:
                print('Invalid response. Please enter \'y/yes\', \'n/no\', or \'q/quit\'.')
    #==============================================================================    
    if Path(os.path.join(datadir, 'scatteredData.npy')).exists():
        scatteredData = os.path.join(datadir, 'scatteredData.npy')
    else:
        userResponded = False
        print('''
              Error: Expected file \'scatteredData.npy\' not found. Does a file
              exist containing the measured scattered waves? (This is a required file.)
              
              Enter 'y/yes' to specify the filename containing the measured
              scattered waves (must be binary NumPy '.npy' format). (Default)
              Enter 'n/no' or 'q/quit to exit this program.
              ''')
        while userResponded == False:
            answer = input('Action: ')
            if answer == '' or answer == 'y' or answer == 'yes':
                dataFile = input('Please specify the filename containing the measured scattered waves: ')
                if '.npy' in dataFile and Path(os.path.join(datadir, dataFile)).exists():
                    scatteredData = os.path.abspath(dataFile)
                    userResponded = True
                    break
                elif '.npy' in dataFile and not Path(os.path.join(datadir, dataFile)).exists():
                    print('Error: file \'%s\' does not exist within the current directory.' %(dataFile))
                    break
                elif '.npy' not in dataFile:
                    print('''Error: file \'%s\' is not NumPy '.npy' format.''' %(dataFile))
                    break  
            elif answer == 'n' or answer == 'no' or answer == 'q' or answer == 'quit':
                sys.exit('Exiting program.')                
            else:
                print('Invalid response. Please enter \'y/yes\', \'n/no\', or \'q/quit\'.')
    #==============================================================================
    if Path(os.path.join(datadir, 'testFunctions.npy')).exists():
        testFuncs = os.path.join(datadir, 'testFunctions.npy')
        noTestFuncs = False
    else:
        userResponded = False
        print('''
              Warning: Expected file \'testFunctions.npy\' not found. Does a file
              exist containing the simulated test functions? (This is NOT a required file.)
              
              Enter 'y/yes' to specify the filename containing the simulated test
              functions (must be binary NumPy '.npy' format).
              Enter 'n/no' to proceed without specifying the test functions. (Default)
              Enter 'q/quit' to exit this program.
              ''')
        while userResponded == False:
            answer = input('Action: ')
            if answer == 'y' or answer == 'yes':
                testFuncsFile = input('Please specify the filename containing the simulated test functions: ')
                if '.npy' in testFuncsFile and Path(os.path.join(datadir, testFuncsFile)).exists():
                    testFuncs = os.path.abspath(testFuncsFile)
                    noTestFuncs = False
                    userResponded = True
                    break
                elif '.npy' in testFuncsFile and not Path(os.path.join(datadir, testFuncsFile)).exists():
                    print('Error: file \'%s\' does not exist within the current directory.' %(testFuncsFile))
                    break
                elif '.npy' not in testFuncsFile:
                    print('''Error: file \'%s\' is not NumPy '.npy' format.''' %(testFuncsFile))
                    break
            elif answer == '' or answer == 'n' or answer == 'no':
                print('Proceeding without specifying the simulated test functions.')
                noTestFuncs = True
                userResponded = True
                break
            elif answer == 'q' or answer == 'quit':
                sys.exit('Exiting program.')
            else:
                print('Invalid response. Please enter \'y/yes\', \'n/no\', or \'q/quit\'.')
    #==============================================================================
    # If a user is supplying test functions, the sampling points from which those
    # test functions were generated must also be supplied.
    if noTestFuncs == False:
        if Path(os.path.join(datadir, 'samplingPoints.npy')).exists():
            samplingPoints = os.path.join(datadir, 'samplingPoints.npy')
        else:
            userResponded = False
            print('''
                  Error: Expected file \'samplingPoints.npy\' not found. Does a file
                  exist containing the space-time sampling points? (This file is
                  required only if test functions are provided.)
                  
                  Enter 'y/yes' to specify the filename containing the sampling points
                  (must be binary NumPy '.npy' format). (Default)
                  Enter 'n/no' or 'q/quit' to exit this program.
                  ''')
            while userResponded == False:
                answer = input('Action: ')
                if answer == '' or answer == 'y' or answer == 'yes':
                    samplingPointsFile = input('Please specify the filename containing the sampling points: ')
                    if '.npy' in samplingPointsFile and Path(os.path.join(datadir, samplingPointsFile)).exists():
                        samplingPoints = os.path.abspath(testFuncsFile)
                        userResponded = True
                        break
                    elif '.npy' in samplingPointsFile and not Path(os.path.join(datadir, samplingPointsFile)).exists():
                        print('Error: file \'%s\' does not exist within the current directory.' %(samplingPointsFile))
                        break
                    elif '.npy' not in samplingPointsFile:
                        print('''Error: file \'%s\' is not NumPy '.npy' format.''' %(samplingPointsFile))
                        break
                elif answer == 'n' or answer == 'no' or answer == 'q' or answer == 'quit':
                    sys.exit('Exiting program.')
                else:
                    print('Invalid response. Please enter \'y/yes\', \'n/no\', or \'q/quit\'.')
    #==============================================================================  
    if noScatterer and noTestFuncs:
        np.savez('datadir.npz',
                 receivers = receivers,
                 sources = sources,
                 recordingTimes = recordingTimes,
                 scatteredData = scatteredData)
    elif noScatterer and not noTestFuncs:
        np.savez('datadir.npz',
                 receivers = receivers,
                 sources = sources,
                 recordingTimes = recordingTimes,
                 scatteredData = scatteredData,
                 testFuncs = testFuncs,
                 samplingPoints = samplingPoints)
    elif not noScatterer and noTestFuncs:
        np.savez('datadir.npz',
                 receivers = receivers,
                 sources = sources,
                 scatterer = scatterer,
                 recordingTimes = recordingTimes,
                 scatteredData = scatteredData)
    else:
        np.savez('datadir.npz',
                 receivers = receivers,
                 sources = sources,
                 scatterer = scatterer,
                 recordingTimes = recordingTimes,
                 scatteredData = scatteredData,
                 testFuncs = testFuncs,
                 samplingPoints=samplingPoints)