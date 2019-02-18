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
import pickle
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from vezda.plot_utils import (FontColor, default_params,
                              setFigure, plotImage, plotMap)

def info():
    commandName = FontColor.BOLD + 'vzimage:' + FontColor.END
    description = ' plot the experimental setup and obtained images'
    
    return commandName + description

def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nfe', action='store_true',
                        help='''Plot the image obtained by solving the near-field equation.''')
    parser.add_argument('--lse', action='store_true',
                        help='''Plot the image obtained by solving the Lippmann-Schwinger equation.''')
    parser.add_argument('--movie', action='store_true',
                        help='Save a three-dimensional figure as a rotating frame.')
    parser.add_argument('--isolevel', type=float, default=None,
                        help='''Specify the contour level of the isosurface for
                        three-dimensional visualizations. Level must be between 0 and 1.''')
    parser.add_argument('--format', '-f', type=str, default=None, choices=['png', 'pdf', 'ps', 'eps', 'svg'],
                        help='''specify the image format of the saved file. Accepted formats are png, pdf,
                        ps, eps, and svg. Default format is set to pdf.''')
    parser.add_argument('--xlabel', type=str, default=None,
                        help='''specify a label for the x-axis.''')
    parser.add_argument('--ylabel', type=str, default=None,
                        help='''specify a label for the y-axis.''')
    parser.add_argument('--zlabel', type=str, default=None,
                        help='''specify a label for the z-axis.''')
    parser.add_argument('--xu', type=str, default=None,
                        help='Specify the units for the x-axis (e.g., \'m\' or \'km\').')
    parser.add_argument('--yu', type=str, default=None,
                        help='Specify the units for the y-axis (e.g., \'m\' or \'km\').')
    parser.add_argument('--zu', type=str, default=None,
                        help='Specify the units for the z-axis (e.g., \'m\' or \'km\').')
    parser.add_argument('--colormap', type=str, default=None, choices=['viridis', 'plasma', 'inferno', 'magma', 'cividis'],
                        help='specify a perceptually uniform sequential colormap. Default is \'magma\'')
    parser.add_argument('--colorbar', type=str, default=None, choices=['n', 'no', 'false', 'y', 'yes', 'true'],
                        help='specify \'y/yes/true\' to plot colorbar. Default is \'n/no/false\'')
    parser.add_argument('--invert_xaxis', type=str, default=None, choices=['n', 'no', 'false', 'y', 'yes', 'true'],
                        help='specify \'y/yes/true\' to invert x-axis. Default is \'n/no/false\'')
    parser.add_argument('--invert_yaxis', type=str, default=None, choices=['n', 'no', 'false', 'y', 'yes', 'true'],
                        help='specify \'y/yes/true\' to invert y-axis. Default is \'n/no/false\'')
    parser.add_argument('--invert_zaxis', type=str, default=None, choices=['n', 'no', 'false', 'y', 'yes', 'true'],
                        help='specify \'y/yes/true\' to invert z-axis. Default is \'n/no/false\'')
    parser.add_argument('--show_scatterer', type=str, default=None, choices=['n', 'no', 'false', 'y', 'yes', 'true'],
                        help='specify \'y/yes/true\' to show scatterer boundary. Default is \'n/no/false\'')
    parser.add_argument('--show_sources', type=str, default=None, choices=['n', 'no', 'false', 'y', 'yes', 'true'],
                        help='specify \'n/no/false\' to hide sources. Default is \'y/yes/true\'')
    parser.add_argument('--show_receivers', type=str, default=None, choices=['n', 'no', 'false', 'y', 'yes', 'true'],
                        help='specify \'n/no/false\' to hide receivers. Default is \'y/yes/true\'')
    parser.add_argument('--mode', type=str, choices=['light', 'dark'], required=False,
                        help='''Specify whether to view plots in light mode for daytime viewing
                        or dark mode for nighttime viewing.
                        Mode must be either \'light\' or \'dark\'.''')
    args = parser.parse_args()
    
    #==============================================================================
    
    # if a plotParams.pkl file already exists, load relevant parameters
    if Path('plotParams.pkl').exists():
        plotParams = pickle.load(open('plotParams.pkl', 'rb'))
        
        # for both wiggle plots and image plots
        if args.format is not None:
            plotParams['pltformat'] = args.format
        
        if args.mode is not None:
            plotParams['view_mode'] = args.mode
        
        # for image/map plots
        if args.isolevel is not None:
            plotParams['isolevel'] = args.isolevel
        
        if args.xlabel is not None:
            plotParams['xlabel'] = args.xlabel
        
        if args.ylabel is not None:
            plotParams['ylabel'] = args.ylabel
            
        if args.zlabel is not None:
            plotParams['zlabel'] = args.zlabel          
        #==============================================================================
        # handle units here
        if args.xu is not None:
            plotParams['xu'] = args.xu
            
        if args.yu is not None:
            plotParams['yu'] = args.yu
            
        if args.zu is not None:
            plotParams['zu'] = args.zu
        #==============================================================================    
        if args.colormap is not None:
            plotParams['colormap'] = args.colormap
        #==============================================================================
        if args.colorbar is not None:
            if args.colorbar == 'n' or args.colorbar == 'no' or args.colorbar == 'false':
                plotParams['colorbar'] = False
        
            elif args.colorbar == 'y' or args.colorbar == 'yes' or args.colorbar == 'true':
                plotParams['colorbar'] = True
        #==============================================================================    
        if args.invert_xaxis is not None:
            if args.invert_xaxis == 'n' or args.invert_xaxis == 'no' or args.invert_xaxis == 'false':
                plotParams['invert_xaxis'] = False
        
            elif args.invert_xaxis == 'y' or args.invert_xaxis == 'yes' or args.invert_xaxis == 'true':
                plotParams['invert_xaxis'] = True
        #==============================================================================
        if args.invert_yaxis is not None:
            if args.invert_yaxis == 'n' or args.invert_yaxis == 'no' or args.invert_yaxis == 'false':
                plotParams['invert_yaxis'] = False
        
            elif args.invert_yaxis == 'y' or args.invert_yaxis == 'yes' or args.invert_yaxis == 'true':
                plotParams['invert_yaxis'] = True
        #==============================================================================
        if args.invert_zaxis is not None:
            if args.invert_zaxis == 'n' or args.invert_zaxis == 'no' or args.invert_zaxis == 'false':
                plotParams['invert_zaxis'] = False
        
            elif args.invert_zaxis == 'y' or args.invert_zaxis == 'yes' or args.invert_zaxis == 'true':
                plotParams['invert_zaxis'] = True
        #==============================================================================
        if args.show_scatterer is not None:
            if args.show_scatterer == 'n' or args.show_scatterer == 'no' or args.show_scatterer == 'false':
                plotParams['show_scatterer'] = False
        
            elif args.show_scatterer == 'y' or args.show_scatterer == 'yes' or args.show_scatterer == 'true':
                plotParams['show_scatterer'] = True
        #==============================================================================
        if args.show_sources is not None:
            if args.show_sources == 'n' or args.show_sources == 'no' or args.show_sources == 'false':
                plotParams['show_sources'] = False
            
            elif args.show_sources == 'y' or args.show_sources == 'yes' or args.show_sources == 'true':
                plotParams['show_sources'] = True
        #==============================================================================
        if args.show_receivers is not None:
            if args.show_receivers == 'n' or args.show_receivers == 'no' or args.show_receivers == 'false':
                plotParams['show_receivers'] = False
        
            elif args.show_receivers == 'y' or args.show_receivers == 'yes' or args.show_receivers == 'true':
                plotParams['show_receivers'] = True

    #==============================================================================
    else: # create a plotParams dictionary file with default values
        plotParams = default_params()
        
        # updated parameters based on passed arguments
        #for both image and wiggle plots
        if args.format is not None:
            plotParams['pltformat'] = args.format
            
        # for image/map plots
        if args.isolevel is not None:
            plotParams['isolevel'] = args.isolevel
        
        if args.colormap is not None:
            plotParams['colormap'] = args.colormap
            
        if args.colorbar is not None:
            if args.colorbar == 'n' or args.colorbar == 'no' or args.colorbar == 'false':
                plotParams['colorbar'] = False
        
            elif args.colorbar == 'y' or args.colorbar == 'yes' or args.colorbar == 'true':
                plotParams['colorbar'] = True
        
        if args.xlabel is not None:
            plotParams['xlabel'] = args.xlabel
        
        if args.ylabel is not None:
            plotParams['ylabel'] = args.ylabel
        
        if args.zlabel is not None:
            plotParams['zlabel'] = args.zlabel          
        
        #==============================================================================
        # update units
        if args.xu is not None:
            plotParams['xu'] = args.xu
            
        if args.yu is not None:
            plotParams['yu'] = args.yu
            
        if args.zu is not None:
            plotParams['zu'] = args.zu
        #==============================================================================
        if args.invert_xaxis is not None:
            if args.invert_xaxis == 'n' or args.invert_xaxis == 'no' or args.invert_xaxis == 'false':
                plotParams['invert_xaxis'] = False
            
            elif args.invert_xaxis == 'y' or args.invert_xaxis == 'yes' or args.invert_xaxis == 'true':
                plotParams['invert_xaxis'] = True
        #==============================================================================    
        if args.invert_yaxis is not None:
            if args.invert_yaxis == 'n' or args.invert_yaxis == 'no' or args.invert_yaxis == 'false':
                plotParams['invert_yaxis'] = False
            
            elif args.invert_yaxis == 'y' or args.invert_yaxis == 'yes' or args.invert_yaxis == 'true':
                plotParams['invert_yaxis'] = True
        #==============================================================================    
        if args.invert_zaxis is not None:
            if args.invert_zaxis == 'n' or args.invert_zaxis == 'no' or args.invert_zaxis == 'false':
                plotParams['invert_zaxis'] = False
            
            elif args.invert_zaxis == 'y' or args.invert_zaxis == 'yes' or args.invert_zaxis == 'true':
                plotParams['invert_zaxis'] = True
        #==============================================================================
        if args.show_scatterer is not None:
            if args.show_scatterer == 'n' or args.show_scatterer == 'no' or args.show_scatterer == 'false':
                plotParams['show_scatterer'] = False
            
            elif args.show_scatterer == 'y' or args.show_scatterer == 'yes' or args.show_scatterer == 'true':
                plotParams['show_scatterer'] = True
        #==============================================================================
        if args.show_sources is not None:
            if args.show_sources == 'n' or args.show_sources == 'no' or args.show_sources == 'false':
                plotParams['show_sources'] = False
            
            elif args.show_sources == 'y' or args.show_sources == 'yes' or args.show_sources == 'true':
                plotParams['show_sources'] = True
        #==============================================================================
        if args.show_receivers is not None:
            if args.show_receivers == 'n' or args.show_receivers == 'no' or args.show_receivers == 'false':
                plotParams['show_receivers'] = False
            
            elif args.show_receivers == 'y' or args.show_receivers == 'yes' or args.show_receivers == 'true':
                plotParams['show_receivers'] = True
        
    pickle.dump(plotParams, open('plotParams.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)        

    #==============================================================================
    # load the shape of the scatterer and source/receiver locations
    datadir = np.load('datadir.npz')
    receiverPoints = np.load(str(datadir['receivers']))
    
    if 'sources' in datadir:
        sourcePoints = np.load(str(datadir['sources']))
    else:
        sourcePoints = None
    
    if 'scatterer' in datadir and plotParams['show_scatterer']:
        scatterer = np.load(str(datadir['scatterer']))  
    else:
        scatterer = None
        if plotParams['show_scatterer']:
            print(textwrap.dedent(
                  '''
                  Warning: Attempted to load the file containing the scatterer coordinates,
                  but no such file exists. If a file exists containing the scatterer
                  points, run:
                      
                      'vzdata --path=<path/to/data/>'
                      
                  and specify the file containing the scatterer points when prompted. Otherwise,
                  specify 'no' when asked if a file containing the scatterer points exists.
                  '''))
    
    if Path('window.npz').exists():
        windowDict = np.load('window.npz')
        
        # Apply the receiver window
        rstart = windowDict['rstart']
        rstop = windowDict['rstop']
        rstep = windowDict['rstep']
        
        rinterval = np.arange(rstart, rstop, rstep)
        receiverPoints = receiverPoints[rinterval, :]
        
        if sourcePoints is not None:
            # Apply the source window
            sstart = windowDict['sstart']
            sstop = windowDict['sstop']
            sstep = windowDict['sstep']
        
            sinterval = np.arange(sstart, sstop, sstep)
            sourcePoints = sourcePoints[sinterval, :]
        
    #==============================================================================
    # Load the user-specified sampling grid
    if 'samplingGrid' in datadir:
        samplingGrid = np.load(str(datadir['samplingGrid']))
    else:
        try:
            samplingGrid = np.load('samplingGrid.npz')
        except FileNotFoundError:
            samplingGrid = None
        
    if samplingGrid is None:
        sys.exit(textwrap.dedent(
                '''
                A sampling grid needs to be set up before it can be plotted.
                Enter:
                    
                    vzgrid --help
                
                from the command-line for more information on how to set up a
                sampling grid.
                '''))
    
    x = samplingGrid['x']
    y = samplingGrid['y']
    tau = samplingGrid['tau']
    if 'z' not in samplingGrid:
        X, Y = np.meshgrid(x, y, indexing='ij')
        Z = None
    else:
        z = samplingGrid['z']
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
    #==============================================================================
    if Path('imageNFE.npz').exists() and not Path('imageLSE.npz').exists():
        if args.lse:
            sys.exit(textwrap.dedent(
                    '''
                    PlotError: User requested to plot an image obtained by solving
                    the Lippmann-Schwinger equation (LSE), but no such image exists.
                    '''))
            
        # plot the image obtained by solving the near-field equaiton (NFE)
        Dict = np.load('imageNFE.npz')
        flag = 'NFE'
        fig, ax = plotImage(Dict, X, Y, Z, tau, plotParams, flag, args.movie)
        
    elif not Path('imageNFE.npz').exists() and Path('imageLSE.npz').exists():
        if args.nfe:
            sys.exit(textwrap.dedent(
                    '''
                    PlotError: User requested to plot an image obtained by solving
                    the near-field equation (NFE), but no such image exists.
                    '''))
            
        # plot the image obtained by solving the Lippmann-Schwinger equation (LSE)
        Dict = np.load('imageLSE.npz')
        flag = 'LSE'
        fig, ax = plotImage(Dict, X, Y, Z, tau, plotParams, flag)
        
    
    elif Path('imageNFE.npz').exists() and Path('imageLSE.npz').exists():
        if args.nfe and not args.lse:
            # plot the image obtained by solving the near-field equaiton (NFE)
            Dict = np.load('imageNFE.npz')
            flag = 'NFE'
            fig, ax = plotImage(Dict, X, Y, Z, tau, plotParams, flag, args.movie)        
                
        elif not args.nfe and args.lse:
            # plot the image obtained by solving the Lippmann-Schwinger equation (LSE)
            Dict = np.load('imageLSE.npz')
            flag = 'LSE'
            fig, ax = plotImage(Dict, X, Y, Z, tau, plotParams, flag)
            
        
        elif args.nfe and args.lse:
            sys.exit(textwrap.dedent(
                    '''
                    PlotError: Please specify only one of the arguments \'--nfe\' or \'--lse\' to
                    view the corresponding image.'''))
        
        
        else:
            flag = ''
            print(textwrap.dedent(
                  '''
                  Images obtained by solving both NFE and LSE are available. Enter:
                      
                      vzimage --nfe
                        
                  to view the image obtained by solving NFE or
                    
                      vzimage --lse
                        
                  to view the image obtained by solving LSE.
                  '''))
            
    else:
        flag = ''
        if args.nfe:
            print('Warning: An image obtained by solving the near-field equation (NFE) does not exist.')
        
        elif args.lse:
            print('Warning: An image obtained by solving the Lippmann-Schwinger equation (LSE) does not exist.')
        
        elif args.nfe and args.lse:
            print(textwrap.dedent(
                    '''
                    Warning: An image has not yet been obtained by solving either the
                    near-field equation (NFE) or the Lippmann-Schwinger equation (LSE).
                    '''))
        
    try:
        ax
    except NameError:
        fig, ax = setFigure(num_axes=1, mode=plotParams['view_mode'], ax1_dim=receiverPoints.shape[1])
    
    plotMap(ax, None, receiverPoints, sourcePoints, scatterer, 'data', plotParams)
        
    #==============================================================================
    
    pltformat = plotParams['pltformat']    
    fig.savefig('image' + flag + '.' + pltformat, format=pltformat, bbox_inches='tight',
                 facecolor=fig.get_facecolor(), transparent=True)    
    plt.show()
