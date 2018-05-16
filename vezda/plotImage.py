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
import pickle
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage import measure
from pathlib import Path
from tqdm import trange

def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spacetime', action='store_true',
                        help='''Plot the support of the source function in space-time.''')
    parser.add_argument('--movie', action='store_true',
                        help='''Save the space-time figure as a rotating frame (2D space)
                        or as a time-lapse structure (3D space).''')
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
    args = parser.parse_args()
    
    #==============================================================================
    
    # if a plotParams.pkl file already exists, load relevant parameters
    if Path('plotParams.pkl').exists():
        plotParams = pickle.load(open('plotParams.pkl', 'rb'))
        
        # for both wiggle plots and image plots
        if args.format is not None:
            pltformat = args.format
            plotParams['pltformat'] = pltformat
        else:
            pltformat = plotParams['pltformat']
        
        # for image/map plots
        if args.isolevel is not None:
            isolevel = args.isolevel
            plotParams['isolevel'] = isolevel
        else:
            isolevel = plotParams['isolevel']
        
        if args.xlabel is not None:
            xlabel = args.xlabel
            plotParams['xlabel'] = xlabel
        else:
            xlabel = plotParams['xlabel']
        
        if args.ylabel is not None:
            ylabel = args.ylabel
            plotParams['ylabel'] = ylabel
        else:
            ylabel = plotParams['ylabel']
            
        if args.zlabel is not None:
            zlabel = args.zlabel
            plotParams['zlabel'] = zlabel
        else:
            xlabel = plotParams['xlabel']            
        #==============================================================================
        # handle units here
        if all(v is not None for v in [args.xu, args.yu, args.zu]):
            xu = args.xu
            yu = args.yu
            zu = args.zu
            plotParams['xu'] = xu
            plotParams['yu'] = yu
            plotParams['zu'] = zu
        elif args.xu is not None and all(v is None for v in [args.yu, args.zu]):
            xu = args.xu
            yu = xu
            zu = xu
            plotParams['xu'] = xu
            plotParams['yu'] = yu
            plotParams['zu'] = zu
        elif args.yu is not None and all(v is None for v in [args.xu, args.zu]):
            yu = args.yu
            xu = yu
            zu = yu
            plotParams['xu'] = xu
            plotParams['yu'] = yu
            plotParams['zu'] = zu
        elif args.zu is not None and all(v is None for v in [args.xu, args.yu]):
            zu = args.zu
            xu = zu
            yu = zu
            plotParams['xu'] = xu
            plotParams['yu'] = yu
            plotParams['zu'] = zu
        elif args.zu is None and all(v is not None for v in [args.xu, args.yu]):
            xu = args.xu
            yu = args.yu
            zu = xu
            plotParams['xu'] = xu
            plotParams['yu'] = yu
            plotParams['zu'] = zu
        elif args.yu is None and all(v is not None for v in [args.xu, args.zu]):
            xu = args.xu
            zu = args.zu
            yu = xu
            plotParams['xu'] = xu
            plotParams['yu'] = yu
            plotParams['zu'] = zu
        elif args.xu is None and all(v is not None for v in [args.yu, args.zu]):
            yu = args.yu
            zu = args.zu
            xu = yu
            plotParams['xu'] = xu
            plotParams['yu'] = yu
            plotParams['zu'] = zu
        elif all(v is None for v in [args.xu, args.yu, args.zu]):
            xu = plotParams['xu']
            yu = plotParams['yu']
            zu = plotParams['zu']
        #==============================================================================    
        if args.colorbar is not None:
            if args.colorbar == 'n' or args.colorbar == 'no' or args.colorbar == 'false':
                wantColorbar = False
                if plotParams['colorbar'] is True:
                    plotParams['colorbar'] = False
            elif args.colorbar == 'y' or args.colorbar == 'yes' or args.colorbar == 'true':
                wantColorbar = True
                if plotParams['colorbar'] is False:
                    plotParams['colorbar'] = True
        else:
            wantColorbar = plotParams['colorbar']
        #==============================================================================    
        if args.invert_xaxis is not None:
            if args.invert_xaxis == 'n' or args.invert_xaxis == 'no' or args.invert_xaxis == 'false':
                invertX = False
                if plotParams['invert_xaxis'] is True:
                    plotParams['invert_xaxis'] = False
            elif args.invert_xaxis == 'y' or args.invert_xaxis == 'yes' or args.invert_xaxis == 'true':
                invertX = True
                if plotParams['invert_xaxis'] is False:
                    plotParams['invert_xaxis'] = True
        else:
            invertX = plotParams['invert_xaxis']
        #==============================================================================
        if args.invert_yaxis is not None:
            if args.invert_yaxis == 'n' or args.invert_yaxis == 'no' or args.invert_yaxis == 'false':
                invertY = False
                if plotParams['invert_yaxis'] is True:
                    plotParams['invert_yaxis'] = False
            elif args.invert_yaxis == 'y' or args.invert_yaxis == 'yes' or args.invert_yaxis == 'true':
                invertY = True
                if plotParams['invert_yaxis'] is False:
                    plotParams['invert_yaxis'] = True
        else:
            invertY = plotParams['invert_yaxis']
        #==============================================================================
        if args.invert_zaxis is not None:
            if args.invert_zaxis == 'n' or args.invert_zaxis == 'no' or args.invert_zaxis == 'false':
                invertZ = False
                if plotParams['invert_zaxis'] is True:
                    plotParams['invert_zaxis'] = False
            elif args.invert_zaxis == 'y' or args.invert_zaxis == 'yes' or args.invert_zaxis == 'true':
                invertZ = True
                if plotParams['invert_zaxis'] is False:
                    plotParams['invert_zaxis'] = True
        else:
            invertZ = plotParams['invert_zaxis']
        #==============================================================================
        if args.show_scatterer is not None:
            if args.show_scatterer == 'n' or args.show_scatterer == 'no' or args.show_scatterer == 'false':
                showScatterer = False
                if plotParams['show_scatterer'] is True:
                    plotParams['show_scatterer'] = False
            elif args.show_scatterer == 'y' or args.show_scatterer == 'yes' or args.show_scatterer == 'true':
                showScatterer = True
                if plotParams['show_scatterer'] is False:
                    plotParams['show_scatterer'] = True
        else:
            showScatterer = plotParams['show_scatterer']
        #==============================================================================
        if args.show_sources is not None:
            if args.show_sources == 'n' or args.show_sources == 'no' or args.show_sources == 'false':
                showSources = False
                if plotParams['show_sources'] is True:
                    plotParams['show_sources'] = False
            elif args.show_sources == 'y' or args.show_sources == 'yes' or args.show_sources == 'true':
                showSources = True
                if plotParams['show_sources'] is False:
                    plotParams['show_sources'] = True
        else:
            showSources = plotParams['show_sources']
        #==============================================================================
        if args.show_receivers is not None:
            if args.show_receivers == 'n' or args.show_receivers == 'no' or args.show_receivers == 'false':
                showReceivers = False
                if plotParams['show_receivers'] is True:
                    plotParams['show_receivers'] = False
            elif args.show_receivers == 'y' or args.show_receivers == 'yes' or args.show_receivers == 'true':
                showReceivers = True
                if plotParams['show_receivers'] is False:
                    plotParams['show_receivers'] = True
        else:
            showReceivers = plotParams['show_receivers']
        #==============================================================================
        tu = plotParams['tu']
        
        pickle.dump(plotParams, open('plotParams.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)
    #==============================================================================
    else: # create a plotParams dictionary file with default values
        plotParams = {}
        #for both image and wiggle plots
        if args.format is None:
            pltformat = 'pdf'
        else:
            pltformat = args.format
        plotParams['pltformat'] = pltformat
        
        # for image/map plots
        if args.isolevel is None:
            isolevel = 0.7
        else:
            isolevel = args.isolevel
        plotParams['isolevel'] = isolevel
        
        if args.colorbar is not None:
            if args.colorbar == 'n' or args.colorbar == 'no' or args.colorbar == 'false':
                wantColorbar = False
                plotParams['colorbar'] = wantColorbar
            elif args.colorbar == 'y' or args.colorbar == 'yes' or args.colorbar == 'true':
                wantColorbar = True
                plotParams['colorbar'] = wantColorbar
        else:
            wantColorbar = False
            plotParams['colorbar'] = wantColorbar
        
        if args.xlabel is None:
            xlabel = ''
        else:
            xlabel = args.xlabel
        plotParams['xlabel'] = xlabel
        
        if args.ylabel is None:
            ylabel = ''
        else:
            ylabel = args.ylabel
        plotParams['ylabel'] = ylabel
        
        if args.zlabel is None:
            zlabel = ''
        else:
            zlabel = args.zlabel
        plotParams['zlabel'] = zlabel
        #==============================================================================
        if all(v is not None for v in [args.xu, args.yu, args.zu]):
            xu = args.xu
            yu = args.yu
            zu = args.zu
            plotParams['xu'] = xu
            plotParams['yu'] = yu
            plotParams['zu'] = zu
        elif args.xu is not None and all(v is None for v in [args.yu, args.zu]):
            xu = args.xu
            yu = xu
            zu = xu
            plotParams['xu'] = xu
            plotParams['yu'] = yu
            plotParams['zu'] = zu
        elif args.yu is not None and all(v is None for v in [args.xu, args.zu]):
            yu = args.yu
            xu = yu
            zu = yu
            plotParams['xu'] = xu
            plotParams['yu'] = yu
            plotParams['zu'] = zu
        elif args.zu is not None and all(v is None for v in [args.xu, args.yu]):
            zu = args.zu
            xu = zu
            yu = zu
            plotParams['xu'] = xu
            plotParams['yu'] = yu
            plotParams['zu'] = zu
        elif args.zu is None and all(v is not None for v in [args.xu, args.yu]):
            xu = args.xu
            yu = args.yu
            zu = xu
            plotParams['xu'] = xu
            plotParams['yu'] = yu
            plotParams['zu'] = zu
        elif args.yu is None and all(v is not None for v in [args.xu, args.zu]):
            xu = args.xu
            zu = args.zu
            yu = xu
            plotParams['xu'] = xu
            plotParams['yu'] = yu
            plotParams['zu'] = zu
        elif args.xu is None and all(v is not None for v in [args.yu, args.zu]):
            yu = args.yu
            zu = args.zu
            xu = yu
            plotParams['xu'] = xu
            plotParams['yu'] = yu
            plotParams['zu'] = zu
        elif all(v is None for v in [args.xu, args.yu, args.zu]):
            xu = ''
            yu = ''
            zu = ''
            plotParams['xu'] = xu
            plotParams['yu'] = yu
            plotParams['zu'] = zu
        #==============================================================================
        if args.invert_xaxis is not None:
            if args.invert_xaxis == 'n' or args.invert_xaxis == 'no' or args.invert_xaxis == 'false':
                invertX = False
                plotParams['invert_xaxis'] = invertX
            elif args.invert_xaxis == 'y' or args.invert_xaxis == 'yes' or args.invert_xaxis == 'true':
                invertX = True
                plotParams['invert_xaxis'] = invertX
        else:
            invertX = False
            plotParams['invert_xaxis'] = invertX
        #==============================================================================    
        if args.invert_yaxis is not None:
            if args.invert_yaxis == 'n' or args.invert_yaxis == 'no' or args.invert_yaxis == 'false':
                invertY = False
                plotParams['invert_yaxis'] = invertY
            elif args.invert_yaxis == 'y' or args.invert_yaxis == 'yes' or args.invert_yaxis == 'true':
                invertY = True
                plotParams['invert_yaxis'] = invertY
        else:
            invertY = False
            plotParams['invert_yaxis'] = invertY
        #==============================================================================    
        if args.invert_zaxis is not None:
            if args.invert_zaxis == 'n' or args.invert_zaxis == 'no' or args.invert_zaxis == 'false':
                invertZ = False
                plotParams['invert_zaxis'] = invertZ
            elif args.invert_zaxis == 'y' or args.invert_zaxis == 'yes' or args.invert_zaxis == 'true':
                invertZ = True
                plotParams['invert_zaxis'] = invertZ
        else:
            invertZ = False
            plotParams['invert_zaxis'] = invertZ
        #==============================================================================
        if args.show_scatterer is not None:
            if args.show_scatterer == 'n' or args.show_scatterer == 'no' or args.show_scatterer == 'false':
                showScatterer = False
                plotParams['show_scatterer'] = showScatterer
            elif args.show_scatterer == 'y' or args.show_scatterer == 'yes' or args.show_scatterer == 'true':
                showScatterer = True
                plotParams['show_scatterer'] = showScatterer
        else:
            showScatterer = False
            plotParams['show_scatterer'] = showScatterer
        #==============================================================================
        if args.show_sources is not None:
            if args.show_sources == 'n' or args.show_sources == 'no' or args.show_sources == 'false':
                showSources = False
                plotParams['show_sources'] = showSources
            elif args.show_sources == 'y' or args.show_sources == 'yes' or args.show_sources == 'true':
                showSources = True
                plotParams['show_sources'] = showSources
        else:
            showSources = True
            plotParams['show_sources'] = showSources
        #==============================================================================
        if args.show_receivers is not None:
            if args.show_receivers == 'n' or args.show_receivers == 'no' or args.show_receivers == 'false':
                showReceivers = False
                plotParams['show_receivers'] = showReceivers
            elif args.show_receivers == 'y' or args.show_receivers == 'yes' or args.show_receivers == 'true':
                showReceivers = True
                plotParams['show_receivers'] = showReceivers
        else:
            showReceivers = True
            plotParams['show_receivers'] = showReceivers
        #==============================================================================
        tu = ''
        plotParams['tu'] = tu
        au = ''
        plotParams['au'] = au
        
        pickle.dump(plotParams, open('plotParams.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)

    #==============================================================================
    def remove_keymap_conflicts(new_keys_set):
        for prop in plt.rcParams:
            if prop.startswith('keymap.'):
                keys = plt.rcParams[prop]
                remove_list = set(keys) & new_keys_set
                for key in remove_list:
                    keys.remove(key)
    
    def process_key(event, Ntau, tau):
        fig = event.canvas.figure
        ax = fig.axes[0]
        if event.key == 'left' or event.key == 'down':
            previous_slice(ax, Ntau, tau)
        elif event.key == 'right' or event.key == 'up':
            next_slice(ax, Ntau, tau)
        fig.canvas.draw()

    def previous_slice(ax, Ntau, tau):
        volume = ax.volume
        ax.index = (ax.index - 1) % Ntau  # wrap around using %
        space_viewer(ax, volume, tau[ax.index])
    
    def next_slice(ax, Ntau, tau):
        volume = ax.volume
        ax.index = (ax.index + 1) % Ntau  # wrap around using %
        space_viewer(ax, volume, tau[ax.index])
        
    def space_viewer(ax, volume, tau):
        ax.clear()
        
        if ndspace == 2:
            im = ax.contourf(X, Y, volume[:, :, ax.index], 100, cmap=plt.cm.magma)
            if wantColorbar:
                divider = make_axes_locatable(ax)
                cax = divider.append_axes('right', size='5%', pad=0.05)
                cbar = plt.colorbar(im, cax=cax)                
                cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                cbar.set_label(r'$\frac{1}{\vert\vert\varphi\vert\vert}$',
                               labelpad=24, rotation=0, fontsize=18)
            
            if xu != '':
                ax.set_xlabel(xlabel + ' (%s)' %(xu))
            else:
                ax.set_xlabel(xlabel)
            
            if yu != '':
                ax.set_ylabel(ylabel + ' (%s)' %(yu))
            else:
                ax.set_ylabel(ylabel)
                
            if tu != '':
                ax.set_title(r'$\tau$ = %0.2f %s' %(tau, tu))
            else:
                ax.set_title(r'$\tau$ = %0.2f' %(tau))
                
        elif ndspace == 3:
            x = X[:, 0, 0]
            y = Y[0, :, 0]
            z = Z[0, 0, :]
            
            verts, faces, normals, values = measure.marching_cubes_lewiner(volume[:, :, :, ax.index], level=isolevel)
            # Rescale coordinates of vertices to lie within x,y,z ranges
            verts[:, 0] = verts[:, 0] * (x[-1] - x[0]) / (np.max(verts[:, 0]) - np.min(verts[:, 0])) + x[0]
            verts[:, 1] = verts[:, 1] * (y[-1] - y[0]) / (np.max(verts[:, 1]) - np.min(verts[:, 1])) + y[0]
            verts[:, 2] = verts[:, 2] * (z[-1] - z[0]) / (np.max(verts[:, 2]) - np.min(verts[:, 2])) + z[0]
            
            # Plot isosurface of support of source function in space
            ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], color='silver')
           
            if xu != '':
                ax.set_xlabel(xlabel + ' (%s)' %(xu))
            else:
                ax.set_xlabel(xlabel)

            if yu != '':
                ax.set_ylabel(ylabel + ' (%s)' %(yu))
            else:
                ax.set_ylabel(ylabel)
                
            if zu != '':
                ax.set_zlabel(zlabel + ' (%s)' %(zu))
            else:
                ax.set_zlabel(zlabel)
                
            if tu != '':
                ax.set_title('''Isosurface @ %s [$\\tau$ = %0.2f %s]''' %(isolevel, tau, tu))
            else:
                ax.set_title('''Isosurface @ %s [$\\tau$ = %0.2f]''' %(isolevel, tau))
        
        ax.set_aspect('equal')
        if invertX:
            ax.invert_xaxis()
        if invertY:
            ax.invert_yaxis()
        if invertZ:
            ax.invert_zaxis()
        
        return ax
    #==============================================================================
    # load the shape of the scatterer and source/receiver locations
    datadir = np.load('datadir.npz')
    receiverPoints = np.load(str(datadir['receivers']))
    sourcePoints = np.load(str(datadir['sources']))
    if 'scatterer' in datadir:
        scatterer = np.load(str(datadir['scatterer']))
    else:
        scatterer = None
        
    if scatterer is None and showScatterer == True:
        sys.exit('''
              Attempted to load the file containing the scatterer coordinates,
              but no such file exists. If a file exists containing the scatterer
              points, run 'vzdata --path=<path/to/data/>' command and specify
              the file containing the scatterer points when prompted. Otherwise,
              specify 'no' when asked if a file containing the scatterer points
              exists.
              ''')
    
    if Path('window.npz').exists():
        windowDict = np.load('window.npz')
        
        # Apply the receiver window
        rstart = windowDict['rstart']
        rstop = windowDict['rstop']
        rstep = windowDict['rstep']
        
        rinterval = np.arange(rstart, rstop, rstep)
        receiverPoints = receiverPoints[rinterval, :]
        
        # Apply the source window
        sstart = windowDict['sstart']
        sstop = windowDict['sstop']
        sstep = windowDict['sstep']
        
        sinterval = np.arange(sstart, sstop, sstep)
        sourcePoints = sourcePoints[sinterval, :]
        
    #==============================================================================
    if args.spacetime is False and Path('image.npz').exists():
        Dict = np.load('image.npz')
        Image = Dict['Image']
        alpha = Dict['alpha']
        tau = Dict['tau']
        ndspace = int(Dict['ndspace'])
        Ntau = len(tau)
        
        if ndspace == 2:
            X = Dict['X']
            Y = Dict['Y']
            
            fig1 = plt.figure()
            ax1 = fig1.add_subplot(111)
            im = ax1.contourf(X, Y, Image, 100, cmap=plt.cm.magma)
            
            if wantColorbar:
                divider = make_axes_locatable(ax1)
                cax = divider.append_axes('right', size='5%', pad=0.05)
                cbar = plt.colorbar(im, cax=cax)
                cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                if Ntau == 1:
                    cbar.set_label(r'$\frac{1}{\vert\vert\varphi\vert\vert}$',
                                   labelpad=24, rotation=0, fontsize=18)
                    
        elif ndspace == 3:
            X = Dict['X']
            Y = Dict['Y']
            Z = Dict['Z']
            x = X[:, 0, 0]
            y = Y[0, :, 0]
            z = Z[0, 0, :]
            
            verts, faces, normals, values = measure.marching_cubes_lewiner(Image, level=isolevel)
            # Rescale coordinates of vertices to lie within x,y,z ranges
            verts[:, 0] = verts[:, 0] * (x[-1] - x[0]) / (np.max(verts[:, 0]) - np.min(verts[:, 0])) + x[0]
            verts[:, 1] = verts[:, 1] * (y[-1] - y[0]) / (np.max(verts[:, 1]) - np.min(verts[:, 1])) + y[0]
            verts[:, 2] = verts[:, 2] * (z[-1] - z[0]) / (np.max(verts[:, 2]) - np.min(verts[:, 2])) + z[0]
            
            fig1 = plt.figure()
            ax1 = fig1.add_subplot(111, projection='3d')
            ax1.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], color='silver')
        
        if Ntau == 1:
            if alpha != 0 and tu != '':
                ax1.set_title(r'$\alpha = %0.1e, \tau = %0.2f$ %s' %(alpha, tau[0], tu))
            elif alpha != 0 and tu == '':
                ax1.set_title(r'$\alpha = %0.1e, \tau = %0.2f$' %(alpha, tau[0]))
            elif alpha == 0 and tu != '':
                ax1.set_title(r'$\alpha = %s, \tau = %0.2f$ %s' %(alpha, tau[0], tu))
            elif alpha == 0 and tu == '':
                ax1.set_title(r'$\alpha = %s, \tau = %0.2f$' %(alpha, tau[0]))
        else:
            if alpha != 0:
                ax1.set_title(r'$\alpha = %0.1e$' %(alpha))
            else:
                ax1.set_title(r'$\alpha = %s$' %(alpha))
    
    elif args.spacetime is True and Path('image.npz').exists():
        Dict = np.load('image.npz')
        Image = Dict['Image']
        alpha = Dict['alpha']
        tau = Dict['tau']
        ndspace = int(Dict['ndspace'])
        Ntau = len(tau)
        
        if ndspace == 2:
            X = Dict['X']
            Y = Dict['Y']
            x = X[:, 0]
            y = Y[0, :]
            
            fig1 = plt.figure()
            ax1 = fig1.add_subplot(111)
            im = ax1.contourf(X, Y, Image, 100, cmap=plt.cm.magma)
            
            if wantColorbar:
                divider = make_axes_locatable(ax1)
                cax = divider.append_axes('right', size='5%', pad=0.05)
                cbar = plt.colorbar(im, cax=cax)
                cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                if Ntau == 1:
                    cbar.set_label(r'$\frac{1}{\vert\vert\varphi\vert\vert}$',
                                   labelpad=24, rotation=0, fontsize=18)
            
            if Ntau > 1:
                # Plot 2D section (slice) viewer
                Histogram = Dict['Histogram']
                fig2 = plt.figure()
                ax2 = fig2.add_subplot(111)
                ax2.volume = Histogram
                ax2.index = Ntau // 2
                space_viewer(ax2, Histogram, tau[ax2.index])
                
                # Plot isosurface of support of source function in space-time
                SpaceTimeVolume = np.swapaxes(Histogram, 1, 2)
                verts, faces, normals, values = measure.marching_cubes_lewiner(SpaceTimeVolume, level=isolevel)
                
                # Rescale coordinates of vertices to lie within x,y,tau ranges
                verts[:, 0] = verts[:, 0] * (x[-1] - x[0]) / (np.max(verts[:, 0]) - np.min(verts[:, 0])) + x[0]
                verts[:, 1] = verts[:, 1] * (tau[-1] - tau[0]) / (np.max(verts[:, 1]) - np.min(verts[:, 1])) + tau[0]
                verts[:, 2] = verts[:, 2] * (y[-1] - y[0]) / (np.max(verts[:, 2]) - np.min(verts[:, 2])) + y[0]
                
                fig3 = plt.figure()
                ax3 = fig3.add_subplot(111, projection='3d')
                ax3.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], color='silver')
                ax3.set_title('''Isosurface @ %s'''%(isolevel))
                if xu != '':
                    ax3.set_xlabel(xlabel + ' (%s)' %(xu))
                else:
                    ax3.set_xlabel(xlabel)
                if yu != '':
                    ax3.set_zlabel(ylabel + ' (%s)' %(yu))
                else:
                    ax3.set_zlabel(ylabel)
                if tu != '':
                    ax3.set_ylabel(r'$\tau$ (%s)' %(tu))
                else:
                    ax3.set_ylabel(r'$\tau$')
                
                # y-axis is taken to be the 'vertical space' axis in 2D space.
                # In the corresponding space-time figure, to maintain that the
                # second space dimension is vertical, the y-axis is now the
                # z-axis, and time (tau) is on the y-axis of the 3D figure.
                if invertY:
                    ax3.invert_zaxis()
                    
                if args.movie:
                    if not Path('./movie').exists():
                        Path('./movie').mkdir(parents=True, exist_ok=True)
                    for angle in trange(360, desc='Saving movie frames'):
                        ax3.view_init(elev=10., azim=angle)
                        fig3.savefig('./movie/movie%d' % angle + '.' + pltformat,
                                     format=pltformat)
            else:
                fig2 = None
                        
        elif ndspace == 3:
            X = Dict['X']
            Y = Dict['Y']
            Z = Dict['Z']
            x = X[:, 0, 0]
            y = Y[0, :, 0]
            z = Z[0, 0, :]
            
            verts, faces, normals, values = measure.marching_cubes_lewiner(Image, level=isolevel)
            # Rescale coordinates of vertices to lie within x,y,z ranges
            verts[:, 0] = verts[:, 0] * (x[-1] - x[0]) / (np.max(verts[:, 0]) - np.min(verts[:, 0])) + x[0]
            verts[:, 1] = verts[:, 1] * (y[-1] - y[0]) / (np.max(verts[:, 1]) - np.min(verts[:, 1])) + y[0]
            verts[:, 2] = verts[:, 2] * (z[-1] - z[0]) / (np.max(verts[:, 2]) - np.min(verts[:, 2])) + z[0]
            
            fig1 = plt.figure()
            ax1 = fig1.add_subplot(111, projection='3d')
            ax1.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], color='silver')
            
            if Ntau > 1:
                # Plot isosurface of support of source function in space-time
                Histogram = Dict['Histogram']
                fig2 = plt.figure()
                ax2 = fig2.add_subplot(111, projection='3d')
                ax2.volume = Histogram
                ax2.index = Ntau // 2
                ax2.set_aspect('equal')
                space_viewer(ax2, Histogram, tau[ax2.index])
            else:
                fig2 = None
        
        if Ntau == 1:
            if alpha != 0 and tu != '':
                ax1.set_title(r'$\alpha = %0.1e, \tau = %0.2f$ %s' %(alpha, tau[0], tu))
            elif alpha != 0 and tu == '':
                ax1.set_title(r'$\alpha = %0.1e, \tau = %0.2f$' %(alpha, tau[0]))
            elif alpha == 0 and tu != '':
                ax1.set_title(r'$\alpha = %s, \tau = %0.2f$ %s' %(alpha, tau[0], tu))
            elif alpha == 0 and tu == '':
                ax1.set_title(r'$\alpha = %s, \tau = %0.2f$' %(alpha, tau[0]))
        else:
            if alpha != 0:
                ax1.set_title(r'$\alpha = %0.1e$' %(alpha))
            else:
                ax1.set_title(r'$\alpha = %s$' %(alpha))
    
    elif args.spacetime is True and not Path('image.npz').exists():
        sys.exit('''
                 Error: Cannot view support of source function in space-time without
                 first solving for the given sampling grid. Run 
                     
                 'vzsolve --regPar=<value>'
                 
                 to make this option available.
                 ''')
        
    try:
        ax1
    except NameError:
        ax1 = None
    
    if ax1 is None:
        if receiverPoints.shape[1] == 2:
            fig1 = plt.figure()
            ax1 = fig1.add_subplot(111)
            
            if showSources:
                ax1.plot(sourcePoints[:, 0], sourcePoints[:, 1], 'k*')
            if showReceivers:
                ax1.plot(receiverPoints[:, 0], receiverPoints[:, 1], 'kv')
            if scatterer is not None and showScatterer:
                ax1.plot(scatterer[:, 0], scatterer[:, 1], '--', color='darkgray')
                
            if xu != '':
                ax1.set_xlabel(xlabel + ' (%s)' %(xu))
            else:
                ax1.set_xlabel(xlabel)

            if yu != '':
                ax1.set_ylabel(ylabel + ' (%s)' %(yu))
            else:
                ax1.set_ylabel(ylabel)
                
        elif receiverPoints.shape[1] == 3:
            fig1 = plt.figure()
            ax1 = fig1.add_subplot(111, projection='3d')
            
            if showSources:
                ax1.plot(sourcePoints[:, 0], sourcePoints[:, 1], sourcePoints[:, 2], 'k*')
            if showReceivers:
                ax1.plot(receiverPoints[:, 0], receiverPoints[:, 1], receiverPoints[:, 2], 'kv')
            if scatterer is not None and showScatterer:
                ax1.plot(scatterer[:, 0], scatterer[:, 1], scatterer[:, 2], '--', color='darkgray')
                
            if xu != '':
                ax1.set_xlabel(xlabel + ' (%s)' %(xu))
            else:
                ax1.set_xlabel(xlabel)

            if yu != '':
                ax1.set_ylabel(ylabel + ' (%s)' %(yu))
            else:
                ax1.set_ylabel(ylabel)
                
            if zu != '':
                ax1.set_zlabel(zlabel + ' (%s)' %(zu))
            else:
                ax1.set_zlabel(zlabel)
                
    else:
        if receiverPoints.shape[1] == 2:
            if showSources:
                ax1.plot(sourcePoints[:, 0], sourcePoints[:, 1], 'k*')
            if showReceivers:
                ax1.plot(receiverPoints[:, 0], receiverPoints[:, 1], 'kv')
            if scatterer is not None and showScatterer:
                ax1.plot(scatterer[:, 0], scatterer[:, 1], '--', color='darkgray')
                
            if xu != '':
                ax1.set_xlabel(xlabel + ' (%s)' %(xu))
            else:
                ax1.set_xlabel(xlabel)

            if yu != '':
                ax1.set_ylabel(ylabel + ' (%s)' %(yu))
            else:
                ax1.set_ylabel(ylabel)
        
        elif receiverPoints.shape[1] == 3:
            if showSources:
                ax1.plot(sourcePoints[:, 0], sourcePoints[:, 1], sourcePoints[:, 2], 'k*')
            if showReceivers:
                ax1.plot(receiverPoints[:, 0], receiverPoints[:, 1], receiverPoints[:, 2], 'kv')
            if scatterer is not None and showScatterer:
                ax1.plot(scatterer[:, 0], scatterer[:, 1], scatterer[:, 2], '--', color='darkgray')
            
            if xu != '':
                ax1.set_xlabel(xlabel + ' (%s)' %(xu))
            else:
                ax1.set_xlabel(xlabel)

            if yu != '':
                ax1.set_ylabel(ylabel + ' (%s)' %(yu))
            else:
                ax1.set_ylabel(ylabel)
                
            if zu != '':
                ax1.set_zlabel(zlabel + ' (%s)' %(zu))
            else:
                ax1.set_zlabel(zlabel)
    #==============================================================================
    ax1.set_aspect('equal')
    if invertX:
        ax1.invert_xaxis()
    if invertY:
        ax1.invert_yaxis()
    if invertZ:
        ax1.invert_zaxis()
        
    plt.tight_layout()
    fig1.savefig('image.' + pltformat, format=pltformat, bbox_inches='tight', transparent=True)
    
    if args.spacetime is True and fig2 is not None:
        remove_keymap_conflicts({'left', 'right', 'up', 'down', 'save'})
        fig2.canvas.mpl_connect('key_press_event', lambda event: process_key(event, Ntau, tau))
    
    plt.show()
