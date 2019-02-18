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
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import FormatStrFormatter
import matplotlib.colors
from skimage import measure

#==============================================================================
# Define color class for printing to terminal
class FontColor:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

# General functions for plotting...

customCmapL = matplotlib.colors.LinearSegmentedColormap.from_list('', ['cyan', 'whitesmoke', 'magenta'])
customCmapD = matplotlib.colors.LinearSegmentedColormap.from_list('', ['cyan', 'black', 'magenta'])

def default_params():
    '''
    Sets the default ploting parameters used in both
    image/map plots as well as wiggle plots.
    
    This function takes no arguments.
    '''
    
    plotParams = {}
    #for both image and wiggle plots
    plotParams['pltformat'] = 'pdf'
    plotParams['view_mode'] = 'light'
    
    # for image/map plots
    plotParams['isolevel'] = 0.7
    plotParams['xlabel'] = ''
    plotParams['ylabel'] = ''
    plotParams['zlabel'] = ''
    plotParams['xu'] = ''
    plotParams['yu'] = ''
    plotParams['zu'] = ''
    plotParams['colormap'] = 'magma'
    plotParams['colorbar'] = False
    plotParams['invert_xaxis'] = False
    plotParams['invert_yaxis'] = False
    plotParams['invert_zaxis'] = False
    plotParams['show_scatterer'] = False
    plotParams['show_sources'] = True
    plotParams['show_receivers'] = True
    
    # for wiggle plots
    plotParams['pclip'] = 1
    plotParams['tu'] = ''
    plotParams['au'] = ''
    plotParams['data_title'] = 'Data'
    plotParams['tf_title'] = 'Test Function'
    
    # for frequency plots
    plotParams['fu'] = ''
    plotParams['fmin'] = 0
    plotParams['fmax'] = None
    plotParams['freq_title'] = 'Mean Amplitude Spectrum'
    plotParams['freq_ylabel'] = 'Amplitude'
    
    return plotParams



def setFigure(num_axes=1, mode='light', ax1_dim=2, ax2_dim=2):
    '''
    Create figure and axes objects styled for daytime viewing
    
    num_axes: number of axes objects to return (either 1 or 2)
    ax(1,2)_dim: number of dimensions in the axis object (i.e., 2D or 3D plot)
    ax(1,2)_grid: boolean (True of False)
    '''
    
    plt.style.use('ggplot')
    plt.rc('axes', titlesize=13)
    
    if num_axes == 1:
        fig = plt.figure()
        
        if ax1_dim == 2:
            ax1 = fig.add_subplot(111)
        elif ax1_dim == 3:
            ax1 = fig.add_subplot(111, projection='3d')
            
        if mode == 'light':
            ax1.spines['left'].set_color('black')
            ax1.spines['right'].set_color('black')
            ax1.spines['top'].set_color('black')
            ax1.spines['bottom'].set_color('black')
            ax1.cbaredgecolor = 'darkgray'
            ax1.alpha = 0.6
            ax1.shadecolor = 'black'
            ax1.shadealpha = 0.2
            ax1.linecolor = 'slategray'
            ax1.pointcolor = 'black'
            ax1.linewidth = 0.95
            ax1.labelcolor = '#4c4c4c'
            ax1.titlecolor = 'black'
            ax1.receivercolor = 'black'
            ax1.sourcecolor = 'black'
            ax1.activesourcecolor = 'darkcyan'
            ax1.inactivesourcecolor = 'darkgray'
            ax1.scatterercolor = 'darkgray'
            ax1.surfacecolor = 'c'
            ax1.customcmap = customCmapL
        
        elif mode == 'dark':
            plt.rc('grid', linestyle='solid', color='dimgray')
            plt.rc('legend', facecolor='353535')
            plt.rc('text', color='whitesmoke')
            
            fig.set_facecolor('black')
    
            ax1.tick_params(colors='555555')
            ax1.set_facecolor('darkslategray')
            ax1.spines['left'].set_color('darkgray')
            ax1.spines['right'].set_color('darkgray')
            ax1.spines['top'].set_color('darkgray')
            ax1.spines['bottom'].set_color('darkgray')
            ax1.cbaredgecolor = 'darkgray'
            ax1.alpha = 0.6
            ax1.shadecolor = 'black'
            ax1.shadealpha = 0.5
            ax1.linecolor = 'silver'
            ax1.pointcolor = 'silver'
            ax1.linewidth = 0.95
            ax1.labelcolor = '555555'
            ax1.titlecolor = '555555'
            ax1.receivercolor = 'darkgray'
            ax1.sourcecolor = 'darkgray'
            ax1.activesourcecolor = 'c'
            ax1.inactivesourcecolor = 'dimgray'
            ax1.scatterercolor = 'lightgray'
            ax1.surfacecolor = 'c'
            ax1.customcmap = customCmapD
            
        return fig, ax1
            
    elif num_axes == 2:
        fig = plt.figure(figsize=plt.figaspect(0.48))
        
        if ax1_dim == 2:
            ax1 = fig.add_subplot(121)
        elif ax1_dim == 3:
            ax1 = fig.add_subplot(121, projection='3d')
            
        if ax2_dim == 2:
            ax2 = fig.add_subplot(122)
        elif ax2_dim == 3:
            ax2 = fig.add_subplot(122, projection='3d')
            
        if mode == 'light':
            ax1.spines['left'].set_color('black')
            ax1.spines['right'].set_color('black')
            ax1.spines['top'].set_color('black')
            ax1.spines['bottom'].set_color('black')
            ax1.cbaredgecolor = 'darkgray'
            ax1.alpha = 0.6
            ax1.shadecolor = 'black'
            ax1.shadealpha = 0.2
            ax1.linecolor = 'slategray'
            ax1.pointcolor = 'black'
            ax1.linewidth = 0.95
            ax1.labelcolor = '#4c4c4c'
            ax1.titlecolor = 'black'
            ax1.receivercolor = 'black'
            ax1.sourcecolor = 'black'
            ax1.activesourcecolor = 'darkcyan'
            ax1.inactivesourcecolor = 'darkgray'
            ax1.scatterercolor = 'darkgray'
            ax1.surfacecolor = 'c'
            ax1.customcmap = customCmapL
            
            ax2.spines['left'].set_color('black')
            ax2.spines['right'].set_color('black')
            ax2.spines['top'].set_color('black')
            ax2.spines['bottom'].set_color('black')
            ax2.cbaredgecolor = 'darkgray'
            ax2.alpha = 0.6
            ax2.shadecolor = 'black'
            ax2.shadealpha = 0.2
            ax2.linecolor = 'slategray'
            ax2.pointcolor = 'black'
            ax2.linewidth = 0.95
            ax2.labelcolor = '#4c4c4c'
            ax2.titlecolor = 'black'
            ax2.receivercolor = 'black'
            ax2.sourcecolor = 'black'
            ax2.activesourcecolor = 'darkcyan'
            ax2.inactivesourcecolor = 'darkgray'
            ax2.scatterercolor = 'darkgray'
            ax2.surfacecolor = 'c'
            ax2.customcmap = customCmapL
            
        
        elif mode == 'dark':
            plt.rc('grid', linestyle='solid', color='dimgray')
            plt.rc('legend', facecolor='353535')
            plt.rc('text', color='whitesmoke')
            
            fig.set_facecolor('black')
    
            ax1.tick_params(colors='555555')
            ax1.set_facecolor('darkslategray')
            ax1.spines['left'].set_color('darkgray')
            ax1.spines['right'].set_color('darkgray')
            ax1.spines['top'].set_color('darkgray')
            ax1.spines['bottom'].set_color('darkgray')
            ax1.cbaredgecolor = 'darkgray'
            ax1.alpha = 0.6
            ax1.shadecolor = 'black'
            ax1.shadealpha = 0.5
            ax1.linecolor = 'silver'
            ax1.pointcolor = 'silver'
            ax1.linewidth = 0.95
            ax1.labelcolor = '555555'
            ax1.titlecolor = '555555'
            ax1.receivercolor = 'darkgray'
            ax1.sourcecolor = 'darkgray'
            ax1.activesourcecolor = 'c'
            ax1.inactivesourcecolor = 'dimgray'
            ax1.scatterercolor = 'lightgray'
            ax1.surfacecolor = 'c'
            ax1.customcmap = customCmapD
            
            ax2.tick_params(colors='555555')
            ax2.set_facecolor('darkslategray')
            ax2.spines['left'].set_color('darkgray')
            ax2.spines['right'].set_color('darkgray')
            ax2.spines['top'].set_color('darkgray')
            ax2.spines['bottom'].set_color('darkgray')
            ax2.cbaredgecolor = 'darkgray'
            ax2.alpha = 0.6
            ax2.shadecolor = 'black'
            ax2.shadealpha = 0.5
            ax2.linecolor = 'silver'
            ax2.pointcolor = 'silver'
            ax2.linewidth = 0.95
            ax2.labelcolor = '555555'
            ax2.titlecolor = '555555'
            ax2.receivercolor = 'darkgray'
            ax2.sourcecolor = 'darkgray'
            ax2.activesourcecolor = 'c'
            ax2.inactivesourcecolor = 'dimgray'
            ax2.scatterercolor = 'lightgray'
            ax2.surfacecolor = 'c'
            ax2.customcmap = customCmapD
        
        return fig, ax1, ax2



#==============================================================================
# Functions for plotting waveforms...
def set_ylabel(N, coordinates, id_number, flag, plotParams):
    '''
    Sets the appropriate y-axis label according to the object being plotted
    
    N: number of points along y-axis (>= 1)
    coordinates: location of object being referenced on y-axis (only used if N == 1)
    id_number: the number of the object being plotted (e.g., 'Receiver 1')
    flag: string parameter describing the type of the object ('data', 'testfunc', 'left', or 'right')
    plotParams: a dictionary of the plot parameters for styling
    '''
    
    if flag == 'data' or flag == 'testfunc' or flag == 'left':
        ylabel = 'Receiver'
    elif flag == 'right':
        ylabel = 'Source'
    
    if N == 1:
        
        # get amplitude units from plotParams
        au = plotParams['au']
        
        if coordinates is not None:
            # get units for x and y axes from plotParams
            xu = plotParams['xu']
            yu = plotParams['yu']
        
            # update ylabel to also show amplitude and coordinate information
            if coordinates.shape[1] == 2:
                if au != '' and xu != '' and yu != '':
                    ylabel = 'Amplitude (%s) [%s %s @ (%0.2f %s, %0.2f %s)]' %(au, ylabel, id_number,
                                                                               coordinates[0, 0], xu,
                                                                               coordinates[0, 1], yu)
                elif au == '' and xu != '' and yu != '':
                    ylabel = 'Amplitude [%s %s @ (%0.2f %s, %0.2f %s)]' %(ylabel, id_number,
                                                                          coordinates[0, 0], xu,
                                                                          coordinates[0, 1], yu)
                elif au != '' and xu == '' and yu == '':
                    ylabel = 'Amplitude (%s) [%s %s @ (%0.2f, %0.2f)]' %(au, ylabel, id_number,
                                                                         coordinates[0, 0],
                                                                         coordinates[0, 1])
                elif au == '' and xu == '' and yu == '':
                    ylabel = 'Amplitude [%s %s @ (%0.2f, %0.2f)]' %(ylabel, id_number,
                                                                    coordinates[0, 0],
                                                                    coordinates[0, 1])
                    
            elif coordinates.shape[1] == 3:
                # get units for z axis from plotParams
                zu = plotParams['zu']
            
                if au != '' and xu != '' and yu != '' and zu != '':
                    ylabel = 'Amplitude (%s) [%s %s @ (%0.2f %s, %0.2f %s, %0.2f %s)]' %(au, ylabel, id_number,
                                                                                         coordinates[0, 0], xu,
                                                                                         coordinates[0, 1], yu,
                                                                                         coordinates[0, 2], zu)
                elif au == '' and xu != '' and yu != '' and zu != '':
                    ylabel = 'Amplitude [%s %s @ (%0.2f %s, %0.2f %s, %0.2f %s)]' %(ylabel, id_number,
                                                                                    coordinates[0, 0], xu,
                                                                                    coordinates[0, 1], yu,
                                                                                    coordinates[0, 2], zu)
                elif au != '' and xu == '' and yu == '' and zu == '':
                    ylabel = 'Amplitude (%s) [%s %s @ (%0.2f, %0.2f, %0.2f)]' %(au, ylabel, id_number,
                                                                                coordinates[0, 0],
                                                                                coordinates[0, 1],
                                                                                coordinates[0, 2])
                elif au == '' and xu == '' and yu == '' and zu == '':
                    ylabel = 'Amplitude [%s %s @ (%0.2f, %0.2f, %0.2f)]' %(ylabel, id_number,
                                                                           coordinates[0, 0],
                                                                           coordinates[0, 1],
                                                                           coordinates[0, 2])
        else:   # coordinates is None
            
            # update ylabel to also show amplitude information
            if au != '':
                ylabel = 'Amplitude (%s) [%s %s]' %(au, ylabel, id_number)
            else:
                ylabel = 'Amplitude [%s %s]' %(ylabel, id_number)
    
    return ylabel



def plotWiggles(ax, X, xvals, interval, coordinates, title, flag, plotParams):
    ax.clear()
    N = X.shape[0]
    
    id_number = interval[0]
    ylabel = set_ylabel(N, coordinates, id_number, flag, plotParams)
    ax.set_ylabel(ylabel, color=ax.labelcolor)
    
    if N > 1:
        
        if N <= 18:
            ax.set_yticks(interval)                
            ax.set_yticklabels(interval)
            plt.setp(ax.get_yticklabels(), visible=True)
            plt.setp(ax.get_yticklines(), visible=True)
            
            # rescale all wiggle traces by largest displacement range
            scaleFactor = np.max(np.ptp(X, axis=1))
            if scaleFactor != 0:
                X /= scaleFactor
            
            for n in range(N):
                ax.plot(xvals, interval[n] + X[n, :], color=ax.linecolor, linewidth=ax.linewidth)
                ax.fill_between(xvals, interval[n], interval[n] + X[n, :],
                                where=(interval[n] + X[n, :] > interval[n]), color='m', alpha=ax.alpha)
                ax.fill_between(xvals, interval[n], interval[n] + X[n, :],
                                where=(interval[n] + X[n, :] < interval[n]), color='c', alpha=ax.alpha)
                
        elif N > 18 and N <= 70:            
            # rescale all wiggle traces by largest displacement range
            scaleFactor = np.max(np.ptp(X, axis=1))
            if scaleFactor != 0:
                X /= scaleFactor
            
            for n in range(N):
                ax.plot(xvals, interval[n] + X[n, :], color=ax.linecolor, linewidth=ax.linewidth)
                ax.fill_between(xvals, interval[n], interval[n] + X[n, :],
                                where=(interval[n] + X[n, :] > interval[n]), color='m', alpha=ax.alpha)
                ax.fill_between(xvals, interval[n], interval[n] + X[n, :],
                                where=(interval[n] + X[n, :] < interval[n]), color='c', alpha=ax.alpha)
        
        else:
            scaleFactor = np.max(np.abs(X))   
            pclip = plotParams['pclip']
            ax.pcolormesh(xvals, interval, X, vmin=-scaleFactor * pclip, vmax=scaleFactor * pclip, cmap=ax.customcmap)
                        
    else: # N == 1
        ax.yaxis.get_offset_text().set_x(-0.1)
        ax.plot(xvals, X[0, :], color=ax.linecolor, linewidth=ax.linewidth)
        ax.fill_between(xvals, 0, X[0, :], where=(X[0, :] > 0), color='m', alpha=ax.alpha)
        ax.fill_between(xvals, 0, X[0, :], where=(X[0, :] < 0), color='c', alpha=ax.alpha)
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        
    ax.set_title(title, color=ax.titlecolor)
    # get time units from plotParams
    tu = plotParams['tu']
    if tu != '':
        ax.set_xlabel('Time (%s)' %(tu), color=ax.labelcolor)
    else:
        ax.set_xlabel('Time', color=ax.labelcolor)
    
    ax.set_xlim([xvals[0], xvals[-1]])
        
    return ax


def plotFreqVectors(ax, volume, xvals, interval, coordinates, title, flag, plotParams):
    ax.clear()
    
    ax.set_title(title, color=ax.titlecolor)
    # get frequency units from plotParams
    fu = plotParams['fu']
    if fu != '':
        ax.set_xlabel('Frequency (%s)' %(fu), color=ax.labelcolor)
    else:
        ax.set_xlabel('Frequency', color=ax.labelcolor)
    
    N = volume.shape[0]
    id_number = interval[0]
    ylabel = set_ylabel(N, coordinates, id_number, flag, plotParams)
    ax.set_ylabel(ylabel, color=ax.labelcolor)
    if N > 1:
        # rescale all wiggle traces by largest displacement range
        scaleFactor = np.max(np.abs(volume))
        if scaleFactor != 0:
            volume /= scaleFactor
        
        return ax.pcolormesh(xvals, interval, volume, vmin=-1, vmax=1, cmap=ax.customcmap)
                        
    else: # N == 1
        ax.yaxis.get_offset_text().set_x(-0.1)
                
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        return ax.stem(xvals, volume[0, :], linefmt=ax.linecolor, markerfmt='mo')


#==============================================================================
# Functions for plotting maps and images
def plotMap(ax, index, receiverPoints, sourcePoints, scatterer, flag, plotParams):
    if index is None:
        
        if sourcePoints is None:
            if receiverPoints.shape[1] == 2:
                if plotParams['show_receivers']:
                    ax.plot(receiverPoints[:, 0], receiverPoints[:, 1], 'v', color=ax.receivercolor)
        
                if scatterer is not None and plotParams['show_scatterer']:
                    ax.plot(scatterer[:, 0], scatterer[:, 1], '--', color=ax.scatterercolor)
                
                  
            elif receiverPoints.shape[1] == 3:
                if plotParams['show_receivers']:
                    ax.plot(receiverPoints[:, 0], receiverPoints[:, 1], receiverPoints[:, 2], 'v', color=ax.receivercolor)
        
                if scatterer is not None and plotParams['show_scatterer']:
                    ax.plot(scatterer[:, 0], scatterer[:, 1], scatterer[:, 2], '--', color=ax.scatterercolor)
                
                zu = plotParams['zu']
                zlabel = plotParams['zlabel']
        
                if zu != '':
                    ax.set_zlabel(zlabel + ' (%s)' %(zu), color=ax.labelcolor)
                else:
                    ax.set_zlabel(zlabel, color=ax.labelcolor)
                    
                if plotParams['invert_zaxis']:
                    ax.invert_zaxis()
            
        else:   # sourcePoints exist
            if receiverPoints.shape[1] == 2:
                if plotParams['show_receivers']:
                    ax.plot(receiverPoints[:, 0], receiverPoints[:, 1], 'v', color=ax.receivercolor)
        
                if flag == 'data' and plotParams['show_sources']:
                    ax.plot(sourcePoints[:, 0], sourcePoints[:, 1], '*', color=ax.sourcecolor)
        
                elif flag == 'testfunc':
                        ax.plot(sourcePoints[:, 0], sourcePoints[:, 1], '.', color=ax.sourcecolor)
        
                if scatterer is not None and plotParams['show_scatterer']:
                    ax.plot(scatterer[:, 0], scatterer[:, 1], '--', color=ax.scatterercolor)
                
                
            elif receiverPoints.shape[1] == 3:
                if plotParams['show_receivers']:
                    ax.plot(receiverPoints[:, 0], receiverPoints[:, 1], receiverPoints[:, 2], 'v', color=ax.receivercolor)
        
                if flag == 'data' and plotParams['show_sources']:
                    ax.plot(sourcePoints[:, 0], sourcePoints[:, 1], sourcePoints[:, 2], '*', color=ax.sourcecolor)
        
                elif flag == 'testfunc':
                    ax.plot(sourcePoints[:, 0], sourcePoints[:, 1], sourcePoints[:, 2], '.', color=ax.sourcecolor)
        
                if scatterer is not None and plotParams['show_scatterer']:
                    ax.plot(scatterer[:, 0], scatterer[:, 1], scatterer[:, 2], '--', color=ax.scatterercolor)
                
                zu = plotParams['zu']
                zlabel = plotParams['zlabel']
        
                if zu != '':
                    ax.set_zlabel(zlabel + ' (%s)' %(zu), color=ax.labelcolor)
                else:
                    ax.set_zlabel(zlabel, color=ax.labelcolor)
                    
                if plotParams['invert_zaxis']:
                    ax.invert_zaxis()
         
    else:
        ax.clear()
        ax.set_title('Map', color=ax.titlecolor)
        
        if sourcePoints is None:
            if receiverPoints.shape[1] == 2:
                if plotParams['show_receivers']:
                    ax.plot(receiverPoints[:, 0], receiverPoints[:, 1], 'v', color=ax.receivercolor)
        
                if scatterer is not None and plotParams['show_scatterer']:
                    ax.plot(scatterer[:, 0], scatterer[:, 1], '--', color=ax.scatterercolor)
                
                  
            elif receiverPoints.shape[1] == 3:
                if plotParams['show_receivers']:
                    ax.plot(receiverPoints[:, 0], receiverPoints[:, 1], receiverPoints[:, 2], 'v', color=ax.receivercolor)
        
                if scatterer is not None and plotParams['show_scatterer']:
                    ax.plot(scatterer[:, 0], scatterer[:, 1], scatterer[:, 2], '--', color=ax.scatterercolor)
                
                zu = plotParams['zu']
                zlabel = plotParams['zlabel']
        
                if zu != '':
                    ax.set_zlabel(zlabel + ' (%s)' %(zu), color=ax.labelcolor)
                else:
                    ax.set_zlabel(zlabel, color=ax.labelcolor)
                
                if plotParams['invert_zaxis']:
                    ax.invert_zaxis()
            
        else:
            # delete the row corresponding to the current source (plot current source separately)
            sources = np.delete(sourcePoints, index, axis=0)
            currentSource = sourcePoints[index, :]
    
            if receiverPoints.shape[1] == 2:
                if plotParams['show_receivers']:
                    ax.plot(receiverPoints[:, 0], receiverPoints[:, 1], 'v', color=ax.receivercolor)
        
                if flag == 'data' and plotParams['show_sources']:
                    ax.plot(sources[:, 0], sources[:, 1], '*', color=ax.inactivesourcecolor)
                    ax.plot(currentSource[0], currentSource[1], marker='*', markersize=12, color=ax.activesourcecolor)
        
                elif flag == 'testfunc':
                    ax.plot(sources[:, 0], sources[:, 1], '.', color=ax.inactivesourcecolor)
                    ax.plot(currentSource[0], currentSource[1], marker='.', markersize=12, color=ax.activesourcecolor)
        
                if scatterer is not None and plotParams['show_scatterer']:
                    ax.plot(scatterer[:, 0], scatterer[:, 1], '--', color=ax.scatterercolor)
                
                  
            elif receiverPoints.shape[1] == 3:
                if plotParams['show_receivers']:
                    ax.plot(receiverPoints[:, 0], receiverPoints[:, 1], receiverPoints[:, 2], 'v', color=ax.receivercolor)
        
                if flag == 'data' and plotParams['show_sources']:
                    ax.plot(sources[:, 0], sources[:, 1], sources[:, 2], '*', color=ax.inactivesourcecolor)
                    ax.plot(currentSource[0], currentSource[1], currentSource[2], marker='*', markersize=12, color=ax.activesourcecolor)
        
                elif flag == 'testfunc':
                    ax.plot(sources[:, 0], sources[:, 1], sources[:, 2], '.', color=ax.inactivesourcecolor)
                    ax.plot(currentSource[0], currentSource[1], currentSource[2], marker='.', markersize=12, color=ax.activesourcecolor)
        
                if scatterer is not None and plotParams['show_scatterer']:
                    ax.plot(scatterer[:, 0], scatterer[:, 1], scatterer[:, 2], '--', color=ax.scatterercolor)
                
                zu = plotParams['zu']
                zlabel = plotParams['zlabel']
        
                if zu != '':
                    ax.set_zlabel(zlabel + ' (%s)' %(zu), color=ax.labelcolor)
                else:
                    ax.set_zlabel(zlabel, color=ax.labelcolor)
                
                if plotParams['invert_zaxis']:
                    ax.invert_zaxis()
        
    ax.grid(False)
    ax.set_aspect('equal')
    
    xu = plotParams['xu']
    xlabel = plotParams['xlabel']
    
    yu = plotParams['yu']
    ylabel = plotParams['ylabel']
    
    if xu != '':
        ax.set_xlabel(xlabel + ' (%s)' %(xu), color=ax.labelcolor)
    else:
        ax.set_xlabel(xlabel, color=ax.labelcolor)
        
    if yu != '':
        ax.set_ylabel(ylabel + ' (%s)' %(yu), color=ax.labelcolor)
    else:
        ax.set_ylabel(ylabel, color=ax.labelcolor)
    
    if plotParams['invert_xaxis']:
        ax.invert_xaxis()
    
    if plotParams['invert_yaxis']:
        ax.invert_yaxis()
    
    return ax


def isosurface(volume, level, x1, x2, x3):
    
    verts, faces, normals, values = measure.marching_cubes_lewiner(volume, level=level)
    
    # Rescale coordinates of vertices to lie within x,y,z ranges
    verts[:, 0] = verts[:, 0] * (x1[-1] - x1[0]) / (np.max(verts[:, 0]) - np.min(verts[:, 0])) + x1[0]
    verts[:, 1] = verts[:, 1] * (x2[-1] - x2[0]) / (np.max(verts[:, 1]) - np.min(verts[:, 1])) + x2[0]
    verts[:, 2] = verts[:, 2] * (x3[-1] - x3[0]) / (np.max(verts[:, 2]) - np.min(verts[:, 2])) + x3[0]
    
    return verts, faces
    

def image_viewer(ax, volume, method, alpha, tol, plotParams, X, Y, Z=None, tau=None):
    ax.clear()
    ax.grid(False)
    
    xu = plotParams['xu']
    xlabel = plotParams['xlabel']
    
    yu = plotParams['yu']
    ylabel= plotParams['ylabel']
    
    if xu != '':
        ax.set_xlabel(xlabel + ' (%s)' %(xu), color=ax.labelcolor)
    else:
        ax.set_xlabel(xlabel, color=ax.labelcolor)
        
    if yu != '':
        ax.set_ylabel(ylabel + ' (%s)' %(yu), color=ax.labelcolor)
    else:
        ax.set_ylabel(ylabel, color=ax.labelcolor)
    
    tu = plotParams['tu']
    
    #if method == 'svd':
    #    title = 'Method: SVD\n'
    #elif method == 'lsmr':
    #    title = 'Method: LSMR\n'
    #elif method == 'cg':
    #    title = 'Method: CG\n'
    
    if Z is None:
        colormap = plt.get_cmap(plotParams['colormap'])
        im = ax.contourf(X, Y, volume, 100, cmap=colormap)
        if plotParams['colorbar']:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            cbar = plt.colorbar(im, cax=cax)                
            cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            if tau is not None:
                cbar.set_label(r'$\frac{1}{\vert\vert\varphi\vert\vert}$',
                               labelpad=24, rotation=0, fontsize=18, color=ax.labelcolor)
        
        if alpha != 0.0:
            title = r'$\alpha = %0.1e$' %(alpha)
        else:
            title = r'$\alpha = %d$' %(alpha)
            
        if method != 'svd':
            title += ', tol = %01.e' %(tol)
    
        if tau is not None:
            if tu != '':
                title += r', $\tau = %0.2f$ %s' %(tau, tu)
            else:
                title += r', $\tau = %0.2f$' %(tau)
                
    else:
        x = X[:, 0, 0]
        y = Y[0, :, 0]
        z = Z[0, 0, :]
        
        isolevel = plotParams['isolevel']
        
        # Plot isosurface of support of source function in space
        verts, faces = isosurface(volume, isolevel, x, y, z)
        ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], color=ax.surfacecolor)
                
        zu = plotParams['zu']
        zlabel = plotParams['zlabel']
        
        if zu != '':
            ax.set_zlabel(zlabel + ' (%s)' %(zu), color=ax.labelcolor)
        else:
            ax.set_zlabel(zlabel, color=ax.labelcolor)
        
        if alpha != 0:
            title += r'Isosurface @ %s [$\alpha = %0.1e$]' %(isolevel, alpha)
        else:
            title += r'Isosurface @ %s [$\alpha = %s$]' %(isolevel, alpha)
    
        if tau is not None:
            if tu != '':
                title = title[:-1] + r', $\tau = %0.2f %s]' %(tau, tu)
            else:
                title = title[:-1] + r', $\tau = %0.2f]' %(tau)
        
    ax.set_title(title, color=ax.titlecolor)
        
    return ax


def plotImage(Dict, X, Y, Z, tau, plotParams, flag, movie=False):
    # Set up a two- or three-dimensional figure
    if Z is None:      
        fig, ax = setFigure(num_axes=1, mode=plotParams['view_mode'], ax1_dim=2) 
    else:
        fig, ax = setFigure(num_axes=1, mode=plotParams['view_mode'], ax1_dim=3)
    
    Image = Dict['Image'].reshape(X.shape)
    method = Dict['method']
    alpha = Dict['alpha']
    tol = Dict['tol']
    
    if Dict['domain'] == 'time':
        image_viewer(ax, Image, method, alpha, tol, plotParams, X, Y, Z, tau)
    else:
        image_viewer(ax, Image, method, alpha, tol, plotParams, X, Y, Z)
        
    return fig, ax
        
        
#==============================================================================
# General functions for interactive plotting...

def remove_keymap_conflicts(new_keys_set):
    '''
    Removes pre-defined keyboard events so that interactive
    plotting with various keys can be used
    '''
    
    for prop in plt.rcParams:
        if prop.startswith('keymap.'):
            keys = plt.rcParams[prop]
            remove_list = set(keys) & new_keys_set
            for key in remove_list:
                keys.remove(key)

#==============================================================================
# Specific functions for plotting data and test functions...

def wave_title(index, sinterval, sourcePoints, flag, plotParams):
    '''
    Creates a plot title for data arrays and test function arrays
    
    Parameters:
    index: a number indicating which source in 'interval' parameter produced the data
    interval: an integer array of source numbers
    sourcePoints: location of the sources, if known (equals 'None' otherwise)
    flag: a string parameter (either 'data' or 'testfunc')
    plotParams: a dictionary of plot parameters for styling
    '''
    
    # get units for x,y axes from plotParams
    xu = plotParams['xu']
    yu = plotParams['yu']
    
    if flag == 'data':
        # get type-specific title from plotParams
        data_title = plotParams['data_title']
        
        if sourcePoints is None:
            title = '%s [Record %s/%s]' %(data_title, sinterval[index], len(sinterval))
        
        elif sourcePoints.shape[1] == 2:
            if  xu != '' and yu != '':
                title = '%s [Source %s @ (%0.2f %s, %0.2f %s)]' %(data_title, sinterval[index],
                                                                  sourcePoints[index, 0], xu,
                                                                  sourcePoints[index, 1], yu)
            else:
                title = '%s [Source %s @ (%0.2f, %0.2f)]' %(data_title, sinterval[index],
                                                            sourcePoints[index, 0],
                                                            sourcePoints[index, 1])
        elif sourcePoints.shape[1] == 3:
            # get units for z axis from plotParams
            zu = plotParams['zu']
            
            if  xu != '' and yu != '' and zu != '':
                title = '%s [Source %s @ (%0.2f %s, %0.2f %s, %0.2f %s)]' %(data_title, sinterval[index],
                                                                            sourcePoints[index, 0], xu,
                                                                            sourcePoints[index, 1], yu,
                                                                            sourcePoints[index, 2], zu)
            else:
                title = '%s [Source %s @ (%0.2f, %0.2f, %0.2f)]' %(data_title, sinterval[index],
                                                                   sourcePoints[index, 0],
                                                                   sourcePoints[index, 1],
                                                                   sourcePoints[index, 2])
        
    else:   # flag == 'testfunc'
        # get type-specific title from plotParams
        tf_title = plotParams['tf_title']
        
        if sourcePoints.shape[1] == 2:
            if xu != '' and yu != '':
                title = '%s [$\\bf{z}$ @ (%0.2f %s, %0.2f %s)]' %(tf_title,
                                                                  sourcePoints[index, 0], xu,
                                                                  sourcePoints[index, 1], yu)
            elif xu == '' and yu == '':
                title = '%s [$\\bf{z}$ @ (%0.2f, %0.2f)]' %(tf_title,
                                                            sourcePoints[index, 0],
                                                            sourcePoints[index, 1])
                    
        elif sourcePoints.shape[1] == 3:
            # get units for z axis from plotParams
            zu = plotParams['zu']
            
            if xu != '' and yu != '' and zu != '':
                title = '%s [$\\bf{z}$ @ (%0.2f %s, %0.2f %s, %0.2f %s)]' %(tf_title,
                                                                            sourcePoints[index, 0], xu,
                                                                            sourcePoints[index, 1], yu,
                                                                            sourcePoints[index, 2], zu)
            elif xu == '' and yu == '' and zu == '':
                title = '%s [$\\bf{z}$ @ (%0.2f, %0.2f, %0.2f)]' %(tf_title,
                                                                   sourcePoints[index, 0],
                                                                   sourcePoints[index, 1],
                                                                   sourcePoints[index, 2])
    
    return title

    

def process_key_waves(event, time, rinterval, sinterval, receiverPoints,
                      sourcePoints, Ns, scatterer, show_map, flag, plotParams):
    '''
    Determines how to draw the next plot based on keyboard events
    
    event: a keyboard hit, either 'left', 'right', 'up', or 'down' arrow keys
    
    Passed parameters:
    time: an array of time values over which the singular vectors are defined
    rinterval: an interval or sampling of the receivers used
    sinterval: an interval or sampling of the sources used
    receiverPoints: coordinates of the receivers
    sourcePoints: coordinates of the sources
    scatterer: coordinates of the scatterer boundary
    show_map: Boolean value (True/False)
    flag: string parameter describing the type of the object ('data', 'testfunc', 'left', or 'right')
    plotParams: a dictionary of plot parameters for styling
    '''
    
    if show_map:
        fig = event.canvas.figure
        ax1 = fig.axes[0]
        ax2 = fig.axes[1]
        
        if event.key == 'left' or event.key == 'down':
            previous_wave(ax1, time, rinterval, receiverPoints,
                          sinterval, sourcePoints, Ns, flag, plotParams)
            previous_map(ax2, receiverPoints, sourcePoints, Ns, scatterer, flag, plotParams)
            
        elif event.key == 'right' or event.key == 'up':
            next_wave(ax1, time, rinterval, receiverPoints,
                      sinterval, sourcePoints, Ns, flag, plotParams)
            next_map(ax2, receiverPoints, sourcePoints, Ns, scatterer, flag, plotParams)
                      
    else:
        fig = event.canvas.figure
        ax = fig.axes[0]
        if event.key == 'left' or event.key == 'down':
            previous_wave(ax, time, rinterval, receiverPoints,
                          sinterval, sourcePoints, Ns, flag, plotParams)
        elif event.key == 'right' or event.key == 'up':
            next_wave(ax, time, rinterval, receiverPoints,
                      sinterval, sourcePoints, Ns, flag, plotParams)
    fig.canvas.draw()
         
    
    
def next_wave(ax, time, rinterval, receiverPoints,
              sinterval, sourcePoints, Ns, flag, plotParams):
    volume = ax.volume
    ax.index = (ax.index + 1) % Ns
    title = wave_title(ax.index, sinterval, sourcePoints, flag, plotParams)
    plotWiggles(ax, volume[:, :, ax.index], time, rinterval, 
                receiverPoints, title, flag, plotParams)
    
def previous_wave(ax, time, rinterval, receiverPoints,
                  sinterval, sourcePoints, Ns, flag, plotParams):
    volume = ax.volume
    ax.index = (ax.index - 1) % Ns  # wrap around using %
    title = wave_title(ax.index, sinterval, sourcePoints, flag, plotParams)
    plotWiggles(ax, volume[:, :, ax.index], time, rinterval,
                receiverPoints, title, flag, plotParams)


def previous_map(ax, receiverPoints, sourcePoints, Ns, scatterer, flag, plotParams):
    ax.index = (ax.index - 1) % Ns  # wrap around using %
    plotMap(ax, ax.index, receiverPoints, sourcePoints, scatterer, flag, plotParams)
    

def next_map(ax, receiverPoints, sourcePoints, Ns, scatterer, flag, plotParams):
    ax.index = (ax.index + 1) % Ns  # wrap around using %
    plotMap(ax, ax.index, receiverPoints, sourcePoints, scatterer, flag, plotParams)


#==============================================================================
# Specific functions for plotting singular vectors...
                
def vector_title(flag, n, cmplx_part=None):
    '''
    Creates a plot title for each left/right singular vector
    Returns a raw formatted string using LaTeX
    
    Parameters:
    flag: a string, either 'left' or 'right'
    n: a nonnegative integer
    dtype: data type (real or complex)
    '''
    
    if flag == 'left':
        if cmplx_part == 'real':
            title = r'$\mathrm{Re}\{\widehat{\phi}_{%d}(\mathbf{x}_r,\nu)\}$' %(n)
        elif cmplx_part == 'imag':
            title = r'$\mathrm{Im}\{\widehat{\phi}_{%d}(\mathbf{x}_r,\nu)\}$' %(n)
        else:
            title = r'Left-Singular Vector $\phi_{%d}(\mathbf{x}_r,t)$' %(n)
            
    
    elif flag == 'right':
        if cmplx_part == 'real':
            title = r'$\mathrm{Re}\{\widehat{\psi}_{%d}(\mathbf{x}_s,\nu)\}$' %(n)
        elif cmplx_part == 'imag':
            title = r'$\mathrm{Im}\{\widehat{\psi}_{%d}(\mathbf{x}_s,\nu)\}$' %(n)
        else:
            title = r'Right-Singular Vector $\psi_{%d}(\mathbf{x}_s,t)$' %(n)
    
    return title
          


def process_key_vectors(event, xvals, rinterval, sinterval,
                        receiverPoints, sourcePoints, plotParams,
                        dtype=None):
    '''
    Determines how to draw the next plot based on keyboard events
    
    event: a keyboard hit, either 'left', 'right', 'up', or 'down' arrow keys
    
    Passed parameters:
    time: an array of time values over which the singular vectors are defined
    t0: left endpoint of the time axis
    tf: right endpoint of the time axis
    rinterval: an interval or sampling of the receivers used
    sinterval: an interval or sampling of the sources used
    receiverPoints: coordinates of the receivers
    sourcePoints: coordinates of the sources
    '''
    
    fig = event.canvas.figure
    ax1 = fig.axes[0]
    ax2 = fig.axes[1]
    
    if event.key == 'left' or event.key == 'down':
        if dtype == 'cmplx_left':
            fig.suptitle('Left-Singular Vector', color=ax1.titlecolor, fontsize=16)
            previous_vector(ax1, xvals, rinterval, receiverPoints, 'left', 'real', plotParams)
            previous_vector(ax2, xvals, rinterval, receiverPoints, 'left', 'imag', plotParams)
        elif dtype == 'cmplx_right':
            fig.suptitle('Right-Singular Vector', color=ax1.titlecolor, fontsize=16)
            previous_vector(ax1, xvals, sinterval, sourcePoints, 'right', 'real', plotParams)
            previous_vector(ax2, xvals, sinterval, sourcePoints, 'right', 'imag', plotParams)
        else:
            previous_vector(ax1, xvals, rinterval, receiverPoints, 'left', None, plotParams)
            previous_vector(ax2, xvals, sinterval, sourcePoints, 'right', None, plotParams)
    
    elif event.key == 'right' or event.key == 'up':
        if dtype == 'cmplx_left':
            fig.suptitle('Left-Singular Vector', color=ax1.titlecolor, fontsize=16)
            next_vector(ax1, xvals, rinterval, receiverPoints, 'left', 'real', plotParams)
            next_vector(ax2, xvals, rinterval, receiverPoints, 'left', 'imag', plotParams)
        elif dtype == 'cmplx_right':
            fig.suptitle('Right-Singular Vector', color=ax1.titlecolor, fontsize=16)
            next_vector(ax1, xvals, sinterval, sourcePoints, 'right', 'real', plotParams)
            next_vector(ax2, xvals, sinterval, sourcePoints, 'right', 'imag', plotParams)
        else:
            next_vector(ax1, xvals, rinterval, receiverPoints, 'left', None, plotParams)
            next_vector(ax2, xvals, sinterval, sourcePoints, 'right', None, plotParams)
    
    fig.canvas.draw()
            


def next_vector(ax, xvals, interval, coordinates, flag, cmplx_part, plotParams):
    volume = ax.volume
    ax.index = (ax.index + 1) % volume.shape[2]
    title = vector_title(flag, ax.index + 1, cmplx_part)
    if cmplx_part is None:
        plotWiggles(ax, volume[:, :, ax.index], xvals, interval, 
                    coordinates, title, flag, plotParams)
    else:
        plotFreqVectors(ax, volume[:, :, ax.index], xvals, interval, 
                    coordinates, title, flag, plotParams)



def previous_vector(ax, xvals, interval, coordinates, flag, cmplx_part, plotParams):
    volume = ax.volume
    ax.index = (ax.index - 1) % volume.shape[2]  # wrap around using %
    title = vector_title(flag, ax.index + 1, cmplx_part)
    if cmplx_part is None:
        plotWiggles(ax, volume[:, :, ax.index], xvals, interval, 
                    coordinates, title, flag, plotParams)
    else:
        plotFreqVectors(ax, volume[:, :, ax.index], xvals, interval, 
                    coordinates, title, flag, plotParams)
    
    

#==============================================================================
# Specific functions for plotting images...    
def process_key_images(event, plotParams, alpha, X, Y, Z, Ntau, tau):
    fig = event.canvas.figure
    ax = fig.axes[0]
    
    if event.key == 'left' or event.key == 'down':
        previous_image(ax, plotParams, alpha, X, Y, Z, Ntau, tau)
    
    elif event.key == 'right' or event.key == 'up':
        next_image(ax, plotParams, alpha, X, Y, Z, Ntau, tau)
    
    fig.canvas.draw()

def previous_image(ax, plotParams, alpha, X, Y, Z, Ntau, tau):
    volume = ax.volume
    ax.index = (ax.index - 1) % Ntau  # wrap around using %
    if Z is None:
        image_viewer(ax, volume[:, :, ax.index], plotParams,
                     alpha, X, Y, Z, tau[ax.index])
    else:
        image_viewer(ax, volume[:, :, :, ax.index], plotParams,
                     alpha, X, Y, Z, tau[ax.index])
    
def next_image(ax, plotParams, alpha, X, Y, Z, Ntau, tau):
    volume = ax.volume
    ax.index = (ax.index + 1) % Ntau  # wrap around using %
    if Z is None:
        image_viewer(ax, volume[:, :, ax.index], plotParams,
                     alpha, X, Y, Z, tau[ax.index])
    else:
        image_viewer(ax, volume[:, :, :, ax.index], plotParams,
                     alpha, X, Y, Z, tau[ax.index])