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
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#==============================================================================
# General functions for plotting...

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
    plotParams['tu'] = ''
    plotParams['au'] = ''
    plotParams['data_title'] = 'Data'
    plotParams['tf_title'] = 'Test Function'
    
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
            ax1.alpha = 0.7
            ax1.shadecolor = 'black'
            ax1.shadealpha = 0.2
            ax1.linecolor = 'slategray'
            ax1.pointcolor = 'black'
            ax1.linewidth = 0.95
            ax1.labelcolor = '#4c4c4c'
            ax1.titlecolor = 'black'
        
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
            ax1.alpha = 0.6
            ax1.shadecolor = 'black'
            ax1.shadealpha = 0.5
            ax1.linecolor = 'silver'
            ax1.pointcolor = 'silver'
            ax1.linewidth = 0.95
            ax1.labelcolor = '555555'
            ax1.titlecolor = '555555'
            
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
            ax1.alpha = 0.7
            ax1.shadecolor = 'black'
            ax1.shadealpha = 0.2
            ax1.linecolor = 'slategray'
            ax1.pointcolor = 'black'
            ax1.linewidth = 0.95
            ax1.labelcolor = '#4c4c4c'
            ax1.titlecolor = 'black'
            
            ax2.spines['left'].set_color('black')
            ax2.spines['right'].set_color('black')
            ax2.spines['top'].set_color('black')
            ax2.spines['bottom'].set_color('black')
            ax2.alpha = 0.7
            ax2.shadecolor = 'black'
            ax2.shadealpha = 0.2
            ax2.linecolor = 'slategray'
            ax2.pointcolor = 'black'
            ax2.linewidth = 0.95
            ax2.labelcolor = '#4c4c4c'
            ax2.titlecolor = 'black'
        
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
            ax1.alpha = 0.6
            ax1.shadecolor = 'black'
            ax1.shadealpha = 0.5
            ax1.linecolor = 'silver'
            ax1.pointcolor = 'silver'
            ax1.linewidth = 0.95
            ax1.labelcolor = '555555'
            ax1.titlecolor = '555555'
            
            ax2.tick_params(colors='555555')
            ax2.set_facecolor('darkslategray')
            ax2.spines['left'].set_color('darkgray')
            ax2.spines['right'].set_color('darkgray')
            ax2.spines['top'].set_color('darkgray')
            ax2.spines['bottom'].set_color('darkgray')
            ax2.alpha = 0.6
            ax2.shadecolor = 'black'
            ax2.shadealpha = 0.5
            ax2.linecolor = 'silver'
            ax2.pointcolor = 'silver'
            ax2.linewidth = 0.95
            ax2.labelcolor = '555555'
            ax2.titlecolor = '555555'
        
        return fig, ax1, ax2


def set_ylabel(N, coordinates, pltstart, flag, plotParams):
    '''
    Sets the appropriate y-axis label according to the object being plotted
    
    N: number of points along y-axis (>= 1)
    coordinates: location of object being referenced on y-axis (only used if N == 1)
    pltstart: the number of the object being plotted (e.g., 'Receiver 1')
    flag: string parameter describing the type of the object ('data', 'testfunc', 'left', or 'right')
    plotParams: a dictionary of the plot parameters for styling
    '''
    
    if flag == 'data' or flag == 'testfunc' or flag == 'left':
        ylabel = 'Receiver'
    elif flag == 'right':
        ylabel = 'Source'
    
    if N == 1:
        # get units for x,y,z axes from plotParams
        xu = plotParams['xu']
        yu = plotParams['yu']
        zu = plotParams['zu']
        
        # get amplitude units from plotParams
        au = plotParams['au']
        
        # update ylabel to also show amplitude and coordinate information
        if coordinates.shape[1] == 2:
            if au != '' and xu != '' and yu != '':
                ylabel = 'Amplitude (%s) [%s %s @ (%0.2f %s, %0.2f %s)]' %(au, ylabel, pltstart,
                                                                           coordinates[0, 0], xu,
                                                                           coordinates[0, 1], yu)
            elif au == '' and xu != '' and yu != '':
                ylabel = 'Amplitude [%s %s @ (%0.2f %s, %0.2f %s)]' %(ylabel, pltstart,
                                                                      coordinates[0, 0], xu,
                                                                      coordinates[0, 1], yu)
            elif au != '' and xu == '' and yu == '':
                ylabel = 'Amplitude (%s) [%s %s @ (%0.2f, %0.2f)]' %(au, ylabel, pltstart,
                                                                     coordinates[0, 0],
                                                                     coordinates[0, 1])
            elif au == '' and xu == '' and yu == '':
                ylabel = 'Amplitude [%s %s @ (%0.2f, %0.2f)]' %(ylabel, pltstart,
                                                                coordinates[0, 0],
                                                                coordinates[0, 1])
                    
        elif coordinates.shape[1] == 3:
            if au != '' and xu != '' and yu != '' and zu != '':
                ylabel = 'Amplitude (%s) [%s %s @ (%0.2f %s, %0.2f %s, %0.2f %s)]' %(au, ylabel, pltstart,
                                                                                     coordinates[0, 0], xu,
                                                                                     coordinates[0, 1], yu,
                                                                                     coordinates[0, 2], zu)
            elif au == '' and xu != '' and yu != '' and zu != '':
                ylabel = 'Amplitude [%s %s @ (%0.2f %s, %0.2f %s, %0.2f %s)]' %(ylabel, pltstart,
                                                                                coordinates[0, 0], xu,
                                                                                coordinates[0, 1], yu,
                                                                                coordinates[0, 2], zu)
            elif au != '' and xu == '' and yu == '' and zu == '':
                ylabel = 'Amplitude (%s) [%s %s @ (%0.2f, %0.2f, %0.2f)]' %(au, ylabel, pltstart,
                                                                            coordinates[0, 0],
                                                                            coordinates[0, 1],
                                                                            coordinates[0, 2])
            elif au == '' and xu == '' and yu == '' and zu == '':
                ylabel = 'Amplitude [%s %s @ (%0.2f, %0.2f, %0.2f)]' %(ylabel, pltstart,
                                                                       coordinates[0, 0],
                                                                       coordinates[0, 1],
                                                                       coordinates[0, 2])
    
    return ylabel



def plotWiggles(ax, X, time, t0, tf, pltstart, interval, coordinates, title, flag, plotParams):
    ax.clear()
    N = X.shape[0]
    
    if N > 1:
        ylabel = set_ylabel(N, coordinates, pltstart, flag, plotParams)
        ax.set_ylabel(ylabel, color=ax.labelcolor)
        ax.set_yticks(interval)                
        ax.set_yticklabels(pltstart + interval)
        plt.setp(ax.get_yticklabels(), visible=True)
        plt.setp(ax.get_yticklines(),visible=True)
        
        # rescale all wiggle traces by largest displacement range
        scaleFactor = np.max(np.ptp(X, axis=1))
        if scaleFactor != 0:
            X /= scaleFactor
            
        for n in range(N):
            ax.plot(time, interval[n] + X[n, :], color=ax.linecolor, linewidth=ax.linewidth)
            ax.fill_between(time, interval[n], interval[n] + X[n, :],
                            where=(interval[n] + X[n, :] > interval[n]), color='m', alpha=ax.alpha)
            ax.fill_between(time, interval[n], interval[n] + X[n, :],
                            where=(interval[n] + X[n, :] < interval[n]), color='c', alpha=ax.alpha)
                        
    else: # N == 1
        ax.yaxis.get_offset_text().set_x(-0.1)
        ylabel = set_ylabel(N, coordinates, pltstart, flag, plotParams)
        ax.set_ylabel(ylabel, color=ax.labelcolor)
                
        ax.plot(time, X[0, :], color=ax.linecolor, linewidth=ax.linewidth)
        ax.fill_between(time, 0, X[0, :], where=(X[0, :] > 0), color='m', alpha=ax.alpha)
        ax.fill_between(time, 0, X[0, :], where=(X[0, :] < 0), color='c', alpha=ax.alpha)
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        
    ax.set_title(title, color=ax.titlecolor)
    
    # get time units from plotParams
    tu = plotParams['tu']
    if tu != '':
        ax.set_xlabel('Time (%s)' %(tu), color=ax.labelcolor)
    else:
        ax.set_xlabel('Time', color=ax.labelcolor)
    
    # highlight regions of interest along the time axis
    if flag == 'data' and (time[0] != t0 or time[-1] != tf):
        # For data plots, the full recording time is always plotted.
        # In this case, t0 and tf denote the beginning and end of a
        # time window. The time window of the interest is highlighted
        # while the rest of the time axis is shaded.
        ax.axvspan(time[0], t0, color=ax.shadecolor, alpha=ax.shadealpha, zorder=10)
        ax.axvspan(tf, time[-1], color=ax.shadecolor, alpha=ax.shadealpha, zorder=10)
    
        # set limits of time axis according to original recording time interval
        ax.set_xlim([time[0], time[-1]])
        
    else:
        # For all other plots, only the time window of interest is plotted.
        # In this case, t0 and tf denote the beginning and end of the full
        # recording time interval. The time window of interest is highlighted
        # while the rest of the time axis is shaded.
        ax.set_xlim([t0, tf])
        
        ax.axvspan(t0, time[0], color=ax.shadecolor, alpha=ax.shadealpha, zorder=10)
        ax.axvspan(time[-1], tf, color=ax.shadecolor, alpha=ax.shadealpha, zorder=10)
        
    return ax



def plotMap(ax, index, receiverPoints, sourcePoints, scatterer, flag, plotParams):
    ax.clear()
    ax.grid(False)
    
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
    
    # delete the row corresponding to the current source (plot current source separately)
    sources = np.delete(sourcePoints, index, axis=0)
    currentSource = sourcePoints[index, :]
    
    if receiverPoints.shape[1] == 2:
        ax.plot(receiverPoints[:, 0], receiverPoints[:, 1], 'v', color='k')
        
        if flag == 'data':
            ax.plot(sources[:, 0], sources[:, 1], '*', color='silver')
            ax.plot(currentSource[0], currentSource[1], marker='*', markersize=12, color='darkcyan')
        
        elif flag == 'testfunc':
            ax.plot(sources[:, 0], sources[:, 1], '.', color='silver')
            ax.plot(currentSource[0], currentSource[1], marker='.', markersize=12, color='darkcyan')
        
        if scatterer is not None and plotParams['show_scatterer']:
            ax.plot(scatterer[:, 0], scatterer[:, 1], '--', color='darkgray')
                
                  
    elif receiverPoints.shape[1] == 3:
        ax.plot(receiverPoints[:, 0], receiverPoints[:, 1], receiverPoints[:, 2], 'v', color='k')
        
        if flag == 'data':
            ax.plot(sources[:, 0], sources[:, 1], sources[:, 2], '*', color='silver')
            ax.plot(currentSource[0], currentSource[1], currentSource[2], marker='*', markersize=12, color='darkcyan')
        
        elif flag == 'testfunc':
            ax.plot(sources[:, 0], sources[:, 1], sources[:, 2], '.', color='silver')
            ax.plot(currentSource[0], currentSource[1], currentSource[2], marker='.', markersize=12, color='darkcyan')
        
        if scatterer is not None and plotParams['show_scatterer']:
            ax.plot(scatterer[:, 0], scatterer[:, 1], scatterer[:, 2], '--', color='darkgray')
                
        zu = plotParams['zu']
        zlabel = plotParams['zlabel']
        
        if zu != '':
            ax.set_zlabel(zlabel + ' (%s)' %(zu), color=ax.labelcolor)
        else:
            ax.set_zlabel(zlabel, color=ax.labelcolor)
        
    ax.set_title('Map')
    ax.set_aspect('equal')
    
    if plotParams['invert_xaxis']:
        ax.invert_xaxis()
    
    if plotParams['invert_yaxis']:
        ax.invert_yaxis()
    
    if plotParams['invert_zaxis']:
        ax.invert_zaxis()
    
    return ax



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
                
#def save_key(event):
#   if args.type == 'data':
#       fig.savefig(args.type + '_src' + str(sinterval[ax.index]) + '.' + args.format,
#                  format=pltformat, bbox_inches='tight', transparent=True)
#   elif args.type == 'testfunc':
#       fig.savefig(args.type + '_pnt' + str(sinterval[ax.index]) + '.' + args.format,
#                  format=pltformat, bbox_inches='tight', transparent=True)
    




#==============================================================================
# Specific functions for plotting data and test functions...

def wave_title(index, sinterval, sourcePoints, flag, plotParams):
    '''
    Creates a plot title for data arrays and test function arrays
    
    Parameters:
    index: a number indicating which source in 'interval' parameter produced the data
    interval: an integer array of source numbers
    sourcePoints: location of the current source
    flag: a string parameter (either 'data' or 'testfunc')
    plotParams: a dictionary of plot parameters for styling
    '''
    
    # get units for x,y,z axes from plotParams
    xu = plotParams['xu']
    yu = plotParams['yu']
    zu = plotParams['zu']
    
    if sourcePoints.shape[1] == 2:
        if flag == 'data':
            # get type-specific title from plotParams
            data_title = plotParams['data_title']
            
            if  xu != '' and yu != '':
                title = '%s [Source %s @ (%0.2f %s, %0.2f %s)]' %(data_title, sinterval[index],
                                                                  sourcePoints[index, 0], xu,
                                                                  sourcePoints[index, 1], yu)
            else:
                title = '%s [Source %s @ (%0.2f, %0.2f)]' %(data_title, sinterval[index],
                                                            sourcePoints[index, 0],
                                                            sourcePoints[index, 1])
                
        elif flag == 'testfunc':
            # get type-specific title from plotParams
            tf_title = plotParams['tf_title']
            
            if xu != '' and yu != '':
                title = '%s [$\\bf{z}$ @ (%0.2f %s, %0.2f %s)]' %(tf_title,
                                                                  sourcePoints[index, 0], xu,
                                                                  sourcePoints[index, 1], yu)
            elif xu == '' and yu == '':
                title = '%s [$\\bf{z}$ @ (%0.2f, %0.2f)]' %(tf_title,
                                                            sourcePoints[index, 0],
                                                            sourcePoints[index, 1])
                    
    elif sourcePoints.shape[1] == 3:
        if flag == 'data':
            # get type-specific title from plotParams
            data_title = plotParams['data_title']
            
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
                
        elif flag == 'testfunc':
            # get type-specific title from plotParams
            tf_title = plotParams['tf_title']
            
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

    

def process_key_waves(event, time, t0, tf, pltstart, rinterval, sinterval,
                receiverPoints, sourcePoints, scatterer, show_map, flag, plotParams):
    '''
    Determines how to draw the next plot based on keyboard events
    
    event: a keyboard hit, either 'left', 'right', 'up', or 'down' arrow keys
    
    Passed parameters:
    time: an array of time values over which the singular vectors are defined
    t0: left endpoint of the time axis
    tf: right endpoint of the time axis
    pltstart: the number of the object being plotted (e.g., Receiver '1')
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
            previous_wave(ax1, time, t0, tf, pltstart, rinterval, receiverPoints,
                          sinterval, sourcePoints, flag, plotParams)
            previous_map(ax2, receiverPoints, sourcePoints, scatterer, flag, plotParams)
            
        elif event.key == 'right' or event.key == 'up':
            next_wave(ax1, time, t0, tf, pltstart, rinterval, receiverPoints,
                      sinterval, sourcePoints, flag, plotParams)
            next_map(ax2, receiverPoints, sourcePoints, scatterer, flag, plotParams)
                      
    else:
        fig = event.canvas.figure
        ax = fig.axes[0]
        if event.key == 'left' or event.key == 'down':
            previous_wave(ax, time, t0, tf, pltstart, rinterval, receiverPoints,
                          sinterval, sourcePoints, flag, plotParams)
        elif event.key == 'right' or event.key == 'up':
            next_wave(ax, time, t0, tf, pltstart, rinterval, receiverPoints,
                      sinterval, sourcePoints, flag, plotParams)
    fig.canvas.draw()
         
    
    
def next_wave(ax, time, t0, tf, pltstart, rinterval, receiverPoints,
              sinterval, sourcePoints, flag, plotParams):
    volume = ax.volume
    ax.index = (ax.index + 1) % volume.shape[2]
    title = wave_title(ax.index, sinterval, sourcePoints, flag, plotParams)
    plotWiggles(ax, volume[:, :, ax.index], time, t0, tf, pltstart, rinterval, 
                receiverPoints, title, flag, plotParams)
    
def previous_wave(ax, time, t0, tf, pltstart, rinterval, receiverPoints,
                  sinterval, sourcePoints, flag, plotParams):
    volume = ax.volume
    ax.index = (ax.index - 1) % volume.shape[2]  # wrap around using %
    title = wave_title(ax.index, sinterval, sourcePoints, flag, plotParams)
    plotWiggles(ax, volume[:, :, ax.index], time, t0, tf, pltstart, rinterval,
                receiverPoints, title, flag, plotParams)



def previous_map(ax, receiverPoints, sourcePoints, scatterer, flag, plotParams):
    ax.index = (ax.index - 1) % sourcePoints.shape[0]  # wrap around using %
    plotMap(ax, ax.index, receiverPoints, sourcePoints, scatterer, flag, plotParams)
    


def next_map(ax, receiverPoints, sourcePoints, scatterer, flag, plotParams):
    ax.index = (ax.index + 1) % sourcePoints.shape[0]  # wrap around using %
    plotMap(ax, ax.index, receiverPoints, sourcePoints, scatterer, flag, plotParams)


#==============================================================================
# Specific functions for plotting singular vectors...
                
def vector_title(flag, n):
    '''
    Creates a plot title for each left/right singular vector
    Returns a raw formatted string using LaTeX
    
    Parameters:
    flag: a string, either 'left' or 'right'
    n: a nonnegative integer 
    '''
    
    if flag == 'left':
        title = r'Left Singular Vector $\phi_{%d}(\mathbf{x}_r,t)$' %(n)
    
    elif flag == 'right':
        title = r'Right Singular Vector $\psi_{%d}(\mathbf{x}_s,t_s)$' %(n)
    
    return title
          


def process_key_vectors(event, time, t0, tf, pltrstart, pltsstart, 
                        rinterval, sinterval, receiverPoints, sourcePoints, plotParams):
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
        previous_vector(ax1, time, t0, tf, pltrstart, rinterval, receiverPoints, 'left', plotParams)
        previous_vector(ax2, time, t0, tf, pltsstart, sinterval, sourcePoints, 'right', plotParams)
    
    elif event.key == 'right' or event.key == 'up':
        next_vector(ax1, time, t0, tf, pltrstart, rinterval, receiverPoints, 'left', plotParams)
        next_vector(ax2, time, t0, tf, pltsstart, sinterval, sourcePoints, 'right', plotParams)
    
    fig.canvas.draw()
            


def next_vector(ax, time, t0, tf, pltstart, interval, coordinates, flag, plotParams):
    volume = ax.volume
    ax.index = (ax.index + 1) % volume.shape[2]
    title = vector_title(flag, ax.index)
    plotWiggles(ax, volume[:, :, ax.index], time, t0, tf, pltstart, interval, 
                coordinates, title, flag, plotParams)



def previous_vector(ax, time, t0, tf, pltstart, interval, coordinates, flag, plotParams):
    volume = ax.volume
    ax.index = (ax.index - 1) % volume.shape[2]  # wrap around using %
    title = vector_title(flag, ax.index)
    plotWiggles(ax, volume[:, :, ax.index], time, t0, tf, pltstart, interval,
                coordinates, title, flag, plotParams)