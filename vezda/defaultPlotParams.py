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
import pickle

def default_params():
    plotParams = {}
    #for both image and wiggle plots
    pltformat = 'pdf'
    plotParams['pltformat'] = pltformat
    
    # for image/map plots
    isolevel = 0.7
    plotParams['isolevel'] = isolevel
    xlabel = ''
    plotParams['xlabel'] = xlabel
    ylabel = ''
    plotParams['ylabel'] = ylabel
    zlabel = ''
    plotParams['zlabel'] = zlabel
    xu = ''
    plotParams['xu'] = xu
    yu = ''
    plotParams['yu'] = yu
    zu = ''
    plotParams['zu'] = zu
    plotParams['colormap'] = 'magma'
    plotParams['colorbar'] = False
    invertX = False
    plotParams['invert_xaxis'] = invertX
    invertY = False
    plotParams['invert_yaxis'] = invertY
    invertZ = False
    plotParams['invert_zaxis'] = invertZ
    showScatterer = False
    plotParams['show_scatterer'] = showScatterer
    plotParams['show_sources'] = True
    plotParams['show_receivers'] = True
    
    # update parameters for wiggle plots based on passed arguments
    if args.tu is not None:
        tu = args.tu
    else:
        tu = ''
    plotParams['tu'] = tu
    
    if args.au is not None:
        au = args.au
    else:
        au = ''
    plotParams['au'] = au
    
    if args.title is not None:
        if args.type == 'data':
            data_title = args.title
            tf_title = 'Test Function'
        elif args.type == 'testfunc':
            data_title = 'Data'
            tf_title = args.title
    else:
        data_title = 'Data'
        tf_title = 'Test Function'
    plotParams['data_title'] = data_title
    plotParams['tf_title'] = tf_title
    
    pickle.dump(plotParams, open('plotParams.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)