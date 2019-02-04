import os
import pkg_resources  # part of setuptools
import textwrap
import vezda
from datetime import datetime
#from vezda.plot_utils import FontColor
#from vezda import (setDataPath, setWindow, plotWiggles, plotImage, setSamplingGrid,
#                   plotSpectra, SVD, Solve, addNoise)

#def VezdaInfo():
#    commandName = FontColor.BOLD + FontColor.GREEN + 'vezda:' + FontColor.END
#    description = ' view distribution metadata and command-line functions'
    
#    return commandName + description
    

def cli():
    version = pkg_resources.require('vezda')[0].version
    vzpath = os.path.dirname(os.path.abspath(vezda.__file__))
    print(textwrap.dedent(
         '''
         Vezda, Version %s
         Copyright \u00a9 2017-%s Aaron C. Prunty. All Rights Reserved.
         Distributed under the terms of the Apache License, Version 2.0.
         
         Install location:
             
         %s
         ''' % (version, datetime.today().year, vzpath)))#,
                #VezdaInfo()))),
                #setDataPath.info(),
                #setWindow.info(),
                #plotWiggles.info(),
                #plotImage.info(),
                #setSamplingGrid.info(),
                #plotSpectra.info(),
                #SVD.info(),
                #Solve.info(),
                #addNoise.info()
                #)))