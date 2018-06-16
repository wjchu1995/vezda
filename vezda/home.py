import os
import vezda
import pkg_resources  # part of setuptools
import textwrap

def cli():
    version = pkg_resources.require('vezda')[0].version
    vzpath = os.path.dirname(os.path.abspath(vezda.__file__))
    print(textwrap.dedent(
         '''
         Vezda, Version %s
         Copyright \u00a9 2017-2018 Aaron C. Prunty
         Distributed under the terms of the Apache License, Version 2.0
         
         Install location:
             
         %s
         ''' % (version, vzpath)))