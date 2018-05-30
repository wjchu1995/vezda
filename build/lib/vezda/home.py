import os
import vezda

def cli():
    vzpath = os.path.dirname(os.path.abspath(vezda.__file__))
    print(vzpath)