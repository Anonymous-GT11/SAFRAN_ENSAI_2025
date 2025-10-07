import os, sys

path_ = os.path.dirname(os.getcwd())
if path_ not in sys.path:
    sys.path.append(path_)