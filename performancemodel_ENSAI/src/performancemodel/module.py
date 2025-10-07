import numpy as np
import pandas as pd


class ModuleTR():
    def __init__(self, T_in, P_in):
        self.T_in = T_in
        self.P_in = P_in
    


class Fan(ModuleTR):


    def __init__(self, T_in, P_in):
        ModuleTR.__init__(T_in, P_in)

