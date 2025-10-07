import numpy as np
import pandas as pd


from performancemodel.turboreactors import SimpleCorpsSimpleFlux_degrad_n_ratio_beta



def h_SCSF(list_degrad=None, S=None, N=0.9, W3R=3, ratio=1100, as_dict=False):
    """Compute measurements thanks to the SCSF simulator

    Args:
        list_degrad (list of list of float): list of degradation, first coordinate is compressor degradation, second coordinate is turbine degradatoin
        S (list of string, optional): sensors from which we want measurements Defaults to None.
        N (float, optional): engine speed in %. Defaults to 0.9.
        W3R (int, optional): Expected corrected airflow in compressor exit. Defaults to 3.
        ratio_design (float, optional): Expected ratio at the nozzle exit. Defaults to 1100.
        as_dict (bool, optional): Boolean if the output is a dict or an array. Defaults to False (array output)

    Returns:
        _type_: list of measurements
    """
    
    if list_degrad is None:
        list_degrad = [[1,1]]
    
    if S is None:
        S = ["P3", "T4", "W5R"]
    
    out = []
    for degrad in list_degrad:
        turboreactor = SimpleCorpsSimpleFlux_degrad_n_ratio_beta(N=N, degrad_comp=degrad[0], degrad_turb=degrad[1], W3R_design=W3R, ratio_design=ratio)
        mes = turboreactor.run(S)
        
        out.append(mes)

    if not as_dict:
        out = [np.array(list(x.values())) for x in out]
    
    return np.array(out)