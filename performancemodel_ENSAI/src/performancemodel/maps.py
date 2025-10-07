"""Here we create a class corresponding to the map of efficiency"""

import numpy as np


from performancemodel.helper import (
    get_interpolation_from_file_withn,
    get_2dinterpolation_from_file,
    get_interp_SRA,
    get_interp_beta
)


class Map:
    def __init__(self):
        self.name = "Fonction"

    def base_eval(self, *kwargs):
        return 1


class MapCompressor(Map):
    def __init__(self):
        Map.__init__(self)

    def eval(self, mair):
        return 10

    def eval_simple(self, mair, W4R, T4, T2):
        return (mair / W4R) * np.sqrt(T4 / T2) * (0.95 / 1.03)

    def eval_map_with_nmassflow(self, file, n, mair):
        function_pr, function_eff = get_interpolation_from_file_withn(file, n)

        pr = function_pr(mair)
        eff = function_eff(mair)
        return pr, eff

    def eval_map(self, file, inputs_name, outputs_name, inputs):
        function_map = get_2dinterpolation_from_file(file, inputs_name, outputs_name)

        outputs = function_map(inputs)
        outputs = [outputs]

        if len(outputs) > 1:
            output1 = [l[0] for l in outputs]
            output2 = [l[1] for l in outputs]
        else:
            output1 = outputs[0]
            output2 = None

        return output1, output2

    def eval_SRA(self, df, inputs):
        output1, output2 = get_interp_SRA(df, inputs)
        # output1, output2 = get_interp_SRA_old(file, inputs)
        return output1, output2

    def eval_beta(self, df, inputs):
        ## Inputs = [N, beta]
        output1, output2, output3 = get_interp_beta(df, inputs, ["N", "beta"], ["Efficiency", "Pressure Ratio", "Mass Flow"])

        return output1, output2, output3


class MapCombustor(Map):
    def __init__(self):
        Map.__init__(self)

    def eval(self, mair):
        return 0.95


class MapTurbine(Map):
    def __init__(self):
        Map.__init__(self)

    def eval(self, mair):
        return 1


class MapFCompressor(Map):
    def __init__(self):
        Map.__init__(self)

    def eval(self, mair):
        return 0.86


class MapFTurbine(Map):
    def __init__(self):
        Map.__init__(self)

    def eval(self, mair):
        return 0.9


class MapFCombustor(Map):
    def __init__(self):
        Map.__init__(self)

    def eval(self, mair):
        return 0.95
