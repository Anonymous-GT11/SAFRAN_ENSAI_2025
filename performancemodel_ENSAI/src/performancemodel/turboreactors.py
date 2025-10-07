"""Here we create a class corresponding to a turboreactor
Add constants and their values
Add references for equations
"""

from importlib.resources import files

import numpy as np
from sympy import solve, symbols
from scipy.optimize import root, fsolve, least_squares, minimize

from performancemodel.helper import (
    create_data,
    create_data_beta,
    create_data_from_csv,
    derivative_function,
    get_minmax,
    get_operation_point,
    newton,
)
from performancemodel.maps import (
    Map,
    MapCombustor,
    MapCompressor,
    MapFCombustor,
    MapFCompressor,
    MapTurbine,
)


class TurboReactor:
    """
    General Class Turboreactor.
    A Turboreactor is the representation of an engine simple corps simple flux where we compute quantities such as
    Temperature, Pressure, Mass Air Flow as the iar travel through the differente modules.

    General Attributes:
        name : str, name of the turboreactor, not important
        gamma : float, gaz constant (see Supelec course)
        PCI : float, constant (see Supelec course)
        R : float, constant (see Supelec course)
        cp : float, constant (see Supelec course)
        FAR4 : float, ratio between mass air flow exiting the combustion chamber and mass air flow entering the combustion chamber
        Tamb : float, ambient temperature around the engine
        Pamb : float, ambient pressure of the air
        M0 : float, initial speed as number of Mach of the ambient air

    """

    def __init__(
        self,
        name="Turboreactor",
        gamma=1.4,
        PCI=44 * 10**6,
        R=287.05,
        FAR4=0.03,
        Tamb=300,
        Pamb=101325,
        M0=1,
    ):
        """
        Initialization function

        Parameters:
            name : str, name of the turboreactor, not important
            gamma : float, gaz constant (see Supelec course)
            PCI : float, constant (see Supelec course)
            R : float, constant (see Supelec course)
            FAR4 : float, ratio between mass air flow exiting the combustion chamber and mass air flow entering the combustion chamber
            Tamb : float, ambient temperature around the engine
            Pamb : float, ambient pressure of the air
            M0 : float, initial speed as number of Mach of the ambient air


        Output:
            None
        """
        self.name = name
        self.gamma = gamma
        self.PCI = PCI
        self.R = R
        self.cp = self.gamma * self.R / (self.gamma - 1)
        self.FAR4 = FAR4

        self.Tamb = Tamb
        self.Pamb = Pamb
        self.M0 = M0

    def compute_plan_0(self):
        """
        Compute plan 0, that is the plan at the entry of the engine. Initialize P0 and T0 from Tamb and Pamb

        Parameters:
            None

        Output:
            None
        """
        self.T0 = self.Tamb * (1 + ((self.gamma - 1) * self.M0**2) / 2)
        self.P0 = self.Pamb * (1 + ((self.gamma - 1) * self.M0**2) / 2) ** (
            self.gamma / (self.gamma - 1)
        )

    def compute_plan_1(self, W0=15):
        """
        Compute plan 1, that is the plan at the entry of the fan. Propagate temprature, pressure and airflow through the engine

        Parameters:
            None

        Output:
            None
        """
        self.W0 = W0

        self.T1 = self.T0
        self.P1 = self.P0
        self.W1 = self.W0

    def compute_plan_2(self, Tref=288.15, Pref=101325):
        """
        Compute plan 2, that is the plan at the entry of the compressor. Propagate temperature, pressure and airflow through the engine

        Parameters:
            None

        Output:
            None
        """
        self.T2 = self.T1
        self.P2 = self.P1
        self.W2 = self.W1
        self.W2R = self.W2 * (np.sqrt(self.T2 / Tref) / (self.P2 / Pref))

    def print_parameters(self):
        print("P2", self.P2)
        print("T2", self.T2)
        print("W4", self.W4)
        print("W2R", self.W2R)
        print("T3", self.T3)
        print("T2", self.T2)
        print("T5", self.T5)
        print("T5is", self.T5is)
        print("P4", self.P4)
        print("P5", self.P5)
        print("P8", self.P8)
        print("T8", self.T8)
        print("Wf", self.Wf)


class SimpleCorpsSimpleFlux_Design(TurboReactor):
    """
    A subclass of Turboreactor with design fonctionnality. No map compressor needed. Only equations and values on how the engine works.

    Special Attributes:


    """

    def __init__(self, name="Engine_design"):
        """Initialize an instance

        Args:
            name (str, optional): Name of the turboreactor. Defaults to "Engine_design".
        """
        TurboReactor.__init__(self, name)

    def compute_plan_3(self, OPR=10, eff_comp=0.86):
        """
        Compute plan 3, that is the plan at the exit of the compressor. In the design case, we simply have a given OPR and efficiency instead of a map.
        There is nothing to optimize, just compute the new P3 and new T3

        Parameters:
            OPR : float, Pressure ratio between the air pressure at the entry and the exit of the compressor
            eff_comp : float, The efficiency of the compressor that helps to calculate the temperature rise when compressing

        Output:
            None
        """
        self.P3 = OPR * self.P2
        self.T3 = self.T2 * (
            1 + ((self.P3 / self.P2) ** ((self.gamma - 1) / self.gamma) - 1) / eff_comp
        )

    def compute_plan_4(self, T4=1700, C=0.05):
        """
        Compute plan 4, that is the plan at the exit of the combustion chamber. The temperature T4 is choosen by the pilot or any guy in charge.
        C represents the loss of pressure. The combustion chamber is seen as a ...

        Parameters:
            T4 : float, Temperature at which the air will be rising. Default to 1400
            C : float (optionnal), The loss of pressure in the combustion chamber. Default to 0.95

        Output:
            None
        """

        self.T4 = T4
        self.P4 = self.P3 * (1 - C)

    def compute_plan_5(self, Tref=288.15, Pref=101325, rend_turb=0.9):
        """
        Compute plan 5, that is the plan at the exit of the HP Turbine. The turbine efficiency is linked to the compressor efficiency so that
        the absolute difference in temperature is the same during the decompression in the turbine and during the compression in the compressor.
        In the design engine, and since the reduce mass air flow at the exit of the combustion chamber is fixed by design, we can now backpropagate the mass air flow
        through plan 5 to 0.


        Parameters:
            Tref : float, reference Temperature in order to reduce the air mass flow and decontextualize.
            Pref : float, reference Pressure in order to reduce the air mass flow and decontextualize.
            rend_turb : float, The efficiency of the turbine that helps to calculate the temperature decrease when decompressing

        Output:
            None
        """
        W4R = 2  # Fixed by design

        self.W4 = (
            W4R * (self.P4 / Pref) / np.sqrt(self.T4 / Tref)
        )  # Compute mass air flow from reduced mass air flow, T4 and P4

        self.retro_W()  # retropropagate the mass air flow from here to plan 0 so that we have the real values of W0

        # Then compute mass air flow, temperature and pressure as in the equations in the Supelec course.
        self.W5 = self.W4
        self.T5 = self.T4 - (self.T3 - self.T2)

        self.T5is = (self.T4 * (rend_turb - 1) + self.T5) / rend_turb
        self.P5 = self.P4 * (self.T5is / self.T4) ** (self.gamma / (self.gamma - 1))

        ### Output :
        self.A4star = 2 / 241.261

    def retro_W(self, Tref=288.15, Pref=101325):
        """
        backpropagate the mass air flow through plan 5 to 0.


        Parameters:
            Tref : float, reference Temperature in order to reduce the air mass flow and decontextualize.
            Pref : float, reference Pressure in order to reduce the air mass flow and decontextualize.

        Output:
            None
        """

        self.W3 = self.W4
        self.W2 = self.W3
        self.W2R = self.W2 * np.sqrt(self.T2 / Tref) / (self.P2 / Pref)
        self.W1 = self.W2
        self.W0 = self.W1

        self.Wf = (
            self.W3
            * self.cp
            * (self.T4 - self.T3)
            / ((self.PCI) * (1 - self.cp * (self.T4 - self.T3) / (self.PCI)))
        )

    def compute_plan_8(self):
        """
        Compute plan 8, that is the plan at the exit of the nozzle. Several quantities other than temperature and pressure are computed as this step.
        Such as Ts8, v8 and F (see Supelec course)


        Parameters:
            None

        Output:
            None
        """
        self.W8 = self.W5
        self.T8 = self.T5
        self.P8 = self.P5

        Ps8 = self.Pamb
        self.M8 = np.sqrt(
            ((self.P8 / Ps8) ** ((self.gamma - 1) / self.gamma) - 1)
            * 2
            / (self.gamma - 1)
        )

        Ts8 = 1 / (self.T8 * (1 + (self.gamma - 1) / 2 * self.M8**2))
        self.v8 = self.M8 * np.sqrt(self.gamma * self.R * Ts8)
        self.F = self.v8 * self.W8

        ###output
        self.A8star = (
            (self.W8 * np.sqrt(self.T8) / self.P8)
            * np.sqrt(self.R / self.gamma)
            * (2 / (self.gamma + 1)) ** (-(self.gamma + 1) / (2 * (self.gamma - 1)))
        )
        self.W8R = self.A8star * 241.261
        self.A8 = (
            self.A8star
            * (1 / self.M8)
            * ((2 / (self.gamma + 1)) * (1 + self.M8**2 * (self.gamma - 1) / 2))
            ** ((self.gamma + 1) / 2 * (self.gamma - 1))
        )

    def pass_forward(self, verbose=0):
        """Pass forward the module of the engine.

        Args:
            verbose (int, optional): _description_. Defaults to 0.
        """
        self.compute_plan_0()
        self.compute_plan_1()
        self.compute_plan_2()
        self.compute_plan_3()
        self.compute_plan_4()
        self.compute_plan_5()
        self.compute_plan_8()
        if verbose > 0:
            self.print_parameters()


class SimpleCorpsSimpleFlux_degrad_n_ratio_beta(TurboReactor):
    """A subclass of Turboreactor.
    Fixed power engine (N) and mass air flow at the exit of the combustion chamber. Map compressor is used. Degrad parameters for compressor, combustion chamber and turbine are possible.

    Special Attributes::
        TurboReactor (_type_): _description_

    Returns:
        None
    """

    def __init__(
        self,
        name="Engine",
        N=1,
        M8=1.54,
        hyp_tuy=241,
        A8=0.0212,
        A8star=0.0172,
        degrad_comp=1,
        degrad_comb=1,
        degrad_turb=1,
        rend_comb=0.95,
        rend_turb=0.9,
        W4R_design=8,
        W3R_design=8,
        ratio_design=860,
        file_map_comp=files("performancemodel.data").joinpath("Ge10stg.txt"),
    ):
        """Initialize an instance

        Args:
            name (str, optional): Name of the engine. Defaults to "Engine".
            N (int, optional): Power engine. Defaults to 1.
            M8 (float, optional): Speed of the air flow at the exit of the nozzle. Defaults to 1.54. In term of number of Mach
            hyp_tuy (int, optional): Constant concerning the airflow at the exit of the nozzle. Defaults to 241.
            A8 (float, optional): Constant concerning the airflow at the exit of the nozzle. Defaults to 0.0212.
            A8star (float, optional): Constant concerning the airflow at the exit of the nozzle. Defaults to 0.0172.
            degrad_comp (int, optional): Coefficient of degradation of the compressor. Defaults to 1.
            degrad_comb (int, optional): Coefficient of degradation of the combustion chamber. Defaults to 1.
            degrad_turb (int, optional): Coefficient of degradation of the HP turbine. Defaults to 1.
            rend_comb (float, optional): Efficiency of the combustion chamber. Defaults to 0.95.
            rend_turb (float, optional): Efficiency of the turbine. Defaults to 0.9.
            W4R_design (int, optional): Reduced mas air flow at the exit of the combustion chamber. Defaults to 8. kg/s
            file_map_comp (str, optional): Path to the file with the compressor map. Defaults to "./data/Ge10stg.txt".

        Returns:
            None
        """

        # W4R_design, W3R_design, ratio_design

        if not 0.5 <= N <= 1.05:
            raise ValueError(
                f"N value is too low or too high. N must be between 0.5 and 1.05"
            )

        if not 0.95 <= degrad_comp <= 1.05:
            raise ValueError(
                f"degrad_comp too low or too high. It must be between 0.95 and 1.05"
            )
        if not 0.95 <= degrad_comb <= 1.05:
            raise ValueError(
                f"degrad_comb too low or too high. It must be between 0.95 and 1.05"
            )
        
        if not 0.95 <= degrad_turb <= 1.05:
            raise ValueError(
                f"degrad_turb too low or too high. It must be between 0.95 and 1.05"
            )

        if not 2 <= W4R_design <= 15:
            raise ValueError(
                f"W4R_design too low or too high. It must be between 7 and 9"
            )

        if not 2 <= W3R_design <= 8:
            raise ValueError(
                f"W3R_design too low or too high. It must be between 2 and 15"
            )

        if not 800 <= ratio_design <= 1500:
            raise ValueError(
                f"ratio_design too low or too high. It must be between 0 and 2500"
            )

        TurboReactor.__init__(self, name)

        self.N = N
        self.W4R = W4R_design
        self.W3R = W3R_design
        self.ratio_design = ratio_design


        self.A8 = A8
        self.A8star = A8star
        self.W8R = hyp_tuy * self.A8star
        self.M8 = M8

        self.degrad_comp = degrad_comp
        self.degrad_comb = degrad_comb
        self.degrad_turb = degrad_turb

        self.rend_comb = rend_comb
        self.rend_turb = rend_turb

        self.file_map_comp = file_map_comp
        self.data = create_data_beta(self.file_map_comp, self.N, self.degrad_comp)



    def compute_plan_3(
        self, map_compressor: Map = MapCompressor(), beta=0.5, Tref=288.15, Pref=101325
    ):
        """Compute plan 3, that is the plan at the exit of the compressor. In this case, we use a compressor map and take into account the degradation coefficient applied to the compressor.

        Args:
            map_compressor : Map Class object, the compressor map that will be use to compute Pressure Ratio and Efficiency based on engine power and air mass flow
            Tref (float, optional): Reference Air Temperature. Defaults to 288.15.
            Pref (int, optional): Reference Air Pressure. Defaults to 101325.

        Returns:
            None
        """
        ## Think about how to incorpore the self.degrad_comp into the map_compressor. Work with operating line ?
        eff, pr, mf = map_compressor.eval_beta(
            self.data, [self.N * self.degrad_comp, beta]
        )

        

        real_rend = eff
        self.W2R = mf

        #self.W2R = self.W2 * np.sqrt(self.T2 / Tref) / (self.P2 / Pref) 
        self.W2 = self.W2R * (self.P2 / Pref)/ np.sqrt(self.T2 / Tref) 

        self.eff = eff
        self.pr = pr

        self.P3 = self.P2 * pr
        self.T3 = self.T2 * (
            1 + ((self.P3 / self.P2) ** ((self.gamma - 1) / self.gamma) - 1) / real_rend
        )

        self.W3 = self.W2
        self.W1 = self.W2
        self.W0 = self.W2

        


    def compute_plan_4(self, Wf=0.5, Tref=288.15, Pref=101325):
        """Compute plan 4, that is the plan at the exit of the combustion chamber.

        Args:
            Tref (float, optional): Reference Air Temperature. Defaults to 288.15.
            Pref (int, optional): Reference Air Pressure. Defaults to 101325.

        Returns:
                None
        """

        self.P4 = self.rend_comb * self.P3
        self.W4 = self.W3 + self.rend_comb * Wf
        self.ratio_fuel_air = Wf/self.W4
        
    
        
        self.T4 = self.T3 + self.rend_comb * Wf * self.PCI / (self.cp*self.W4)
        


        


    def compute_plan_5(self, Tref: float = 288.15, Pref: float = 101325):
        """Compute plan 5, that is the plan at the exit of the HP Turbine. The turbine efficiency is linked to the compressor efficiency so that
            the absolute difference in temperature is the same during the decompression in the turbine and during the compression in the compressor.


        Args:
            Tref : float, reference Temperature in order to reduce the air mass flow and decontextualize.
            Pref : float, reference Pressure in order to reduce the air mass flow and decontextualize.

        Returns:
            None
        """
        self.W5 = self.W4

        self.T5 = self.T4 - (self.T3 - self.T2)

        self.T5is = (self.T4 * (self.rend_turb * self.degrad_turb - 1) + self.T5) / (self.rend_turb * self.degrad_turb)
        self.P5 = self.P4 * (self.T5is / self.T4) ** (self.gamma / (self.gamma - 1))
        self.W5R = self.W5 * (np.sqrt(self.T5 / Tref) / (self.P5 / Pref))

    def compute_plan_8(self, Tref=288.15, Pref=101325):
        """Compute plan 8, that is the plan at the exit of the nozzle. Several quantities other than temperature and pressure are computed as this step.
            Such as Ts8, v8 and F (see Supelec course)


        Args:
            Tref (float, optional): Reference Air Temperature. Defaults to 288.15.
            Pref (int, optional): Reference Air Pressure. Defaults to 101325.

        """
        self.W8 = self.W5
        self.T8 = self.T5
        self.P8 = self.P5

        Ps8 = self.Pamb
        self.M8 = np.sqrt(
            ((self.P8 / Ps8) ** ((self.gamma - 1) / self.gamma) - 1)
            * 2
            / (self.gamma - 1)
        )

        Ts8 = 1 / (self.T8 * (1 + (self.gamma - 1) / 2 * self.M8**2))
        self.v8 = self.M8 * np.sqrt(self.gamma * self.R * Ts8)
        self.F = self.v8 * self.W8

        self.W8R = self.W8 * np.sqrt(self.T8 / Tref) / (self.P8 / Pref)


    def calculate_ratios(self, Tref=288.15, Pref=101325):

        self.ratio = self.W8R / self.A8star

        try:
            test = self.ratio_design
        except:
            self.ratio_design = 241.221

        self.delta_ratio = (self.ratio - self.ratio_design)

        self.W4R_computed = self.W4 * np.sqrt(self.T4 / Tref) / (self.P4 / Pref)
        self.delta_W4R = self.W4R - self.W4R_computed

        self.W3R_computed = self.W3 * np.sqrt(self.T3 / Tref) / (self.P3 / Pref)
        self.delta_W3R = self.W3R - self.W3R_computed
        


    def pass_forward(
        self,
        beta: float = 0.5,
        Wf:float = 0.5,
        map_compressor: Map = MapCompressor(),
        Tref: float = 288.15,
        Pref: float = 101325,
        for_opti=False,
    ):
        """A pass forward the entire engine

        Args:
            W0 (int, optional): Mass Air Flow at the entry of the engine. Defaults to 20.
            map_compressor (Map, optional): The type compressor map used. Defaults to MapCompressor().
            Tref (float, optional): Reference Air Temperature. Defaults to 288.15.
            Pref (float, optional): Reference Air Pressure. Defaults to 101325.
            for_opti (bool, optional): Are we using the function for opitmization or for real pass. Defaults to False.

        Returns:
            float: The delta between the expected Reducted Mass Air Flow and the Reducted Mass Air Flow observed at the exit of the engine
        """

        self.compute_plan_0()
        self.compute_plan_1()
        self.compute_plan_2(Tref=Tref, Pref=Pref)

        self.compute_plan_3(map_compressor=map_compressor, beta=beta)

        self.compute_plan_4(Tref=Tref, Pref=Pref, Wf=Wf)

        self.compute_plan_5(Tref=Tref, Pref=Pref)

        self.compute_plan_8()




    def pass_forward_iter(
        self,
        inputs=None, T4_max=1800
    ):
        """An interrupted pass forward the engine from plan 0 to plan 3 in order to get the right value for the fuel flow.

        Args:
            W0 (int, optional): Mass Air Flow entering the engine. Defaults to 20.
            map_compressor (Map, optional): Clas of map used for the compressor. Defaults to MapCompressor().
            Tref (float, optional): Reference Air Temperature. Defaults to 288.15.
            Pref (int, optional): Reference Air Pressure. Defaults to 101325.
            option (str, optional): Are we fixing ratio or W4R. Defaults to "W4R". "ratio" does not work at the moment.
        """
        if inputs is None:
            inputs=[0.5,0.5]
        beta = inputs[0]
        Wf = inputs[1]
        self.compute_plan_0()
        self.compute_plan_1()
        self.compute_plan_2()
        self.compute_plan_3(beta=beta) 
        self.compute_plan_4(Wf=Wf)
        self.compute_plan_5()
        self.compute_plan_8()

        
        self.calculate_ratios()

        '''
        if self.T4 > T4_max:
            return [1e6, 1e6]
        '''
        
        return [self.delta_ratio, self.delta_W4R]

    def objective(self, inputs):
        
        delta_ratio, delta_W4R = self.pass_forward_iter(inputs)
        
        return (delta_ratio/self.ratio_design)**2 + (delta_W4R/self.W4R)**2
        

    def solve_system(self):
        solution = minimize(self.objective, [0.5,0.5], bounds=[(0,1), (0.1,3)], method="L-BFGS-B")
        if solution.success:
            return solution.x
        
        else:
            print("Erreur de convergence :", solution.message)





    def simulator(self):
        """Run a pass at the simulator.

        Args:
            None
        Returns:
            None
        """
        solutions = self.solve_system()
        true_beta = solutions[0]
        true_Wf = solutions[1]
        
        self.Wf = true_Wf
        self.beta = true_beta

        self.pass_forward(beta=true_beta, Wf=true_Wf)
        

        return self.get_measurements()
    





    def get_measurements(self, list_captor=None):
        out = {}
        if list_captor is None:
            list_captor = ["P3", "T4", "W5R"]
        
        for n in list_captor:
            if hasattr(self, n):
                out[n] = getattr(self, n)
            else:
                out[n] = None

        return out
    
    def run(self, list_captor=None):
        
        if list_captor is None:
            list_captor = ["P3", "T4", "W5R"]
        
        self.simulator()
        measurements = self.get_measurements(list_captor)
        
        return measurements



def outside_passforward(x):
        W0 = x[0]
        Wf = x[1]
        gamma=1.4
        PCI=44 * 10**6
        R=287.05
        M0 = 1
        Tamb = 300
        Pamb = 101325
        Tref = 288.15
        Pref = 101325
        N = 0.9
        cp = gamma * R / (gamma - 1)
        file_map_comp = files("performancemodel.data").joinpath("Ge10stg.txt")

        degrad_comp = 1
        degrad_comb = 1
        degrad_turb = 1
        ratio_design = 860
        hyp_tuy = 241
        W4R = 8
        W3R = 4
        rend_comb = 0.95
        rend_turb = 0.9

        verbose = 0


        A8 = 0.0212
        A8star = 0.0172
        W8R = hyp_tuy * A8star
        M8 = 1.54

        data = create_data(file_map_comp, N, degrad_comp)
        map_compressor = MapCompressor()

        ## Plan 0
        T0 = Tamb * (1 + ((gamma - 1) * M0**2) / 2)
        P0 = Pamb * (1 + ((gamma - 1) * M0**2) / 2) ** (
            gamma / (gamma - 1)
        )



        ## Plan 1

        T1 = T0
        P1 = P0
        W1 = W0

        ## Plan 2

        T2 = T1
        P2 = P1
        W2 = W1
        W2R = W2 * (np.sqrt(T2 / Tref) / (P2 / Pref))

        ## Plan 3

        ## Think about how to incorpore the self.degrad_comp into the map_compressor. Work with operating line ?
        eff, pr = map_compressor.eval_SRA(
            data, [N * degrad_comp, W2R]
        )
        # real_rend = eff*degrad_comp
        real_rend = eff

        if np.isnan([eff, pr]).any():
            if verbose:
                print("Impossible value for W0 and N")
            min_W2R, max_W2R = get_minmax(
            data, N, degrad_comp, P2, T2, verbose
        )
            if W2R < min_W2R:
                if verbose:
                    print("Mass flow too low, setting it to minimum allowed")
                W2R = min_W2R

            elif W2R > max_W2R:
                if verbose:
                    print("Mass flow too high, setting it to maximum allowed")
                W2R = max_W2R
            eff, pr = map_compressor.eval_SRA(
                data, [N * degrad_comp, W2R]
            )

            real_rend = eff

        eff = eff
        pr = pr

        P3 = P2 * pr
        T3 = T2 * (
            1 + ((P3 / P2) ** ((gamma - 1) / gamma) - 1) / real_rend
        )

        W3 = W2




        ## Plan 4
        P4 = rend_comb * P3
        W4 = W3 + rend_comb * Wf
        T4 = T3 + rend_comb * Wf * PCI / (W4 * cp)


        W4R_computed = W4 * np.sqrt(T4 / Tref) / (P4 / Pref)
        delta_W4R = W4R - W4R_computed

        ## Plan 5
        W5 = W4

        T5 = T4 - (T3 - T2)

        T5is = (T4 * (rend_turb * degrad_turb - 1) + T5) / (rend_turb * degrad_turb)
        P5 = P4 * (T5is / T4) ** (gamma / (gamma - 1))
        W5R = W5 * (np.sqrt(T5 / Tref) / (P5 / Pref))

        ## Plan 8
        W8 = W5
        T8 = T5
        P8 = P5

        Ps8 = Pamb
        M8 = np.sqrt(
            ((P8 / Ps8) ** ((gamma - 1) / gamma) - 1)
            * 2
            / (gamma - 1)
        )

        Ts8 = 1 / (T8 * (1 + (gamma - 1) / 2 * M8**2))
        v8 = M8 * np.sqrt(gamma * R * Ts8)
        F = v8 * W8

        W8R = W8 * np.sqrt(T8 / Tref) / (P8 / Pref)
        ratio = W8R / A8star

            

        delta_ratio = ratio - ratio_design

        return delta_ratio, delta_W4R

