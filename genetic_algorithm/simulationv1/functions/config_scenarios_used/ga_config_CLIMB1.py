# ga_config.py
# Update this file if you have new configuration for the GA


from odsmr.sensors import HPC_Tout, HP_Nmech, HPC_Tin, LPT_Tin, Fuel_flow, HPC_Pout_st, LP_Nmech
from odsmr.predefined_flight_conditions import Cruise_DeckSMR, Takeoff_DeckSMR, Climb1_DeckSMR, Climb2_DeckSMR
from odsmr.generation_functions import decksmr_1forall
from odsmr.constants import ROOT_OPENDECK, STATE_LABELS, STATE_BOUNDS



from dataclasses import dataclass

@dataclass
class GAConfig:
    population_size: int = 150
    n_generations: int = 250
    tournament_rate: float = 0.1
    elitism_rate: float = 0.05
    crossover_rate: float = 0.85
    blx_alpha: float = 0.5
    mutation_rate: float = 0.5
    mutation_strength_initial: float = 0.15
    mutation_decay: float = 0.998
    mutation_strength_min: float = 0.01
    early_stop_generations: int =  40
    early_stop_tolerance: float = 1e-10

    @property
    def elitism_count(self) -> int:
        return max(1, int(self.population_size * self.elitism_rate))

    @property
    def tournament_size(self) -> int:
        return max(1, int(self.population_size * self.tournament_rate))


# ============================================================
# DATA PATHS
# ============================================================
CLEAN_DATA_PATH = "../data/synthetic_data.csv" 

NOISE_COV_PATH = "../data/noise_covariances.pkl"

NOISY_DATA_PATH = "../data/synthetic_data_noisy.csv" 

OUTPUT_BASE_DIR = "../experiments"



# ============================================================
# ALL AVAILABLE CONTEXTS AND SENSORS
# ============================================================
# ALL_CONTEXTS =  ["CRUISE"] #  "TAKEOFF", "CLIMB1", "CLIMB2"]
# ALL_SENSORS = ["HPC_Tin", "LPT_Tin", "HPC_Pout_st"]

# Indicators to estimate (fixed - 3 components)
INDICATORS_TO_ESTIMATE = [
    "deg_CmpFan_s_mapWc_in",
    "deg_CmpH_s_mapEff_in",
    "deg_TrbH_s_mapEff_in",
]

# Context and sensor mappings
CONTEXT_MAP = {
    # "CRUISE": Cruise_DeckSMR,
    # "TAKEOFF": Takeoff_DeckSMR,
    "CLIMB1": Climb1_DeckSMR,
    # "CLIMB2": Climb2_DeckSMR,
}

SENSOR_OBJECTS = {
    "HPC_Tout": HPC_Tout(), 
    "HP_Nmech": HP_Nmech(), 
    "HPC_Tin": HPC_Tin(),
    "LPT_Tin": LPT_Tin(), 
    "Fuel_flow": Fuel_flow(), 
    "HPC_Pout_st": HPC_Pout_st(),
    "LP_Nmech": LP_Nmech(),
}

INDICATOR_SHORT_NAMES = {
    "deg_CmpBst_s_mapEff_in": "Booster Eff", 
    "deg_CmpBst_s_mapWc_in": "Booster Wc",
    "deg_CmpFan_s_mapEff_in": "Fan Eff", 
    "deg_CmpFan_s_mapWc_in": "Fan Wc",
    "deg_CmpH_s_mapEff_in": "HPC Eff", 
    "deg_CmpH_s_mapWc_in": "HPC Wc",
    "deg_TrbH_s_mapEff_in": "HPT Eff", 
    "deg_TrbH_s_mapWc_in": "HPT Wc",
    "deg_TrbL_s_mapEff_in": "LPT Eff", 
    "deg_TrbL_s_mapWc_in": "LPT Wc",
}



##------------------
# Define differents scenarios for the ablation (sensor/context removal) study
# Define scenarios
# CONTEXT_SCENARIOS = [
#         ["CRUISE"],
#         # ["CRUISE", "TAKEOFF"],
#         # ["CRUISE", "TAKEOFF", "CLIMB1"],
#         # ["CRUISE", "TAKEOFF", "CLIMB1", "CLIMB2"],
#     ]
    
# SENSOR_SCENARIOS = [
#         ["HPC_Tin"],
#         ["HPC_Tin", "LPT_Tin"],
#         ["HPC_Tin", "LPT_Tin", "HPC_Pout_st"],
#     ]




# ============================================================
# RECOMMENDED CONFIGURATION
# ============================================================

ALL_CONTEXTS = ["CLIMB1"] #, "TAKEOFF", "CLIMB1", "CLIMB2"]
ALL_SENSORS = ["HPC_Tin", "LPT_Tin", "HPC_Pout_st"]

CONTEXT_MAP = {
    # "CRUISE": Cruise_DeckSMR,
    # "TAKEOFF": Takeoff_DeckSMR,
    "CLIMB1": Climb1_DeckSMR,
    # "CLIMB2": Climb2_DeckSMR,
}

CONTEXT_SCENARIOS = [
    # ["CRUISE"],  
    # ["TAKEOFF"], # 1 ctx - baseline
    ["CLIMB1"],
   # [ "CLIMB2"]
    # ["CRUISE", "TAKEOFF"],                         # 2 ctx - add transient
    # ["CRUISE", "TAKEOFF", "CLIMB1", "CLIMB2"],     # 4 ctx - all
]

SENSOR_SCENARIOS = [
    ["HPC_Tin"],                                   # 1 sens - temperature only
    ["HPC_Tin", "LPT_Tin"],                        # 2 sens - both temperatures
    ["HPC_Tin", "LPT_Tin", "HPC_Pout_st"],         # 3 sens - add pressure
]




# Sensor order in noise covariance matrices (6x6)
NOISE_SENSOR_ORDER = ["HP_Nmech", "HPC_Tout", "HPC_Tin", "LPT_Tin", "Fuel_flow", "HPC_Pout_st"]

# Experiment settings
N_TEST_ROWS = 5 # 10  # Number of rows to test per scenario
ROW_TO_USE = None

INDEX_TO_RUN_ABLATION = 53
