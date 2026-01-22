import numpy as np
import pandas as pd
from scipy.stats import qmc
from typing import List, Dict, Tuple, Optional
import pickle


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



from odsmr.sensors import HPC_Tout, HP_Nmech, HPC_Tin, LPT_Tin, Fuel_flow, HPC_Pout_st, LP_Nmech
from odsmr.context import FlightCondDeckSMR, ContextDeckSMR
from odsmr.predefined_flight_conditions import Cruise_DeckSMR, Takeoff_DeckSMR, Climb1_DeckSMR, Climb2_DeckSMR
from odsmr.generation_functions import decksmr_1forall
from odsmr.constants import ROOT_OPENDECK, STATE_LABELS, STATE_BOUNDS

SENSOR_OBJECTS = {
    "HPC_Tout": HPC_Tout(), 
    "HP_Nmech": HP_Nmech(), 
    "HPC_Tin": HPC_Tin(),
    "LPT_Tin": LPT_Tin(), 
    "Fuel_flow": Fuel_flow(), 
    "HPC_Pout_st": HPC_Pout_st(),
    "LP_Nmech": LP_Nmech(),
}



CONTEXT_TEMPLATE = {
    "CRUISE": Cruise_DeckSMR,
    "TAKEOFF": Takeoff_DeckSMR,
    "CLIMB1": Climb1_DeckSMR,
    "CLIMB2": Climb2_DeckSMR,
}



NOISE_COV_MAP = {
    "CRUISE": "Cruise_noise_covariance",
    "TAKEOFF": "Takeoff_noise_covariance",
    "CLIMB1": "Climb1_noise_covariance",
    "CLIMB2": "Climb2_noise_covariance",
}

NOISE_SENSOR_ORDER = ["HP_Nmech", "HPC_Tout", "HPC_Tin", "LPT_Tin", "Fuel_flow", "HPC_Pout_st"]





def generate_multicontext_synthetic_data_ru(n_samples, flight_contexts, sensor_short_names, seed=42):
    """
    Generate synthetic data with ALL flight contexts that is CONSISTENT with our simulator.
    """
    np.random.seed(seed)
    
    sensors_list = [SENSOR_OBJECTS[s] for s in sensor_short_names]
    
    data = []
    
    # print(f"Generating {n_samples} multi-context synthetic samples...")
    # print(f"Contexts: {flight_contexts}")
    
    for i in range(n_samples):
        # Generate random health indicators
        state = np.array([
            np.random.uniform(STATE_BOUNDS[label][0], STATE_BOUNDS[label][1])
            for label in STATE_LABELS
        ])
        
        row = {'sample_id': i}
        
        # Health indicators
        for j, label in enumerate(STATE_LABELS):
            row[label] = state[j]
        
        # For each context, run simulator and store results
        context_list = [CONTEXT_TEMPLATE[ctx] for ctx in flight_contexts]
        result = decksmr_1forall([state], context_list, sensors_list, ROOT_OPENDECK)
        
        for ctx_idx, ctx in enumerate(flight_contexts):
            # Store sensor measurements
            for s in sensors_list:
                row[f'{ctx}_DECKSMR{s.name}'] = result[s.name].values[ctx_idx]
            
            # Store flight conditions (from template)
            fc = CONTEXT_TEMPLATE[ctx].flight_condition
            row[f'{ctx}_DTAMB'] = fc.DTAMB
            row[f'{ctx}_ALT'] = fc.ALT
            row[f'{ctx}_MACH'] = fc.MACH
            row[f'{ctx}_COMMAND'] = fc.COMMAND
        
        data.append(row)
        
        if (i + 1) % 20 == 0:
            print(f"  Generated {i+1}/{n_samples} samples")
    
    df_synthetic = pd.DataFrame(data)
    print(f"\n✓ Generated {len(df_synthetic)} consistent multi-context samples")
    
    return df_synthetic





def generate_multicontext_synthetic_data_lhs(
    n_samples,
    flight_contexts,
    sensor_short_names,
    seed=42
):
    """
    Generate synthetic data with ALL flight contexts using
    Latin Hypercube Sampling (SciPy) for health indicators.
    """
    rng = np.random.default_rng(seed)

    sensors_list = [SENSOR_OBJECTS[s] for s in sensor_short_names]

    n_states = len(STATE_LABELS)

    # print(f"Generating {n_samples} multi-context synthetic samples (LHS)")
    # print(f"Contexts: {flight_contexts}")

    # --------------------------------------------------
    # 1. Latin Hypercube in [0,1]^d
    # --------------------------------------------------
    sampler = qmc.LatinHypercube(d=n_states, seed=seed)
    lhs_unit = sampler.random(n_samples)

    # --------------------------------------------------
    # 2. Scale to physical bounds
    # --------------------------------------------------
    lower = np.array([STATE_BOUNDS[l][0] for l in STATE_LABELS])
    upper = np.array([STATE_BOUNDS[l][1] for l in STATE_LABELS])

    lhs_states = qmc.scale(lhs_unit, lower, upper)

    data = []

    # --------------------------------------------------
    # 3. Simulation loop
    # --------------------------------------------------
    for i in range(n_samples):
        state = lhs_states[i]

        row = {'sample_id': i}

        # Store health indicators
        for j, label in enumerate(STATE_LABELS):
            row[label] = state[j]

        # Build simulator contexts
        context_list = [CONTEXT_TEMPLATE[ctx] for ctx in flight_contexts]

        result = decksmr_1forall(
            [state],
            context_list,
            sensors_list,
            ROOT_OPENDECK
        )

        for ctx_idx, ctx in enumerate(flight_contexts):
            # Sensor outputs
            for s in sensors_list:
                row[f'{ctx}_DECKSMR{s.name}'] = result[s.name].values[ctx_idx]

            # Flight conditions (copied for completeness)
            fc = CONTEXT_TEMPLATE[ctx].flight_condition
            row[f'{ctx}_DTAMB'] = fc.DTAMB
            row[f'{ctx}_ALT'] = fc.ALT
            row[f'{ctx}_MACH'] = fc.MACH
            row[f'{ctx}_COMMAND'] = fc.COMMAND

        data.append(row)

        if (i + 1) % 20 == 0 or i == n_samples - 1:
            print(f"  Generated {i+1}/{n_samples} samples")

    df_synthetic = pd.DataFrame(data)

    print(f"\n✓ Generated {len(df_synthetic)} consistent multi-context samples")

    return df_synthetic



NOISE_COV_PATH = "../data/noise_covariances.pkl"
with open(NOISE_COV_PATH, 'rb') as f:
    NOISE_COVARIANCES = pickle.load(f)



def add_noise_to_data(df_clean: pd.DataFrame, contexts: List[str], sensors: List[str], seed: int = 42) -> pd.DataFrame:
    """
    Add realistic measurement noise to clean data.
    
    Uses the noise covariance matrices to sample from a zero-mean Gaussian distribution.
    For each row and each context, samples noise and adds to corresponding sensor measurements.
    """
    np.random.seed(seed)
    df_noisy = df_clean.copy()
    
    for i in range(len(df_noisy)):
        for ctx in contexts:
            # Get noise covariance for this context
            cov_key = NOISE_COV_MAP[ctx]
            cov_matrix = NOISE_COVARIANCES[cov_key]
            
            # Sample noise for all 6 sensors (noise cov is 6x6)
            noise_sample = np.random.multivariate_normal(
                mean=np.zeros(6),
                cov=cov_matrix
            )
            
            # Apply noise only to sensors we're using
            for s_idx, s_name in enumerate(NOISE_SENSOR_ORDER):
                if s_name in sensors:
                    col = f'{ctx}_DECKSMR{s_name}'
                    if col in df_noisy.columns:
                        df_noisy.loc[i, col] += noise_sample[s_idx]
    
    return df_noisy


#-----------------------
# Default
INDICATORS_TO_ESTIMATE = [
    "deg_CmpFan_s_mapWc_in",   # Fan mass flow
    "deg_CmpH_s_mapEff_in",    # HPC efficiency
    "deg_TrbH_s_mapEff_in",    # HPT efficiency
]
def scatter_each_column(df, column_to_estimate = INDICATORS_TO_ESTIMATE ):
    """
    Scatter plot of each column in df (value vs index).
    """
    for col in column_to_estimate:
        plt.figure(figsize=(5, 3))
        plt.scatter(df.index, df[col], alpha=0.7)
        plt.title(col)
        # plt.xlabel("Index")
        # plt.ylabel("Value")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()




def plot_3d_df(df, x_col, y_col, z_col):
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(df[x_col], df[y_col], df[z_col], alpha=0.7)

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_zlabel(z_col)
    ax.set_title("3D Scatter Plot")

    plt.tight_layout()
    plt.show()


    # 3D plot
def plot_3d_first_three(df, cols):
    assert len(cols) >= 3, "Need at least 3 columns"
    plot_3d_df(df, cols[0], cols[1], cols[2])



#----------------------