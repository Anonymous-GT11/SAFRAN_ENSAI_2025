#!/usr/bin/env python
# coding: utf-8

# # GA Experiment Framework - Complete
# 
# ## Systematic Experiments with Multiple Scenarios
# 
# This notebook includes:
# 1. **All original plots** (convergence, 2D/3D search space, ablation)
# 2. **Multi-scenario experiments** (1-4 contexts × 1-3 sensors)
# 3. **Clean and noisy data** comparison
# 4. **Uses your existing CSV file** and adds noise version

# In[295]:


import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path().resolve().parent
sys.path.append(str(project_root))


# In[296]:


project_root 


# In[297]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import time
import os
import pickle
import warnings
from datetime import datetime
import copy

warnings.filterwarnings('ignore')

# Simulator imports
from odsmr.sensors import HPC_Tout, HP_Nmech, HPC_Tin, LPT_Tin, Fuel_flow, HPC_Pout_st, LP_Nmech
from odsmr.predefined_flight_conditions import Cruise_DeckSMR, Takeoff_DeckSMR, Climb1_DeckSMR, Climb2_DeckSMR
from odsmr.generation_functions import decksmr_1forall
from odsmr.constants import ROOT_OPENDECK, STATE_LABELS, STATE_BOUNDS
from scipy.stats import qmc



# Ga config
from functions.ga_config import GAConfig


np.random.seed(42)
print("Imports OK")


# In[298]:


# In[ ]:


# ---
# ## 1. Configuration

# In[299]:


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
ALL_CONTEXTS = ["CRUISE"] #, "TAKEOFF", "CLIMB1", "CLIMB2"]
ALL_SENSORS = ["HPC_Tin", "LPT_Tin", "HPC_Pout_st"]

# Indicators to estimate (fixed - 3 components)
INDICATORS_TO_ESTIMATE = [
    "deg_CmpFan_s_mapWc_in",
    "deg_CmpH_s_mapEff_in",
    "deg_TrbH_s_mapEff_in",
]

# Context and sensor mappings
CONTEXT_MAP = {
    "CRUISE": Cruise_DeckSMR,
    # "TAKEOFF": Takeoff_DeckSMR,
    # "CLIMB1": Climb1_DeckSMR,
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
#Define differents scenarios
# Define scenarios
CONTEXT_SCENARIOS = [
        ["CRUISE"],
        # ["CRUISE", "TAKEOFF"],
        # ["CRUISE", "TAKEOFF", "CLIMB1"],
        # ["CRUISE", "TAKEOFF", "CLIMB1", "CLIMB2"],
    ]

SENSOR_SCENARIOS = [
        ["HPC_Tin"],
        ["HPC_Tin", "LPT_Tin"],
        ["HPC_Tin", "LPT_Tin", "HPC_Pout_st"],
    ]



# Sensor order in noise covariance matrices (6x6)
NOISE_SENSOR_ORDER = ["HP_Nmech", "HPC_Tout", "HPC_Tin", "LPT_Tin", "Fuel_flow", "HPC_Pout_st"]

# Experiment settings
N_TEST_ROWS = 1 # Number of rows to test per scenario

INDEX_TO_RUN_ABLATION = 15


"""
print(f"Contexts available: {ALL_CONTEXTS}")
print(f"Sensors available: {ALL_SENSORS}")
print(f"Indicators to estimate: {[INDICATOR_SHORT_NAMES[i] for i in INDICATORS_TO_ESTIMATE]}")
"""

# In[ ]:





# In[300]:


# # Load noise covariances
# with open(NOISE_COV_PATH, 'rb') as f:
#     NOISE_COVARIANCES = pickle.load(f)

# NOISE_COV_MAP = {
#     "CRUISE": "Cruise_noise_covariance",
#     # "TAKEOFF": "Takeoff_noise_covariance",
#     # "CLIMB1": "Climb1_noise_covariance",
#     # "CLIMB2": "Climb2_noise_covariance",
# }

# print("Noise covariances loaded:")
# for ctx, key in NOISE_COV_MAP.items():
#     cov = NOISE_COVARIANCES[key]
#     std_devs = np.sqrt(np.diag(cov))
#     print(f"  {ctx}: std_devs = {std_devs}")


# In[301]:

"""
# Load clean data
df_clean = pd.read_csv(CLEAN_DATA_PATH)
print(f"Loaded {len(df_clean)} rows from {CLEAN_DATA_PATH}")


# In[ ]:





# In[302]:


# LOAD noisy version of data
df_noisy =pd.read_csv (NOISY_DATA_PATH)
print(f"Loaded {len(df_clean)} rows from {NOISY_DATA_PATH}")


# Show noise effect
print("\nNoise effect sample (row 0, CRUISE):")
for s in ALL_SENSORS:
    col = f'CRUISE_DECKSMR{s}'
    clean_val = df_clean.loc[0, col]
    noisy_val = df_noisy.loc[0, col]
    diff = noisy_val - clean_val
    print(f"  {s}: clean={clean_val:.4f}, noisy={noisy_val:.4f}, noise={diff:.4f}")

"""
# In[ ]:





# ---
# ## 2. GA Configuration and Class

# In[303]:


# @dataclass
# class GAConfig:
#     population_size: int = 5#100
#     n_generations: int = 3 #30 #0
#     tournament_rate: float = 0.1
#     elitism_rate: float = 0.05
#     crossover_rate: float = 0.85
#     blx_alpha: float = 0.5
#     mutation_rate: float = 0.5
#     mutation_strength_initial: float = 0.15
#     mutation_decay: float = 0.998
#     mutation_strength_min: float = 0.01
#     early_stop_generations: int = 2 #30
#     early_stop_tolerance: float = 1e-10

#     @property
#     def elitism_count(self) -> int:
#         return max(1, int(self.population_size * self.elitism_rate))

#     @property
#     def tournament_size(self) -> int:
#         return max(1, int(self.population_size * self.tournament_rate))

config = GAConfig()
print(f"Population: {config.population_size}, Elitism: {config.elitism_count}, Tournament: {config.tournament_size}")




#-----------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def plot_initial_population(
    pop,
    indicator_names,
    indicator_short_names,
    save_path: str = None,
    show: bool = True
):
    """
    Scatter plot of the initial GA population.

    Parameters
    ----------
    pop : np.ndarray
        Shape (n_individuals, n_genes)
    indicator_names : list[str]
        Full indicator names
    indicator_short_names : dict[str -> str]
        Mapping to short labels
    save_path : str, optional
        If provided, saves the figure (e.g. 'init_population.png')
    show : bool
        Whether to display the figure
    """
    n_genes = pop.shape[1]
    short = [indicator_short_names[ind] for ind in indicator_names]

    fig = None

    # ---------------------------
    # 1D
    # ---------------------------
    if n_genes == 1:
        fig = plt.figure(figsize=(6, 4))
        plt.scatter(pop[:, 0], np.zeros(len(pop)), alpha=0.6)
        plt.axvline(0, color='green', ls='--', lw=2, label='Healthy')
        plt.xlabel(short[0])
        plt.title("Initial Population (1D)")
        plt.legend()
        plt.grid(True, alpha=0.3)

    # ---------------------------
    # 2D
    # ---------------------------
    elif n_genes == 2:
        fig = plt.figure(figsize=(6, 6))
        plt.scatter(pop[:, 0], pop[:, 1], alpha=0.6)
        plt.scatter(0, 0, c='green', s=150, marker='*', label='Healthy')
        plt.xlabel(short[0])
        plt.ylabel(short[1])
        plt.title("Initial Population (2D)")
        plt.legend()
        plt.grid(True, alpha=0.3)

    # ---------------------------
    # 3D
    # ---------------------------
    elif n_genes == 3:
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(pop[:, 0], pop[:, 1], pop[:, 2], alpha=0.6)
        ax.scatter(0, 0, 0, c='green', s=200, marker='*', label='Healthy')
        ax.set_xlabel(short[0])
        ax.set_ylabel(short[1])
        ax.set_zlabel(short[2])
        ax.set_title("Initial Population (3D)")
        ax.legend()

    # ---------------------------
    # >3D → pairwise
    # ---------------------------
    else:
        import itertools

        pairs = list(itertools.combinations(range(n_genes), 2))
        n_pairs = len(pairs)
        ncols = min(3, n_pairs)
        nrows = int(np.ceil(n_pairs / ncols))

        fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
        axes = np.atleast_1d(axes).flatten()

        for ax, (i, j) in zip(axes, pairs):
            ax.scatter(pop[:, i], pop[:, j], alpha=0.6)
            ax.scatter(0, 0, c='green', s=100, marker='*')
            ax.set_xlabel(short[i])
            ax.set_ylabel(short[j])
            ax.grid(True, alpha=0.3)

        for ax in axes[len(pairs):]:
            ax.axis("off")

        fig.suptitle("Initial Population (Pairwise Scatter)", fontweight="bold")
        plt.tight_layout()

    # ---------------------------
    # Save / show / close
    # ---------------------------
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)





    

# In[304]:


class GeneticAlgorithm:
    """GA for health indicator estimation."""

    def __init__(self, config: GAConfig,
                 target_measurements: Dict[str, Dict[str, float]],
                 normalization_factors: Dict[str, Dict[str, float]],
                 data_row: pd.Series,
                 indicators_to_estimate: List[str],
                 flight_contexts: List[str],
                 sensor_short_names: List[str],
                 context_map: Dict,
                 sensor_objects: Dict):

        self.config = config
        self.target_measurements = target_measurements
        self.normalization_factors = normalization_factors
        self.data_row = data_row
        self.indicators_to_estimate = indicators_to_estimate
        self.flight_contexts = flight_contexts
        self.sensor_short_names = sensor_short_names
        self.context_map = context_map
        self.sensor_objects = sensor_objects

        self.sensors_list = [sensor_objects[s] for s in sensor_short_names]

        self.n_genes = len(indicators_to_estimate)
        self.lower = np.array([STATE_BOUNDS[ind][0] for ind in indicators_to_estimate])
        self.upper = np.array([STATE_BOUNDS[ind][1] for ind in indicators_to_estimate])
        self.range = self.upper - self.lower

        self.mutation_strength = config.mutation_strength_initial
        self.cache = {}
        self.sim_calls = 0
        self.history = {'best_fitness': [], 'mean_fitness': [], 'best_individual': [],
                        'mutation_strength': [], 'diversity': []}
        self.initial_population = None
        self.final_population = None
        self.final_fitness = None

    def _create_state_vector(self, individual: np.ndarray) -> np.ndarray:
        state = np.zeros(10)
        for i, label in enumerate(STATE_LABELS):
            if label in self.indicators_to_estimate:
                idx = self.indicators_to_estimate.index(label)
                state[i] = individual[idx]
            else:
                state[i] = self.data_row[label]
        return state

    def _simulate_all_contexts(self, individual: np.ndarray) -> Dict[str, Dict[str, float]]:
        key = tuple(np.round(individual, 10))
        if key in self.cache:
            return self.cache[key]

        state = self._create_state_vector(individual)
        simulator_contexts = []

        for ctx_name in self.flight_contexts:
            ctx = copy.deepcopy(self.context_map[ctx_name])
            prefix = f"{ctx_name}_"

            fc = ctx.flight_condition
            fc.DTAMB = float(self.data_row[prefix + "DTAMB"])
            fc.ALT = float(self.data_row[prefix + "ALT"])
            fc.MACH = float(self.data_row[prefix + "MACH"])
            fc.COMMAND = float(self.data_row[prefix + "COMMAND"])
            simulator_contexts.append(ctx)

        result = decksmr_1forall([state], simulator_contexts, self.sensors_list, ROOT_OPENDECK)
        self.sim_calls += 1

        predictions = {}
        for i, ctx_name in enumerate(self.flight_contexts):
            predictions[ctx_name] = {s.name: result[s.name].values[i] for s in self.sensors_list}

        self.cache[key] = predictions
        return predictions

    def _fitness(self, individual: np.ndarray) -> float:
        predicted = self._simulate_all_contexts(individual)
        sq_sum = 0.0
        for context in self.flight_contexts:
            for sensor in self.sensor_short_names:
                pred_val = predicted[context][sensor]
                target_val = self.target_measurements[context][sensor]
                norm = self.normalization_factors[context][sensor]
                sq_sum += ((pred_val - target_val) / norm) ** 2
        return np.sqrt(sq_sum / (len(self.flight_contexts) * len(self.sensor_short_names)))

    # def _init_population(self) -> np.ndarray:
    #     pop = np.random.uniform(self.lower, self.upper, (self.config.population_size, self.n_genes))
    #     pop[0] = np.zeros(self.n_genes)  # Healthy baseline
    #     self.initial_population = pop.copy()
    #     return pop

    def _init_population(self) -> np.ndarray:
        """
        Initialize GA population using Latin Hypercube Sampling (LHS)
        for better space-filling properties.
        """

        sampler = qmc.LatinHypercube(d=self.n_genes)

        # Generate samples in [0, 1]
        sample_unit = sampler.random(n=self.config.population_size)

        # Scale to parameter bounds
        pop = qmc.scale(sample_unit, self.lower, self.upper)

        # Force first individual to be the reference (all zeros)
        pop[0] = np.zeros(self.n_genes)
        self.initial_population = pop.copy()

        return pop


    def _tournament(self, pop: np.ndarray, fitness: np.ndarray) -> np.ndarray:
        idx = np.random.choice(len(pop), self.config.tournament_size, replace=False)
        return pop[idx[np.argmin(fitness[idx])]].copy()
        

    def _blx_crossover(self, p1: np.ndarray, p2: np.ndarray):
        if np.random.random() > self.config.crossover_rate:
            return p1.copy(), p2.copy()
        alpha = self.config.blx_alpha
        c1, c2 = np.zeros(self.n_genes), np.zeros(self.n_genes)
        for i in range(self.n_genes):
            d = abs(p1[i] - p2[i])
            lo = max(min(p1[i], p2[i]) - alpha * d, self.lower[i])
            hi = min(max(p1[i], p2[i]) + alpha * d, self.upper[i])
            c1[i] = np.random.uniform(lo, hi)
            c2[i] = np.random.uniform(lo, hi)
        return c1, c2

    def _mutate(self, ind: np.ndarray) -> np.ndarray:
        mut = ind.copy()
        for i in range(self.n_genes):
            if np.random.random() < self.config.mutation_rate:
                sigma = self.mutation_strength * self.range[i]
                mut[i] = np.clip(mut[i] + np.random.normal(0, sigma), self.lower[i], self.upper[i])
        return mut

    def _diversity(self, pop: np.ndarray) -> float:
        normalized = (pop - self.lower) / self.range
        return np.mean(np.std(normalized, axis=0))

    #----------------------------------------------------------------------------------
    def get_multiple_solutions(self, threshold_factor: float = 10) -> list:
        """
        Return multiple near-optimal solutions from the final population.
        """
        if self.final_population is None:
            return []
    
        if threshold_factor is None:
            threshold_factor = self.config.solution_threshold_factor
    
        best_fitness = np.min(self.final_fitness)
        threshold = best_fitness * threshold_factor
    
        solutions = [
            (ind.copy(), fit)
            for ind, fit in zip(self.final_population, self.final_fitness)
            if fit <= threshold
        ]
    
        solutions.sort(key=lambda x: x[1])  # best first
        return solutions
    
    #----------------------------------------------------------------------------------------------------------
    def export_multiple_solutions_df(self, row_index: int, threshold_factor : float=2) -> pd.DataFrame:
        """
        Export near-optimal solutions with row_index for saving.
        """
        thresh_factor = threshold_factor
        solutions = self.get_multiple_solutions(threshold_factor = thresh_factor )
    
        records = []
    
        for sol_id, (ind, fit) in enumerate(solutions):
            record = {
                "row_index": row_index,
                "solution_id": sol_id,
                "fitness": fit,
            }
    
            # Store estimated indicators
            for name, value in zip(self.indicators_to_estimate, ind):
                record[name] = value
    
            records.append(record)
    
        return pd.DataFrame(records)
    
    def evolve(self, verbose: bool = False):
        pop = self._init_population()
        # save_path = 
        # Plot initial plots
        plot_initial_population(
            pop,
            self.indicators_to_estimate,
            INDICATOR_SHORT_NAMES,
            save_path= f"../init_population/init_population_row_{self.data_row.name}.png",
            show=False
        )


        fitness = np.array([self._fitness(ind) for ind in pop])

        best_idx = np.argmin(fitness)
        best_sol = pop[best_idx].copy()
        best_fit = fitness[best_idx]

        no_improve = 0
        prev_best = best_fit

        if verbose:
            print(f"{'Gen':>4} {'Best':>12} {'Mean':>12} {'σ':>10} {'Div':>8}")
            print("-" * 50)

        for gen in range(self.config.n_generations):
            self.mutation_strength = max(
                self.config.mutation_strength_initial * (self.config.mutation_decay ** gen),
                self.config.mutation_strength_min
            )

            new_pop = []
            elite_idx = np.argsort(fitness)[:self.config.elitism_count]

            for idx in elite_idx:
                new_pop.append(pop[idx].copy())

            while len(new_pop) < self.config.population_size:
                p1 = self._tournament(pop, fitness)
                p2 = self._tournament(pop, fitness)
                c1, c2 = self._blx_crossover(p1, p2)
                new_pop.extend([self._mutate(c1), self._mutate(c2)])

            pop = np.array(new_pop[:self.config.population_size])
            fitness = np.array([self._fitness(ind) for ind in pop])
            diversity = self._diversity(pop)

            cur_best_idx = np.argmin(fitness)
            if fitness[cur_best_idx] < best_fit:
                best_fit = fitness[cur_best_idx]
                best_sol = pop[cur_best_idx].copy()

            self.history['best_fitness'].append(best_fit)
            self.history['mean_fitness'].append(np.mean(fitness))
            self.history['best_individual'].append(best_sol.copy())
            self.history['mutation_strength'].append(self.mutation_strength)
            self.history['diversity'].append(diversity)

            if prev_best - best_fit < self.config.early_stop_tolerance:
                no_improve += 1
            else:
                no_improve = 0
            prev_best = best_fit

            if verbose and (gen % 25 == 0 or gen == self.config.n_generations - 1):
                print(f"{gen:4d} {best_fit:12.2e} {np.mean(fitness):12.2e} "
                      f"{self.mutation_strength:10.2e} {diversity:8.4f}")

            if no_improve >= self.config.early_stop_generations:
                if verbose:
                    print(f"Early stop at gen {gen}")
                break

        self.final_population = pop.copy()
        self.final_fitness = fitness.copy()


        return best_sol, best_fit, self.history

print("GeneticAlgorithm class defined.")


# ---
# ## 3. Noise Addition Function

# In[305]:




# ---
# ## 4. Plotting Functions (from your original notebook)

# In[307]:


def plot_convergence(history: Dict, true_indicators: Dict, config: GAConfig, 
                     save_path: str = None, title_suffix: str = ""):
    """
    Plot convergence history (6 subplots).
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    gens = range(len(history['best_fitness']))
    n_ind = len(INDICATORS_TO_ESTIMATE)
    colors = plt.cm.tab10(np.linspace(0, 1, n_ind))

    best_inds = np.array(history['best_individual'])
    short_names = [INDICATOR_SHORT_NAMES[ind] for ind in INDICATORS_TO_ESTIMATE]

    # 1. Fitness
    axes[0,0].semilogy(gens, history['best_fitness'], 'b-', lw=2, label='Best')
    axes[0,0].semilogy(gens, history['mean_fitness'], 'r--', alpha=0.7, label='Mean')
    axes[0,0].set_xlabel('Generation'); axes[0,0].set_ylabel('Fitness')
    axes[0,0].set_title('Fitness Evolution'); axes[0,0].legend(); axes[0,0].grid(True, alpha=0.3)

    # 2. Mutation Decay
    axes[0,1].semilogy(gens, history['mutation_strength'], 'g-', lw=2)
    axes[0,1].axhline(config.mutation_strength_min, color='r', ls='--', label='Min')
    axes[0,1].set_xlabel('Generation'); axes[0,1].set_ylabel('Mutation σ')
    axes[0,1].set_title('Adaptive Mutation Decay'); axes[0,1].legend(); axes[0,1].grid(True, alpha=0.3)

    # 3. Diversity
    axes[0,2].plot(gens, history['diversity'], 'm-', lw=2)
    axes[0,2].axhline(0.1, color='orange', ls='--', label='Warning')
    axes[0,2].axhline(0.01, color='red', ls='--', label='Critical')
    axes[0,2].set_xlabel('Generation'); axes[0,2].set_ylabel('Diversity')
    axes[0,2].set_title('Population Diversity'); axes[0,2].legend(); axes[0,2].grid(True, alpha=0.3)

    # 4. Convergence per indicator
    for i, (ind, c) in enumerate(zip(INDICATORS_TO_ESTIMATE, colors)):
        axes[1,0].plot(gens, best_inds[:, i], color=c, lw=2, label=short_names[i])
        axes[1,0].axhline(true_indicators[ind], color=c, ls='--', alpha=0.5)
    axes[1,0].set_xlabel('Generation'); axes[1,0].set_ylabel('Value')
    axes[1,0].set_title('Convergence (dashed=true)'); axes[1,0].legend(); axes[1,0].grid(True, alpha=0.3)

    # # 5. Error convergence
    # error_hist = np.array([[abs(best_inds[g, i] - true_indicators[INDICATORS_TO_ESTIMATE[i]])
    #                         for i in range(n_ind)] for g in range(len(gens))])
    # for i, c in enumerate(colors):
    #     axes[1,1].semilogy(gens, error_hist[:, i] + 1e-15, color=c, lw=2, label=short_names[i])
    # axes[1,1].set_xlabel('Generation'); axes[1,1].set_ylabel('Error')
    # axes[1,1].set_title('Error Convergence'); axes[1,1].legend(); axes[1,1].grid(True, alpha=0.3)

    # 6. Final Error Bar
    final_errors = [abs(best_inds[-1, i] - true_indicators[INDICATORS_TO_ESTIMATE[i]]) for i in range(n_ind)]
    axes[1,2].bar(range(n_ind), final_errors, color=colors)
    axes[1,2].set_xticks(range(n_ind)); axes[1,2].set_xticklabels(short_names)
    rmse_final = np.sqrt(np.mean(np.array(final_errors)**2))
    axes[1,2].set_ylabel('Error'); axes[1,2].set_title(f'Final Errors (RMSE={rmse_final:.2e})')
    axes[1,2].grid(True, alpha=0.3, axis='y')

    plt.suptitle(f'Convergence Analysis {title_suffix}', fontweight='bold')

    # Remove the last axis entirely
    fig.delaxes(axes[1,1])
    #---------------------------------------

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_3d_search_trajectory(history: Dict, true_vals: List[float], save_path: str = None, title_suffix: str = ""):
    """
    3D plot of search trajectory (for 3 indicators).
    """

    best_inds = np.array(history['best_individual'])
    short_names = [INDICATOR_SHORT_NAMES[ind] for ind in INDICATORS_TO_ESTIMATE]

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Trajectory colored by generation
    colors = plt.cm.viridis(np.linspace(0, 1, len(best_inds)))
    for i in range(len(best_inds) - 1):
        ax.plot([best_inds[i, 0], best_inds[i+1, 0]],
               [best_inds[i, 1], best_inds[i+1, 1]],
               [best_inds[i, 2], best_inds[i+1, 2]], c=colors[i], alpha=0.5)

    # Points
    scatter = ax.scatter(best_inds[:, 0], best_inds[:, 1], best_inds[:, 2], 
                        c=range(len(best_inds)), cmap='viridis', s=20)

    # Reference points
    ax.scatter([0], [0], [0], c='green', s=300, marker='*', label='Healthy')
    ax.scatter([true_vals[0]], [true_vals[1]], [true_vals[2]], c='red', s=300, marker='X', label='True')
    ax.scatter([best_inds[-1, 0]], [best_inds[-1, 1]], [best_inds[-1, 2]], 
              c='blue', s=300, marker='s', label='Final')

    ax.set_xlabel(short_names[0])
    ax.set_ylabel(short_names[1])
    ax.set_zlabel(short_names[2])
    ax.set_title(f'3D Search Trajectory {title_suffix}')
    ax.legend()

    plt.colorbar(scatter, ax=ax, label='Generation', shrink=0.5)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


#-------------------------------------------------
def plot_3d_search_trajectory_interactive(history: dict, true_vals: np.ndarray, save_path: str = None, title: str = ""):
    """
    Interactive 3D visualization of GA search trajectory using Plotly.
    """
    import plotly.graph_objects as go

    n_ind = len(INDICATORS_TO_ESTIMATE)
    if n_ind != 3:
        print("Interactive 3D plot requires exactly 3 indicators.")
        return

    best_inds = np.array(history["best_individual"])
    gens = np.arange(len(best_inds))
    short_names = [INDICATOR_SHORT_NAMES[ind] for ind in INDICATORS_TO_ESTIMATE]

    fig = go.Figure()

    # Trajectory line
    fig.add_trace(go.Scatter3d(
        x=best_inds[:, 0],
        y=best_inds[:, 1],
        z=best_inds[:, 2],
        mode="lines",
        line=dict(color="rgba(100,100,100,0.5)", width=4),
        name="Trajectory"
    ))

    # Points colored by generation
    fig.add_trace(go.Scatter3d(
        x=best_inds[:, 0],
        y=best_inds[:, 1],
        z=best_inds[:, 2],
        mode="markers",
        marker=dict(
            size=6,
            color=gens,
            colorscale="Viridis",
            colorbar=dict(title="Generation"),
            opacity=0.85
        ),
        text=[f"Gen {g}" for g in gens],
        hoverinfo="text",
        name="Best individual"
    ))

    # Reference points
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode="markers",
        marker=dict(size=12, color="green", symbol="diamond"),
        name="Healthy"
    ))

    fig.add_trace(go.Scatter3d(
        x=[true_vals[0]],
        y=[true_vals[1]],
        z=[true_vals[2]],
        mode="markers",
        marker=dict(size=12, color="red", symbol="x"),
        name="True"
    ))

    fig.add_trace(go.Scatter3d(
        x=[best_inds[-1, 0]],
        y=[best_inds[-1, 1]],
        z=[best_inds[-1, 2]],
        mode="markers",
        marker=dict(size=10, color="blue", symbol="square"),
        name="Final"
    ))

    fig.update_layout(
        title=title if title else "Interactive 3D GA Search Trajectory",
        width=1100,
        height=900,
        scene=dict(
            xaxis_title=short_names[0],
            yaxis_title=short_names[1],
            zaxis_title=short_names[2],
        ),
        margin=dict(l=40, r=40, b=40, t=80),
    )

    if save_path:
        fig.write_html(save_path.replace('.png', '.html'))

    fig.show()





def plot_results_summary(results_df: pd.DataFrame, save_path: str = None, title: str = ""):
    """
    Plot results summary (4 subplots) - using scatter plots.
    """
    n_ind = len(INDICATORS_TO_ESTIMATE)
    short_names = [INDICATOR_SHORT_NAMES[ind] for ind in INDICATORS_TO_ESTIMATE]
    colors = plt.cm.tab10(np.linspace(0, 1, n_ind))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. RMSE distribution (histogram)
    axes[0,0].hist(results_df['rmse'], bins=15, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0,0].axvline(results_df['rmse'].mean(), color='red', linestyle='--', lw=2, 
                      label=f"Mean: {results_df['rmse'].mean():.2e}")
    axes[0,0].set_xscale("log")  # LOG SCALE
    axes[0,0].set_xlabel('RMSE'); axes[0,0].set_ylabel('Count')
    axes[0,0].set_title('RMSE Distribution'); axes[0,0].legend(); axes[0,0].grid(True, alpha=0.3)

    # 2. Fitness distribution (histogram)
    axes[0,1].hist(results_df['best_fitness'], bins=15, color='forestgreen', edgecolor='black', alpha=0.7)
    axes[0,1].axvline(results_df['best_fitness'].mean(), color='red', linestyle='--', lw=2,
                      label=f"Mean: {results_df['best_fitness'].mean():.2e}")
    axes[0,0].set_xscale("log")  #  LOG SCALE

    axes[0,1].set_xlabel('Fitness'); axes[0,1].set_ylabel('Count')
    axes[0,1].set_title('Fitness Distribution'); axes[0,1].legend(); axes[0,1].grid(True, alpha=0.3)

    # 3. Per-indicator error boxplots
    error_data = []
    labels = []
    for i, ind in enumerate(INDICATORS_TO_ESTIMATE):
        short = INDICATOR_SHORT_NAMES[ind]
        if f'err_{short}' in results_df.columns:
            error_data.append(results_df[f'err_{short}'].values)
            labels.append(short)

    bp = axes[1,0].boxplot(error_data, labels=labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors[:len(error_data)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[1,0].set_ylabel('Error')
    axes[1,0].set_title('Per-Indicator Error Distribution')
    axes[1,0].grid(True, alpha=0.3, axis='y')
    axes[1,0].set_yscale('log')

    # 4. True vs Estimated scatter (keep as is - this one is good)
    for i, ind in enumerate(INDICATORS_TO_ESTIMATE):
        short = INDICATOR_SHORT_NAMES[ind]
        if f'true_{short}' in results_df.columns and f'est_{short}' in results_df.columns:
            axes[1,1].scatter(results_df[f'true_{short}'], results_df[f'est_{short}'],
                             c=[colors[i]], alpha=0.7, label=short, s=40)

    all_vals = []
    for ind in INDICATORS_TO_ESTIMATE:
        short = INDICATOR_SHORT_NAMES[ind]
        if f'true_{short}' in results_df.columns:
            all_vals.extend(results_df[f'true_{short}'].tolist())
            all_vals.extend(results_df[f'est_{short}'].tolist())
    if all_vals:
        lim = [min(all_vals) - 0.01, max(all_vals) + 0.01]
        axes[1,1].plot(lim, lim, 'k--', alpha=0.5, label='Perfect')
    axes[1,1].set_xlabel('True'); axes[1,1].set_ylabel('Estimated')
    axes[1,1].set_title('True vs Estimated'); axes[1,1].legend(); axes[1,1].grid(True, alpha=0.3)

    plt.suptitle(title, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

print("Plotting functions defined.")


# ---
# ## 5. Ablation Study Function

# In[308]:


def run_ablation_study(row_index: int, df: pd.DataFrame, config: GAConfig,
                       base_contexts: List[str], base_sensors: List[str]) -> pd.DataFrame:
    """
    Run ablation study: remove each sensor/context and measure impact.
    """
    ablation_results = []
    test_row = df.iloc[row_index]
    true_indicators = {ind: test_row[ind] for ind in INDICATORS_TO_ESTIMATE}

    # Build base measurements and norms
    target_meas_base = {}
    norm_base = {}
    for ctx in base_contexts:
        target_meas_base[ctx] = {}
        norm_base[ctx] = {}
        for short in base_sensors:
            col = f"{ctx}_DECKSMR{short}"
            target_meas_base[ctx][short] = test_row[col]
            std_val = df[col].std()
            norm_base[ctx][short] = std_val if std_val > 1e-10 else abs(df[col].mean())

    # BASELINE
    print("Running BASELINE...")
    ga_base = GeneticAlgorithm(
        config=config, target_measurements=target_meas_base,
        normalization_factors=norm_base, data_row=test_row,
        indicators_to_estimate=INDICATORS_TO_ESTIMATE,
        flight_contexts=base_contexts, sensor_short_names=base_sensors,
        context_map=CONTEXT_MAP, sensor_objects=SENSOR_OBJECTS
    )
    best_sol_base, best_fit_base, _ = ga_base.evolve(verbose=False)

    base_errors = {ind: abs(best_sol_base[i] - true_indicators[ind]) 
                   for i, ind in enumerate(INDICATORS_TO_ESTIMATE)}
    base_rmse = np.sqrt(np.mean([e**2 for e in base_errors.values()]))

    ablation_results.append({
       'row_index':row_index, 'configuration': 'BASELINE', 'removed': 'None', 'type': 'baseline',
        'fitness': best_fit_base, 'rmse': base_rmse,
        **{f'err_{INDICATOR_SHORT_NAMES[ind]}': base_errors[ind] for ind in INDICATORS_TO_ESTIMATE}
    })
    print(f"  RMSE: {base_rmse:.2e}")

    # SENSOR ABLATION
    if len(base_sensors) > 1:
        print("\nSensor ablation:")
        for remove_sensor in base_sensors:
            sub_sensors = [s for s in base_sensors if s != remove_sensor]
            print(f"  Without {remove_sensor}...", end=" ")

            target_meas = {ctx: {s: target_meas_base[ctx][s] for s in sub_sensors} for ctx in base_contexts}
            norm_sub = {ctx: {s: norm_base[ctx][s] for s in sub_sensors} for ctx in base_contexts}

            ga_sub = GeneticAlgorithm(
                config=config, target_measurements=target_meas,
                normalization_factors=norm_sub, data_row=test_row,
                indicators_to_estimate=INDICATORS_TO_ESTIMATE,
                flight_contexts=base_contexts, sensor_short_names=sub_sensors,
                context_map=CONTEXT_MAP, sensor_objects=SENSOR_OBJECTS
            )
            best_sol, best_fit, _ = ga_sub.evolve(verbose=False)

            errors = {ind: abs(best_sol[i] - true_indicators[ind]) for i, ind in enumerate(INDICATORS_TO_ESTIMATE)}
            rmse = np.sqrt(np.mean([e**2 for e in errors.values()]))

            ablation_results.append({
               'row_index':row_index, 'configuration': f'No {remove_sensor}', 'removed': remove_sensor, 'type': 'sensor',
                'fitness': best_fit, 'rmse': rmse,
                **{f'err_{INDICATOR_SHORT_NAMES[ind]}': errors[ind] for ind in INDICATORS_TO_ESTIMATE}
            })

            impact = (rmse - base_rmse) / base_rmse * 100 if base_rmse > 0 else 0
            print(f"RMSE: {rmse:.2e} ({impact:+.1f}%)")

    # CONTEXT ABLATION
    if len(base_contexts) > 1:
        print("\nContext ablation:")
        for remove_ctx in base_contexts:
            sub_contexts = [c for c in base_contexts if c != remove_ctx]
            print(f"  Without {remove_ctx}...", end=" ")

            target_meas = {ctx: target_meas_base[ctx] for ctx in sub_contexts}
            norm_sub = {ctx: norm_base[ctx] for ctx in sub_contexts}

            ga_sub = GeneticAlgorithm(
                config=config, target_measurements=target_meas,
                normalization_factors=norm_sub, data_row=test_row,
                indicators_to_estimate=INDICATORS_TO_ESTIMATE,
                flight_contexts=sub_contexts, sensor_short_names=base_sensors,
                context_map=CONTEXT_MAP, sensor_objects=SENSOR_OBJECTS
            )
            best_sol, best_fit, _ = ga_sub.evolve(verbose=False)

            errors = {ind: abs(best_sol[i] - true_indicators[ind]) for i, ind in enumerate(INDICATORS_TO_ESTIMATE)}
            rmse = np.sqrt(np.mean([e**2 for e in errors.values()]))

            ablation_results.append({
               'row_index':row_index, 'configuration': f'No {remove_ctx}', 'removed': remove_ctx, 'type': 'context',
                'fitness': best_fit, 'rmse': rmse,
                **{f'err_{INDICATOR_SHORT_NAMES[ind]}': errors[ind] for ind in INDICATORS_TO_ESTIMATE}
            })

            impact = (rmse - base_rmse) / base_rmse * 100 if base_rmse > 0 else 0
            print(f"RMSE: {rmse:.2e} ({impact:+.1f}%)")



    

    return pd.DataFrame(ablation_results)


def plot_ablation_results(ablation_df: pd.DataFrame, save_path: str = None, title: str = ""):
    """
    Plot ablation study results.
    """
    n_configs = len(ablation_df)
    n_ind = len(INDICATORS_TO_ESTIMATE)
    short_names = [INDICATOR_SHORT_NAMES[ind] for ind in INDICATORS_TO_ESTIMATE]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    configs = ablation_df['configuration'].tolist()
    rmses = ablation_df['rmse'].tolist()
    base_rmse = rmses[0]
    impacts = [(r - base_rmse) / base_rmse * 100 if base_rmse > 0 else 0 for r in rmses]

    # Color based on impact
    colors = ['green']  # Baseline
    for imp in impacts[1:]:
        if imp > 20:
            colors.append('red')
        elif imp > 5:
            colors.append('orange')
        elif imp < -5:
            colors.append('blue')  # Detrimental
        else:
            colors.append('gray')

    # 1. RMSE comparison
    axes[0,0].bar(range(n_configs), rmses, color=colors)
    axes[0,0].set_xticks(range(n_configs))
    axes[0,0].set_xticklabels(configs, rotation=45, ha='right')
    axes[0,0].set_ylabel('RMSE'); axes[0,0].set_title('RMSE by Configuration')
    axes[0,0].axhline(base_rmse, color='green', ls='--', alpha=0.5)
    axes[0,0].grid(True, alpha=0.3, axis='y')

    # 2. Impact plot
    axes[0,1].bar(range(n_configs), impacts, color=colors)
    axes[0,1].set_xticks(range(n_configs))
    axes[0,1].set_xticklabels(configs, rotation=45, ha='right')
    axes[0,1].set_ylabel('Impact (%)'); axes[0,1].set_title('Impact of Removal')
    axes[0,1].axhline(0, color='black', ls='-', lw=0.5)
    axes[0,1].grid(True, alpha=0.3, axis='y')



    # 3. Error per indicator per configuration
    x = np.arange(n_configs)
    width = 0.8 / n_ind
    ind_colors = plt.cm.tab10(np.linspace(0, 1, n_ind))

    for i, ind in enumerate(INDICATORS_TO_ESTIMATE):
        short = INDICATOR_SHORT_NAMES[ind]
        err_col = f'err_{short}'
        if err_col in ablation_df.columns:
            errs = ablation_df[err_col].tolist()
            axes[1,0].bar(x + i * width - 0.4 + width/2, errs, width, 
                         label=short, color=ind_colors[i])

    axes[1,0].set_xticks(range(n_configs))
    axes[1,0].set_xticklabels(configs, rotation=45, ha='right')
    axes[1,0].set_ylabel('Error'); axes[1,0].set_title('Error per Indicator')
    axes[1,0].legend(); axes[1,0].grid(True, alpha=0.3, axis='y')

    # 4. Interpretation
    axes[1,1].axis('off')
    interpretation_text = "INTERPRETATION:\n\n"
    for i, row in ablation_df.iterrows():
        if i == 0:
            interpretation_text += f"BASELINE: RMSE = {row['rmse']:.2e}\n\n"
            continue

        impact = impacts[i]
        removed = row['removed']

        if impact < -20:
            level = "DETRIMENTAL"
        elif impact < -5:
            level = "WEAKLY DETRIMENTAL"
        elif impact <= 5:
            level = "NEUTRAL"
        elif impact <= 20:
            level = "MODERATE IMPORTANCE"
        else:
            level = "HIGH IMPORTANCE"

        interpretation_text += f"{removed}: {level} ({impact:+.1f}%)\n"

    interpretation_text += "\n\nLegend:\n"
    interpretation_text += "  DETRIMENTAL: Removing HELPS\n"
    interpretation_text += "  NEUTRAL: Little effect\n"
    interpretation_text += "  HIGH IMPORTANCE: Removing HURTS"

    axes[1,1].text(0.1, 0.9, interpretation_text, transform=axes[1,1].transAxes,
                  fontsize=10, verticalalignment='top', fontfamily='monospace',
                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle(f'Ablation Study Results {title}', fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


#===========================

def plot_ablation_resultsv2(ablation_df: pd.DataFrame, save_path: str = None, title: str = "Sensor Ablation Study Results"):
    """
    Plot 2-panel sensor ablation results.

    Left:  Estimation Error by Sensor Configuration
    Right: Error per Indicator and Configuration
    """
    configs = ablation_df['configuration'].tolist()
    n_configs = len(configs)
    short_names = [INDICATOR_SHORT_NAMES[ind] for ind in INDICATORS_TO_ESTIMATE]


    rmses = ablation_df['rmse'].tolist()
    base_rmse = rmses[0]
    impacts = [(r - base_rmse) / base_rmse * 100 if base_rmse > 0 else 0 for r in rmses]

    # Colors: green for baseline, orange, red, purple for ablations
    colors = ['green', 'orange', 'red', 'purple', 'brown', 'pink', 'cyan'][:n_configs]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # === Left plot: Estimation Error by Sensor Configuration ===
    rmse_col = 'mean_rmse' if 'mean_rmse' in ablation_df.columns else 'rmse'
    rmse_vals = ablation_df[rmse_col].values

    bars1 = axes[0,0].bar(configs, rmse_vals, color=colors, edgecolor='none', width=0.6)
    axes[0,0].axhline(base_rmse, color='green', ls='--', alpha=0.5)
    axes[0,0].set_title('Estimation Error by Sensor Configuration', fontsize=12)
    axes[0,0].set_yscale('log')
    axes[0,0].grid(True, alpha=0.3, axis='y')
    axes[0,0].set_axisbelow(True)

    #  # Value labels on bars
    # for bar, val in zip(bars1, rmse_vals):
    #     axes[0,0].text(bar.get_x() + bar.get_width()/2, val * 0.7, f'{val:.1e}',
    #                 ha='center', va='top', fontsize=10, fontweight='bold', color='black')
    # Value labels on bars (log-scale safe positioning)
    for bar, val in zip(bars1, rmse_vals):
        x = bar.get_x() + bar.get_width() / 2

        # place label slightly ABOVE the bar
        y = val * 1   # multiplicative offset works on log scale

        axes[0,0].text(
            x, y,
            f'{val:.1e}',
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold'
        )



    axes[0,0].set_xticklabels(configs, rotation=0, ha='center', fontsize=10)





    # === Right plot: Error per Indicator and Configuration ===
    err_cols = []
    for short in short_names:
        for prefix in ['mean_err_', 'err_']:
            col = f'{prefix}{short}'
            if col in ablation_df.columns:
                err_cols.append(col)
                break

    n_ind = len(short_names)
    x = np.arange(n_ind)
    width = 0.8 / n_configs

    for i, (_, row) in enumerate(ablation_df.iterrows()):
        errors = [row[c] for c in err_cols]
        offset = (i - n_configs/2 + 0.5) * width
        axes[0,1].bar(x + offset, errors, width, label=row['configuration'],
                   color=colors[i], edgecolor='none')

    axes[0,1].set_ylabel('Absolute Error', fontsize=11)
    axes[0,1].set_title('Error per Indicator and Configuration', fontsize=12)
    axes[0,1].set_xticks(x)
    axes[0,1].set_xticklabels(short_names, fontsize=10)
    axes[0,1].set_yscale('log')
    axes[0,1].legend(loc='upper right', fontsize=9)
    axes[0,1].grid(True, alpha=0.3, axis='y')
    axes[0,1].set_axisbelow(True)



    # 3. Impact plot
    axes[1,0].bar(range(n_configs), impacts, color=colors)
    axes[1,0].set_xticks(range(n_configs))
    axes[1,0].set_xticklabels(configs, rotation=45, ha='right')
    axes[1,0].set_ylabel('Impact (%)'); axes[0,1].set_title('Impact of Removal')
    axes[1,0].axhline(0, color='black', ls='-', lw=0.5)
    axes[1,0].grid(True, alpha=0.3, axis='y')




    # 4. Interpretation
    axes[1,1].axis('off')
    interpretation_text = "INTERPRETATION:\n\n"
    for i, row in ablation_df.iterrows():
        if i == 0:
            interpretation_text += f"BASELINE: RMSE = {row['rmse']:.2e}\n\n"
            continue

        impact = impacts[i]
        removed = row['removed']

        if impact < -20:
            level = "DETRIMENTAL"
        elif impact < -5:
            level = "WEAKLY DETRIMENTAL"
        elif impact <= 5:
            level = "NEUTRAL"
        elif impact <= 20:
            level = "MODERATE IMPORTANCE"
        else:
            level = "HIGH IMPORTANCE"

        interpretation_text += f"{removed}: {level} ({impact:+.1f}%)\n"

    interpretation_text += "\n\nLegend:\n"
    interpretation_text += "  DETRIMENTAL: Removing HELPS\n"
    interpretation_text += "  NEUTRAL: Little effect\n"
    interpretation_text += "  HIGH IMPORTANCE: Removing HURTS"

    axes[1,1].text(0.1, 0.9, interpretation_text, transform=axes[1,1].transAxes,
                  fontsize=10, verticalalignment='top', fontfamily='monospace',
                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle(f'Ablation Study Results {title}', fontweight='bold')
    plt.tight_layout()


    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()







# ---
# ## 6. Multi-Scenario Experiment Runner

# In[309]:


def run_scenario_experiment(df: pd.DataFrame, 
                            contexts: List[str], 
                            sensors: List[str],
                            n_test_rows: int,
                            config: GAConfig) -> Tuple[pd.DataFrame, Dict, Dict]:
    """
    Run GA on multiple rows for a given scenario.
    Returns results dataframe and one example history/true_indicators for plotting.
    """
    results = []
    example_history = None
    example_true = None
    multiple_solutions_list = []

    # Compute normalization factors
    norm_factors = {}
    for ctx in contexts:
        norm_factors[ctx] = {}
        for short in sensors:
            col = f"{ctx}_DECKSMR{short}"
            std_val = df[col].std()
            norm_factors[ctx][short] = std_val if std_val > 1e-10 else abs(df[col].mean())

    test_indices = np.random.choice(len(df), min(n_test_rows, len(df)), replace=False)

    for idx_num, idx in enumerate(test_indices):
        test_row = df.iloc[idx]
        true_indicators = {ind: test_row[ind] for ind in INDICATORS_TO_ESTIMATE}

        target_measurements = {}
        for ctx in contexts:
            target_measurements[ctx] = {}
            for short in sensors:
                col = f"{ctx}_DECKSMR{short}"
                target_measurements[ctx][short] = test_row[col]

        ga = GeneticAlgorithm(
            config=config,
            target_measurements=target_measurements,
            normalization_factors=norm_factors,
            data_row=test_row,
            indicators_to_estimate=INDICATORS_TO_ESTIMATE,
            flight_contexts=contexts,
            sensor_short_names=sensors,
            context_map=CONTEXT_MAP,
            sensor_objects=SENSOR_OBJECTS
        )

        start_time = time.time()
        best_solution, best_fitness, history = ga.evolve(verbose=False)
        elapsed = time.time() - start_time

        #  collect multiple solution
        df_multi = ga.export_multiple_solutions_df(row_index=idx, )
        
        multiple_solutions_list.append(df_multi)


        # Save first example for plotting
        if idx_num == 0:
            example_history = history
            example_true = true_indicators

        errors = {ind: abs(best_solution[i] - true_indicators[ind]) 
                  for i, ind in enumerate(INDICATORS_TO_ESTIMATE)}
        rmse = np.sqrt(np.mean([e**2 for e in errors.values()]))

        result = {
            'row_index': idx,
            'best_fitness': best_fitness,
            'rmse': rmse,
            'elapsed_time': elapsed,
            'sim_calls': ga.sim_calls,
            'generations': len(history['best_fitness']),
        }

        for i, ind in enumerate(INDICATORS_TO_ESTIMATE):
            short = INDICATOR_SHORT_NAMES[ind]
            result[f'true_{short}'] = true_indicators[ind]
            result[f'est_{short}'] = best_solution[i]
            result[f'err_{short}'] = errors[ind]


        
        results.append(result)

    # --------------------------------------------------
    # Export all multiple solutions once
    # --------------------------------------------------
    
    df_all_multi = None
    if len(multiple_solutions_list) > 0:
        df_all_multi = pd.concat(multiple_solutions_list, ignore_index=True)
        


    return pd.DataFrame(results), example_history, example_true, df_all_multi

# print("Scenario experiment function defined.")


# ---
# ## 7. Run All Experiments

# In[310]:


def run_all_experiments(df_clean: pd.DataFrame, df_noisy: pd.DataFrame,
                        n_test_rows: int, output_base_dir: str ):
    """
    Run all scenario experiments on clean and noisy data.
    """
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(output_base_dir, f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # # Define scenarios
    # context_scenarios = [
    #     ["CRUISE"],
    #     # ["CRUISE", "TAKEOFF"],
    #     # ["CRUISE", "TAKEOFF", "CLIMB1"],
    #     # ["CRUISE", "TAKEOFF", "CLIMB1", "CLIMB2"],
    # ]

    # sensor_scenarios = [
    #     ["HPC_Tin"],
    #     ["HPC_Tin", "LPT_Tin"],
    #     ["HPC_Tin", "LPT_Tin", "HPC_Pout_st"],
    # ]

    all_summary = []

    for data_type, df in [("clean", df_clean), ("noisy", df_noisy)]:
        data_dir = os.path.join(output_dir, data_type)
        os.makedirs(data_dir, exist_ok=True)

        print(f"\n{'='*70}")
        print(f"Running experiments on {data_type.upper()} data")
        print(f"{'='*70}")

        for contexts in CONTEXT_SCENARIOS:
            n_ctx = len(contexts)

            for sensors in SENSOR_SCENARIOS:
                n_sens = len(sensors)
                n_measurements = n_ctx * n_sens

                scenario_name = f"{n_ctx}ctx_{n_sens}sensors"
                scenario_dir = os.path.join(data_dir, scenario_name)
                os.makedirs(scenario_dir, exist_ok=True)

                print(f"\n  {scenario_name}: {contexts} × {sensors}")
                print(f"    {n_measurements} measurements → {len(INDICATORS_TO_ESTIMATE)} indicators")

                # Run experiment
                results_df, example_history, example_true, dfmultiple_solutions = run_scenario_experiment(
                    df=df,
                    contexts=contexts,
                    sensors=sensors,
                    n_test_rows=n_test_rows,
                    config=config
                )

                #--------------------------
                dfmultiple_solutions.to_csv ( os.path.join (output_dir, "multiple_solutions.csv") , index=False)
                #--------------------------
                

                # Save results CSV
                results_path = os.path.join(scenario_dir, "results.csv")
                results_df.to_csv(results_path, index=False)

                # Plot results summary
                plot_results_summary(
                    results_df, 
                    save_path=os.path.join(scenario_dir, "results_summary.png"),
                    title=f"{data_type.capitalize()} | {n_ctx} ctx × {n_sens} sens"
                )

                # Plot convergence (example)
                if example_history:
                    plot_convergence(
                        example_history, example_true, config,
                        save_path=os.path.join(scenario_dir, "convergence.png"),
                        title_suffix=f"({data_type}, {scenario_name})"
                    )

                    # 3D trajectory
                    true_vals = [example_true[ind] for ind in INDICATORS_TO_ESTIMATE]
                    plot_3d_search_trajectory(
                        example_history, true_vals,
                        save_path=os.path.join(scenario_dir, "trajectory_3d.png"),
                        title_suffix=f"({data_type}, {scenario_name})"
                    )


                    # 3D. interactive plot
                    plot_3d_search_trajectory_interactive(
                        history=example_history,
                        save_path=os.path.join(scenario_dir, "trajectory_3d.png"),
                        true_vals=true_vals,

                        title="GA Search Trajectory"
                    )


                # Run ablation (only for scenarios with >1 context or >1 sensor)
                if n_ctx > 1 or n_sens > 1:
                    print(f"    Running ablation study...")
                    ablation_df = run_ablation_study(
                        row_index= INDEX_TO_RUN_ABLATION , df=df, config=config,
                        base_contexts=contexts, base_sensors=sensors
                    )
                    ablation_df.to_csv(os.path.join(scenario_dir, "ablation.csv"), index=False)
                    plot_ablation_results(
                        ablation_df,
                        save_path=os.path.join(scenario_dir, "ablation_plot.png"),
                        title=f"({data_type}, {scenario_name} | row {INDEX_TO_RUN_ABLATION} )"
                    )

                    plot_ablation_resultsv2(
                        ablation_df,
                        save_path=os.path.join(scenario_dir, "ablation_plotv2.png"),
                        title=f"({data_type}, {scenario_name}  | row {INDEX_TO_RUN_ABLATION})"
                    )
                # Summary stats
                summary = {
                    'data_type': data_type,
                    'n_contexts': n_ctx,
                    'n_sensors': n_sens,
                    'n_measurements': n_measurements,
                    'contexts': str(contexts),
                    'sensors': str(sensors),
                    'mean_rmse': results_df['rmse'].mean(),
                    'std_rmse': results_df['rmse'].std(),
                    'mean_fitness': results_df['best_fitness'].mean(),
                    'std_fitness': results_df['best_fitness'].std(),
                    'mean_time': results_df['elapsed_time'].mean(),
                    'std_time': results_df['elapsed_time'].std(),
                    'total_time': results_df['elapsed_time'].sum(),
                    'mean_sim_calls': results_df['sim_calls'].mean(),
                    'mean_generations': results_df['generations'].mean(),
                }
                all_summary.append(summary)

                print(f"    Mean RMSE: {summary['mean_rmse']:.2e} ± {summary['std_rmse']:.2e}")
                print(f"    Mean Time: {summary['mean_time']:.2f}s ± {summary['std_time']:.2f}s")

    # Save overall summary
    summary_df = pd.DataFrame(all_summary)
    summary_df.to_csv(os.path.join(output_dir, "experiment_summary.csv"), index=False)

    return output_dir, summary_df

    
# print("Main experiment function defined.")


# In[311]:

"""
# Run all experiments
output_dir, summary_df = run_all_experiments(
    df_clean=df_clean,
    df_noisy=df_noisy,
    n_test_rows=N_TEST_ROWS,
    output_base_dir=OUTPUT_BASE_DIR
)
"""

# In[ ]:








# ---
# ## 8. Summary Comparison Plot

# In[312]:


def plot_summary_comparison(summary_df: pd.DataFrame, save_path: str = None):
    """
    Compare clean vs noisy across all scenarios.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    clean_df = summary_df[summary_df['data_type'] == 'clean'].copy()
    noisy_df = summary_df[summary_df['data_type'] == 'noisy'].copy()

    clean_df['scenario'] = clean_df['n_contexts'].astype(str) + 'ctx_' + clean_df['n_sensors'].astype(str) + 'sens'
    noisy_df['scenario'] = noisy_df['n_contexts'].astype(str) + 'ctx_' + noisy_df['n_sensors'].astype(str) + 'sens'

    scenarios = clean_df['scenario'].tolist()
    x = np.arange(len(scenarios))
    width = 0.35

    # 1. RMSE comparison
    ax1 = axes[0, 0]
    ax1.bar(x - width/2, clean_df['mean_rmse'], width, label='Clean', color='green', alpha=0.7,
            yerr=clean_df['std_rmse'], capsize=3)
    ax1.bar(x + width/2, noisy_df['mean_rmse'], width, label='Noisy', color='red', alpha=0.7,
            yerr=noisy_df['std_rmse'], capsize=3)
    ax1.set_ylabel('Mean RMSE')
    ax1.set_title('RMSE: Clean vs Noisy')
    ax1.set_xticks(x)
    ax1.set_xticklabels(scenarios, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # 2. Fitness comparison
    ax2 = axes[0, 1]
    ax2.bar(x - width/2, clean_df['mean_fitness'], width, label='Clean', color='green', alpha=0.7)
    ax2.bar(x + width/2, noisy_df['mean_fitness'], width, label='Noisy', color='red', alpha=0.7)
    ax2.set_ylabel('Mean Fitness')
    ax2.set_title('Fitness: Clean vs Noisy')
    ax2.set_xticks(x)
    ax2.set_xticklabels(scenarios, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    # 3. RMSE by contexts
    ax3 = axes[1, 0]
    for n_sens in clean_df['n_sensors'].unique():  #[1, 2, 3]:
        clean_sub = clean_df[clean_df['n_sensors'] == n_sens]
        noisy_sub = noisy_df[noisy_df['n_sensors'] == n_sens]
        ax3.plot(clean_sub['n_contexts'], clean_sub['mean_rmse'], 'o-', label=f'{n_sens} sens (clean)')
        ax3.plot(noisy_sub['n_contexts'], noisy_sub['mean_rmse'], 's--', label=f'{n_sens} sens (noisy)')
    ax3.set_xlabel('Number of Contexts')
    ax3.set_ylabel('Mean RMSE')
    ax3.set_title('RMSE vs Contexts')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. RMSE by sensors
    ax4 = axes[1, 1]
    for n_ctx in clean_df['n_contexts'].unique():  #[1, 2, 3, 4]:
        clean_sub = clean_df[clean_df['n_contexts'] == n_ctx]
        noisy_sub = noisy_df[noisy_df['n_contexts'] == n_ctx]
        ax4.plot(clean_sub['n_sensors'], clean_sub['mean_rmse'], 'o-', label=f'{n_ctx} ctx (clean)')
        ax4.plot(noisy_sub['n_sensors'], noisy_sub['mean_rmse'], 's--', label=f'{n_ctx} ctx (noisy)')
    ax4.set_xlabel('Number of Sensors')
    ax4.set_ylabel('Mean RMSE')
    ax4.set_title('RMSE vs Sensors')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.suptitle('Experiment Summary: Clean vs Noisy Data', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

# Generate summary plot
"""
plot_summary_comparison(summary_df, save_path=os.path.join(output_dir, "summary_comparison.png"))
"""

# In[ ]:





# In[313]:


def plot_execution_times(summary_df: pd.DataFrame, save_path: str = None):
    """
    Plot execution time analysis across scenarios.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    clean_df = summary_df[summary_df['data_type'] == 'clean'].copy()
    noisy_df = summary_df[summary_df['data_type'] == 'noisy'].copy()

    clean_df['scenario'] = clean_df['n_contexts'].astype(str) + 'ctx_' + clean_df['n_sensors'].astype(str) + 'sens'
    noisy_df['scenario'] = noisy_df['n_contexts'].astype(str) + 'ctx_' + noisy_df['n_sensors'].astype(str) + 'sens'

    scenarios = clean_df['scenario'].tolist()
    x = np.arange(len(scenarios))
    width = 0.35

    # 1. Mean execution time comparison
    ax1 = axes[0, 0]
    ax1.bar(x - width/2, clean_df['mean_time'], width, label='Clean', color='green', alpha=0.7,
            yerr=clean_df['std_time'], capsize=3)
    ax1.bar(x + width/2, noisy_df['mean_time'], width, label='Noisy', color='red', alpha=0.7,
            yerr=noisy_df['std_time'], capsize=3)
    ax1.set_ylabel('Mean Time (s)')
    ax1.set_title('Execution Time: Clean vs Noisy')
    ax1.set_xticks(x)
    ax1.set_xticklabels(scenarios, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # 2. Total time per scenario
    ax2 = axes[0, 1]
    ax2.bar(x - width/2, clean_df['total_time'], width, label='Clean', color='green', alpha=0.7)
    ax2.bar(x + width/2, noisy_df['total_time'], width, label='Noisy', color='red', alpha=0.7)
    ax2.set_ylabel('Total Time (s)')
    ax2.set_title('Total Execution Time per Scenario')
    ax2.set_xticks(x)
    ax2.set_xticklabels(scenarios, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')



    # 3. Time vs number of measurements
    ax3 = axes[1, 0]
    ax3.scatter(clean_df['n_measurements'], clean_df['mean_time'], s=100, c='green', 
                label='Clean', alpha=0.7, marker='o')
    ax3.scatter(noisy_df['n_measurements'], noisy_df['mean_time'], s=100, c='red', 
                label='Noisy', alpha=0.7, marker='s')



    # Add trend lines
    z_clean = np.polyfit(clean_df['n_measurements'], clean_df['mean_time'], 1)
    p_clean = np.poly1d(z_clean)
    x_line = np.linspace(clean_df['n_measurements'].min(), clean_df['n_measurements'].max(), 100)
    ax3.plot(x_line, p_clean(x_line), 'g--', alpha=0.5)

    z_noisy = np.polyfit(noisy_df['n_measurements'], noisy_df['mean_time'], 1)
    p_noisy = np.poly1d(z_noisy)
    ax3.plot(x_line, p_noisy(x_line), 'r--', alpha=0.5)

    ax3.set_xlabel('Number of Measurements')
    ax3.set_ylabel('Mean Time (s)')
    ax3.set_title('Execution Time vs Measurements')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Simulator calls vs time
    ax4 = axes[1, 1]
    ax4.scatter(clean_df['mean_sim_calls'], clean_df['mean_time'], s=100, c='green', 
                label='Clean', alpha=0.7, marker='o')
    ax4.scatter(noisy_df['mean_sim_calls'], noisy_df['mean_time'], s=100, c='red', 
                label='Noisy', alpha=0.7, marker='s')
    ax4.set_xlabel('Mean Simulator Calls')
    ax4.set_ylabel('Mean Time (s)')
    ax4.set_title('Execution Time vs Simulator Calls')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.suptitle('Execution Time Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()



"""
# Generate execution time plot
plot_execution_times(summary_df, save_path=os.path.join(output_dir, "execution_time_analysis.png"))
print(f"Execution time plot saved to: {os.path.join(output_dir, 'execution_time_analysis.png')}")

"""
# In[ ]:





# In[314]:
"""

# Display summary table
print("\n" + "="*100)
print("EXPERIMENT SUMMARY")
print("="*100)
print(summary_df[['data_type', 'n_contexts', 'n_sensors', 'n_measurements', 
                  'mean_rmse', 'std_rmse', 'mean_fitness', 'mean_time', 'std_time', 
                  'mean_sim_calls', 'mean_generations']].to_string(index=False))

# Print timing summary
print("\n" + "="*100)
print("TIMING SUMMARY")
print("="*100)
clean_total = summary_df[summary_df['data_type']=='clean']['total_time'].sum()
noisy_total = summary_df[summary_df['data_type']=='noisy']['total_time'].sum()
print(f"Total time for CLEAN experiments: {clean_total:.1f}s ({clean_total/60:.1f} min)")
print(f"Total time for NOISY experiments: {noisy_total:.1f}s ({noisy_total/60:.1f} min)")
print(f"Grand total: {clean_total + noisy_total:.1f}s ({(clean_total + noisy_total)/60:.1f} min)")


# In[315]:


# Print output directory structure
print(f"\n\nAll results saved in: {output_dir}")
print("\nDirectory structure:")
for root, dirs, files in os.walk(output_dir):
    level = root.replace(output_dir, '').count(os.sep)
    indent = ' ' * 2 * level
    print(f"{indent}{os.path.basename(root)}/")
    subindent = ' ' * 2 * (level + 1)
    for file in files[:5]:  # Limit files shown
        print(f"{subindent}{file}")
    if len(files) > 5:
        print(f"{subindent}... and {len(files)-5} more files")
"""
