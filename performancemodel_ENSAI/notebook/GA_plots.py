
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from typing import Tuple, List

# You might need to import the class for proper type hinting
# from ga_core import GeneticAlgorithm # If you put them in separate cells, 
                                      # Colab might not need this line until the notebook is restarted.

# (Paste all your plotting functions: plot_ga_seaborn, plot_convergence_seaborn, etc.)
# ...
def plot_ga_seaborn(ga, best_solution, true_health=None):
    # ...
    pass
    
def plot_convergence_seaborn(ga):
    # ...
    pass
    
def plot_final_fitness_scatter(ga: 'GeneticAlgorithm', best_solution: torch.Tensor, true_health: Tuple[float, float] = None):
    # ...
    pass
