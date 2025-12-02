import torch
from typing import Tuple, List, Optional
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class GeneticAlgorithm:
    """
    Genetic Algorithm for solving an INVERSE PROBLEM.

    Goal:
    -----
    Given sensor measurements (e.g. P3, T5),
    find the health parameters (e.g., η_comp, η_turb)
    such that the simulator reproduces these measurements
    as closely as possible.
    """

    def __init__(self,
                 simulator,
                 target_measurements: List[float],
                 sensors: List[str],
                 pop_size: int = 100,
                 n_generations: int = 50,
                 bounds: Optional[List[Tuple[float, float]]] = None,
                 target_fitness: float = -5.0,
                 stagnation_generations: int = 10,
                 min_improvement: float = 0.1,
                 mutation_rate: float = 0.1,  # Added for clarity in mutation
                 n_elite_ratio: float = 0.1): # Ratio for elitism
        
        self.simulator = simulator
        self.target_measurements = np.array(target_measurements)
        self.sensors = sensors
        self.n_sensors = len(sensors)
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.target_fitness = target_fitness
        self.stagnation_generations = stagnation_generations
        self.min_improvement = min_improvement
        self.mutation_rate = mutation_rate
        self.n_elite = max(1, int(pop_size * n_elite_ratio)) # Ensure at least 1 elite

        # Default bounds if none provided (assuming 2 genes)
        if bounds is None:
            self.bounds = [(0.95, 1.02), (0.95, 1.02)]
        else:
            self.bounds = bounds
            
        self.n_genes = len(self.bounds) # Dynamically determine n_genes

        # For storing results during the run
        self.best_fitness_history = []  # best fitness per generation
        self.population_history = []    # save population for visualization
        self.population_sd_history = [] # Standard deviation of population
        self.avg_fitness_history = []   # Average fitness per generation


    ## Population Initialization
    def initialize_population(self) -> torch.Tensor:
        """
        Create the initial population randomly within given bounds (generalized).
        """
        population = torch.zeros((self.pop_size, self.n_genes))

        for i in range(self.n_genes):  # loop over each gene
            min_val, max_val = self.bounds[i]
            population[:, i] = torch.rand(self.pop_size) * (max_val - min_val) + min_val

        return population


    ##  Fitness Function
    def fitness(self, individual: torch.Tensor) -> float:
        """
        Compute how good an individual is compared to the target measurements.
        We define fitness = -log(L2_norm(relative_error) + epsilon).
        """
        # Extract genes as a Python list of floats
        genes = [g.item() for g in individual] 

        # Compute simulated measurements
        sim_result = list(self.simulator([genes], S=self.sensors))
        simulated_measurements = np.array(sim_result[0]).flatten()

        # Calculate relative errors: (sim - target) / target
        relative_errors = (simulated_measurements - self.target_measurements) / self.target_measurements

        # Calculate the L2 norm of the relative error vector
        l2_norm_error = np.linalg.norm(relative_errors)
        
        # Add a small epsilon to avoid log(0)
        epsilon = 1e-12 
        
        # Fitness is -log(L2_norm) - higher is better 
        fitness_value = -np.log(l2_norm_error + epsilon)
        
        return float(fitness_value)


    def evaluate_population(self, population: torch.Tensor) -> torch.Tensor:
        """Compute fitness for every individual in the population."""
        fitness_values = torch.zeros(self.pop_size)
        for i in range(self.pop_size):
            fitness_values[i] = self.fitness(population[i])
        return fitness_values


    ##  Metric Calculation
    def compute_population_metrics(self, population: torch.Tensor, fitness_values: torch.Tensor):
        """Compute and store the standard deviation of the population and average fitness."""
        
        # Standard deviation of the population in the search space (over all dimensions)
        sd_vals = population.std(dim=0).cpu().numpy()
        self.population_sd_history.append(np.mean(sd_vals).item()) # Average SD across all genes

        # Average fitness
        self.avg_fitness_history.append(torch.mean(fitness_values).item())


    ##  Selection
    def selection(self, population: torch.Tensor, fitness_values: torch.Tensor, n_parents: int) -> torch.Tensor:
        """Select parents using tournament selection."""
       
        parents = torch.zeros((n_parents, self.n_genes))
        for i in range(n_parents):
           
            candidates_idx = torch.randint(0, self.pop_size, (3,)) 
            best_idx = candidates_idx[torch.argmax(fitness_values[candidates_idx])]
            parents[i] = population[best_idx]

        return parents


    ## Crossover
    def crossover(self, parent1: torch.Tensor, parent2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Combine two parents to create two children using uniform crossover."""
        child1 = parent1.clone()
        child2 = parent2.clone()
        alpha = torch.rand(1).item() 

        child1 = alpha * parent1 + (1 - alpha) * parent2
        child2 = (1 - alpha) * parent1 + alpha * parent2

        # Ensure genes remain within allowed bounds
        for i in range(self.n_genes):
            min_val, max_val = self.bounds[i]
            child1[i] = torch.clamp(child1[i], min_val, max_val)
            child2[i] = torch.clamp(child2[i], min_val, max_val)

        return child1, child2


    ##  Mutation (Adaptive)
    def mutate(self, individual: torch.Tensor, generation: int) -> torch.Tensor:
        """
        Introduce small random changes (mutation) with adaptively shrinking 
        standard deviation.
        """
        mutated = individual.clone() 
        
        # Hyperparameters for mutation scale
        range_factor = 0.05 
        max_generations = self.n_generations 

        # Adaptive Noise Scale: Decays over generations
        decay_factor = max(1e-4, 1.0 - (generation / max_generations))

        for i in range(self.n_genes):
            # Perform mutation only with probability = self.mutation_rate
            if torch.rand(1).item() < self.mutation_rate:
                min_val, max_val = self.bounds[i]

                # Dynamic noise standard deviation
                noise_std = (max_val - min_val) * range_factor * decay_factor
                
                
                mutated[i] += torch.normal(0, noise_std, (1,)).item()
                
                # Ensure gene stays inside [min_val, max_val]
                mutated[i] = torch.clamp(mutated[i], min_val, max_val)

        return mutated


    ## Stagnation Check
    def check_stagnation(self, current_best_fitness: float, generation: int) -> bool:
        """
        Checks for stagnation based on the best fitness improvement over the 
        stagnation_generations window.
        """
        if generation < self.stagnation_generations:
            return False

        # Get the best fitness from the generation 'stagnation_generations' ago
        previous_best_fitness = self.best_fitness_history[generation - self.stagnation_generations]

        # Improvement is current_best - previous_best.
        improvement = current_best_fitness - previous_best_fitness

        if improvement < self.min_improvement:
            print(f"\nSTOPPING: Average best fitness improvement over the last {self.stagnation_generations} generations ({improvement:.4f}) is below the minimum threshold ({self.min_improvement:.2f}).")
            return True

        return False


    ## Main Evolution Loop
    def evolve(self) -> Tuple[torch.Tensor, List[float]]:
        """Main GA evolution loop."""

        population = self.initialize_population()

        for generation in range(self.n_generations):

            # Evaluate Fitness
            fitness_values = self.evaluate_population(population)
            current_best_fitness = torch.max(fitness_values).item()
            best_individual = population[torch.argmax(fitness_values)]

            #  Compute and Store History
            self.compute_population_metrics(population, fitness_values)
            self.best_fitness_history.append(current_best_fitness)
            self.population_history.append(population.clone())

            # Stopping Criteria
            if current_best_fitness >= self.target_fitness:
                print(f"\nSTOPPING: Target fitness ({self.target_fitness:.2f}) achieved at Generation {generation}.")
                break
            if self.check_stagnation(current_best_fitness, generation):
                break


            #  Elitism
            elite_idx = torch.argsort(fitness_values, descending=True)[:self.n_elite]
            elite = population[elite_idx]

            #  Selection
            n_parents = self.pop_size - self.n_elite
            parents = self.selection(population, fitness_values, n_parents)

            # New Generation Setup
            new_population = torch.zeros((self.pop_size, self.n_genes))
            new_population[:self.n_elite] = elite

            #  Crossover + Mutation
            for i in range(self.n_elite, self.pop_size, 2):
                parent1 = parents[torch.randint(0, n_parents, (1,)).item()]
                parent2 = parents[torch.randint(0, n_parents, (1,)).item()]

                child1, child2 = self.crossover(parent1, parent2)

                # Mutate with generation-dependent scale
                child1 = self.mutate(child1, generation)
                child2 = self.mutate(child2, generation)

                new_population[i] = child1
                if i + 1 < self.pop_size:
                    new_population[i + 1] = child2

            population = new_population

            # Logging
            if generation % 10 == 0:
                # --- FIXED: Only log the first two genes for display, as n_genes is currently 2 ---
                gene_display = best_individual[:2].tolist() 
                print(f"Gen {generation}: "
                      f"Best Fit = {current_best_fitness:.2f}, "
                      f"Avg Fit = {self.avg_fitness_history[-1]:.2f}, "
                      f"Pop SD = {self.population_sd_history[-1]:.4f}, "
                      f"Genes (η_c, η_t) = [{gene_display[0]:.4f}, {gene_display[1]:.4f}]")

        # Final best
        final_fitness = self.evaluate_population(population)
        best_idx = torch.argmax(final_fitness)
        best_solution = population[best_idx]

        return best_solution, self.best_fitness_history
