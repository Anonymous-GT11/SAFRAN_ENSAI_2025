import torch
from typing import Tuple, List
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
    find the pair of efficiencies (η_comp, η_turb)
    such that the simulator reproduces these measurements
    as closely as possible.

    The algorithm evolves a population of candidate solutions
    using selection, crossover, and mutation.
    """

    def __init__(self,
                 simulator,
                 target_measurements: List[float],
                 sensors: List[str],
                 pop_size: int = 100,
                 n_generations: int = 50,
                 bounds: List[Tuple[float, float]] = None,
                 target_fitness: float = -5.0,
                 stagnation_generations: int = 10,
                 min_improvement: float = 0.1):

        """
        Initialize all settings for the Genetic Algorithm.

        simulator: function that takes [η_comp, η_turb] and returns simulated sensor readings
        target_measurements: measured sensor values we want to reproduce
        sensors: list of sensors used (e.g. ['P3', 'T5'])
        pop_size: number of individuals in each generation
        n_generations: how many times we evolve the population
        bounds: min/max search intervals for each variable
        """

        self.simulator = simulator
        self.target_measurements = np.array(target_measurements)
        self.sensors = sensors
        self.n_sensors = len(sensors)
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.target_fitness = target_fitness
        self.stagnation_generations = stagnation_generations
        self.min_improvement = min_improvement
        self._stagnation_counter = 0

        # Default bounds if none provided
        if bounds is None:
            self.bounds = [(0.95, 1.02), (0.95, 1.02)]
        else:
            self.bounds = bounds

        # For storing results during the run
        self.best_fitness_history = []  # best fitness per generation
        self.population_history = []    # save population for visualization
        self.population_sd_history = [] # Standard deviation of population
        self.avg_fitness_history = []   # Average fitness per generation


    def initialize_population(self) -> torch.Tensor:
        """
        Create the initial population randomly within given bounds.
        Each individual = [η_comp, η_turb].
        """

        population = torch.zeros((self.pop_size, 2))

        for i in range(2):  # loop over each gene
            min_val, max_val = self.bounds[i]

            # torch.rand() generates numbers in [0,1)
            # We scale them to fit within [min_val, max_val]
            population[:, i] = torch.rand(self.pop_size) * (max_val - min_val) + min_val

        return population


    
    def fitness(self, individual: torch.Tensor) -> float:
        """
        Compute how good an individual is compared to the target measurements.

        We define fitness = -log(L2_norm(relative_error) + epsilon)
        so that smaller errors correspond to higher fitness.
        """

        # Extract the two genes from the individual tensor.
        eta_comp = individual[0].item()
        eta_turb = individual[1].item()

        # Compute simulated measurements using the direct model h_SCSF.
        sim_result = list(self.simulator([[eta_comp, eta_turb]], S=self.sensors))
        simulated_measurements = np.array(sim_result[0]).flatten()

        # For each sensor, compute relative error = (sim - target) / target
        relative_errors = (simulated_measurements - self.target_measurements) / self.target_measurements

        # Calculate the L2 norm (Euclidean distance) of the relative error vector
        l2_norm_error = np.linalg.norm(relative_errors)

        # Add a small epsilon to avoid log(0)
        epsilon = 1e-6
        # Fitness is -log(L2_norm) - higher is better (i.e., smaller L2 norm)
        fitness_value = -np.log(l2_norm_error + epsilon)
        return float(fitness_value)


    def evaluate_population(self, population: torch.Tensor) -> torch.Tensor:
        """
        Compute fitness for every individual in the population.
        Returns a vector of fitness values.
        """
        fitness_values = torch.zeros(self.pop_size)
        for i in range(self.pop_size):
            # fitness() returns a float; we store it in a tensor
            fitness_values[i] = self.fitness(population[i])
        return fitness_values


    #  Compute SD for each generation 
    def compute_population_metrics(self, population: torch.Tensor, fitness_values: torch.Tensor):
        """
        Compute and store the standard deviation of the population and average fitness.
        """
        # Standard deviation of the population in the search space (over both dimensions)
        sd_comp = population[:, 0].std().item()
        sd_turb = population[:, 1].std().item()
        # Average SD of the two parameters
        self.population_sd_history.append((sd_comp + sd_turb) / 2.0)

        # Average fitness
        self.avg_fitness_history.append(torch.mean(fitness_values).item())


    def selection(self, population: torch.Tensor, fitness_values: torch.Tensor, n_parents: int) -> torch.Tensor:
        """
        Select parents using tournament selection (unchanged).
        """
        parents = torch.zeros((n_parents, 2))
        for i in range(n_parents):
            # Pick 3 random indices from the population
            candidates_idx = torch.randint(0, self.pop_size, (3,))
            # Find the one with the best fitness among them
            best_idx = candidates_idx[torch.argmax(fitness_values[candidates_idx])]
            # Add that individual to the parent pool
            parents[i] = population[best_idx]

        return parents


    def crossover(self, parent1: torch.Tensor, parent2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Combine two parents to create two children using uniform crossover (unchanged).
        """
        child1 = parent1.clone()
        child2 = parent2.clone()
        alpha = torch.rand(1).item()  # random mixing weight between 0 and 1

        child1 = alpha * parent1 + (1 - alpha) * parent2
        child2 = (1 - alpha) * parent1 + alpha * parent2

        # Ensure genes remain within allowed bounds
        for i in range(2):
            min_val, max_val = self.bounds[i]
            child1[i] = torch.clamp(child1[i], min_val, max_val)
            child2[i] = torch.clamp(child2[i], min_val, max_val)

        return child1, child2


    # Adopt the scale of mutation (goes smaller across time)
    def mutate(self, individual: torch.Tensor, generation: int, mutation_rate: float = 0.1) -> torch.Tensor:
        """
        Introduce small random changes (mutation) to an individual with an
        adaptively shrinking standard deviation for the noise.
        """
        mutated = individual.clone()  # avoid overwriting the original

        # Adaptive Noise Scale: Decreases over generations
        # We use a simple linear decay for the noise scale: range_factor * (1 - gen/n_gen)
        # The noise scale shrinks from 5% of the range down to 0%.
        range_factor = 0.05
        max_generations = self.n_generations # Use the max generations to scale the decay

        # Ensure factor is not negative, use 1e-4 as a minimum
        decay_factor = max(1e-4, 1.0 - (generation / max_generations))

        for i in range(2):
            # Perform mutation only with probability = mutation_rate
            if torch.rand(1).item() < mutation_rate:
                min_val, max_val = self.bounds[i]

                # Dynamic noise standard deviation
                noise_std = (max_val - min_val) * range_factor * decay_factor

                # torch.normal(mean, std, size) generates Gaussian noise
                mutated[i] += torch.normal(0, noise_std, (1,)).item()

                # Clamp to ensure gene stays inside [min_val, max_val]
                mutated[i] = torch.clamp(mutated[i], min_val, max_val)

        return mutated


    # Average, after 10 generation measure the improvement across generations
   
    def check_stagnation(self, current_best_fitness: float, generation: int) -> bool:
        """
        Checks for stagnation based on the average improvement over the last
        'stagnation_generations' (default 10) generations.
        """
        if generation < self.stagnation_generations:
            # Not enough history to check for stagnation
            return False

        # Get the best fitness from the generation 'stagnation_generations' ago
        previous_best_fitness = self.best_fitness_history[generation - self.stagnation_generations]

        # Improvement is current_best - previous_best. We maximize fitness, so a positive change is good.
        improvement = current_best_fitness - previous_best_fitness

        # Check if the improvement over the window is less than the minimum required
        if improvement < self.min_improvement:
            print(f"\nSTOPPING: Average best fitness improvement over the last {self.stagnation_generations} generations ({improvement:.4f}) is below the minimum threshold ({self.min_improvement:.2f}).")
            return True

        return False


    def evolve(self) -> Tuple[torch.Tensor, List[float]]:
        """
        Main GA loop:
        1. Initialize population
        2. Evaluate fitness
        3. Select parents
        4. Apply crossover and mutation
        5. Keep best individuals (elitism)
        6. Repeat for n_generations
        """

        # Step 1: create initial population
        population = self.initialize_population()
        self._stagnation_counter = 0

        for generation in range(self.n_generations):

            # Step 2: evaluate fitness of all individuals
            fitness_values = self.evaluate_population(population)
            current_best_fitness = torch.max(fitness_values).item()
            best_individual = population[torch.argmax(fitness_values)]

            # Compute population metrics
            self.compute_population_metrics(population, fitness_values)

            # Store current best and population
            self.best_fitness_history.append(current_best_fitness)
            self.population_history.append(population.clone())


            # --- STOPPING CRITERIA CHECKS ---

            # Criterion 1: Fitness Threshold Achieved
            if current_best_fitness >= self.target_fitness:
                print(f"\nSTOPPING: Target fitness ({self.target_fitness:.2f}) achieved at Generation {generation}.")
                break

            # Criterion 2: Convergence/Stagnation Check (Modified)
            if self.check_stagnation(current_best_fitness, generation):
                break


            # --- ELITISM ---
            # Preserve the best few individuals unchanged
            n_elite = int(self.pop_size * 0.1) # 10% elitism
            elite_idx = torch.argsort(fitness_values, descending=True)[:n_elite]
            elite = population[elite_idx]

            # --- SELECTION ---
            # Select remaining parents for breeding
            n_parents = self.pop_size - n_elite
            parents = self.selection(population, fitness_values, n_parents)

            # --- NEW GENERATION ---
            new_population = torch.zeros((self.pop_size, 2))
            new_population[:n_elite] = elite  # keep elite individuals

            # --- CROSSOVER + MUTATION ---
            # Generate new individuals two by two
            for i in range(n_elite, self.pop_size, 2):
                # Tournament selection re-applies here, but for simplicity,
                # we'll just sample from the selected parent pool
                parent1 = parents[torch.randint(0, n_parents, (1,)).item()]
                parent2 = parents[torch.randint(0, n_parents, (1,)).item()]

                # Crossover to create two children
                child1, child2 = self.crossover(parent1, parent2)

                # Mutate both children with generation-dependent scale
                child1 = self.mutate(child1, generation)
                child2 = self.mutate(child2, generation)

                # Add them to the new population
                new_population[i] = child1
                if i + 1 < self.pop_size:
                    new_population[i + 1] = child2

            # Replace old population with the new one
            population = new_population

            # Display progress every 10 generations
            if generation % 10 == 0:
                print(f"Generation {generation}: "
                      f"Best Fitness = {current_best_fitness:.2f}, "
                      f"Avg Fitness = {self.avg_fitness_history[-1]:.2f}, "
                      f"Pop SD = {self.population_sd_history[-1]:.4f}, "
                      f"eta_comp = {best_individual[0]:.4f}, "
                      f"eta_turb = {best_individual[1]:.4f}")

        # Final evaluation after evolution
        final_fitness = self.evaluate_population(population)
        best_idx = torch.argmax(final_fitness)
        best_solution = population[best_idx]

        return best_solution, self.best_fitness_history
