import torch
from typing import Tuple, List,Callable, Optional
import numpy as np



import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time



from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go

#====================================================================
#----------------------------VERSION 1
#====================================================================
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
                 bounds: List[Tuple[float, float]] = None):
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

        # Default bounds if none provided
        if bounds is None:
            self.bounds = [(0.95, 1.02), (0.95, 1.02)]
        else:
            self.bounds = bounds

        # For storing results during the run
        self.best_fitness_history = []  # best fitness per generation
        self.population_history = []    # save population for visualization


    def initialize_population(self) -> torch.Tensor:
        """
        Create the initial population randomly within given bounds.
        Each individual = [η_comp, η_turb].
        """

        # Create an empty tensor for all individuals
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

        We define fitness = -mean(relative_error)
        so that smaller errors correspond to higher fitness.
        """

        # Extract the two genes from the individual tensor.
        # .item() converts a single-element tensor into a Python float.
        # This is necessary because the simulator likely expects standard Python floats.
        eta_comp = individual[0].item()
        eta_turb = individual[1].item()

        # Compute simulated measurements using the direct model h_SCSF.
        # The simulator returns results for all sensors at once.
        sim_result = list(self.simulator([[eta_comp, eta_turb]], S=self.sensors))
        simulated_measurements = np.array(sim_result[0]).flatten()

        errors = []
        # For each sensor, compute relative error = |sim - target| / |target|
        for i in range(self.n_sensors):
            target = self.target_measurements[i]
            simulated = simulated_measurements[i]
            #print("simulated: ", simulated)
            error = abs(simulated - target) / abs(target)
            errors.append(error)

        total_error = np.mean(errors)

        # Multiply by -1000 just to make fitness values more readable (maximize fitness)
        fitness_value = -total_error * 1000
        return fitness_value


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


    def selection(self, population: torch.Tensor, fitness_values: torch.Tensor, n_parents: int) -> torch.Tensor:
        """
        Select parents using tournament selection.

        For each parent slot:
        - Randomly select 3 individuals from the population
        - Keep the one with the highest fitness
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
        Combine two parents to create two children using uniform crossover.

        alpha is a random number in [0, 1].
        child1 = α * parent1 + (1 - α) * parent2
        child2 = (1 - α) * parent1 + α * parent2
        """
        # Copy parents to avoid modifying them
        child1 = parent1.clone()
        child2 = parent2.clone()

        alpha = torch.rand(1).item()  # random mixing weight between 0 and 1

        # Create new individuals as weighted averages of the parents
        child1 = alpha * parent1 + (1 - alpha) * parent2
        child2 = (1 - alpha) * parent1 + alpha * parent2

        # Ensure genes remain within allowed bounds
        for i in range(2):
            min_val, max_val = self.bounds[i]
            child1[i] = torch.clamp(child1[i], min_val, max_val)
            child2[i] = torch.clamp(child2[i], min_val, max_val)

        return child1, child2


    def mutate(self, individual: torch.Tensor, mutation_rate: float = 0.1) -> torch.Tensor:
        """
        Introduce small random changes (mutation) to an individual.

        Each gene has a chance 'mutation_rate' to be modified.
        The modification adds Gaussian noise N(0, σ²)
        where σ = 5% of the search range.
        """
        mutated = individual.clone()  # avoid overwriting the original

        for i in range(2):
            # Perform mutation only with probability = mutation_rate
            if torch.rand(1).item() < mutation_rate:
                min_val, max_val = self.bounds[i]
                noise_std = (max_val - min_val) * 0.05  # standard deviation of noise

                # torch.normal(mean, std, size) generates Gaussian noise
                mutated[i] += torch.normal(0, noise_std, (1,)).item()

                # Clamp to ensure gene stays inside [min_val, max_val]
                mutated[i] = torch.clamp(mutated[i], min_val, max_val)

        return mutated


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

        for generation in range(self.n_generations):

            # Step 2: evaluate fitness of all individuals
            fitness_values = self.evaluate_population(population)

            # Keep best fitness value for plotting convergence
            best_fitness = torch.max(fitness_values).item()
            self.best_fitness_history.append(best_fitness)
            self.population_history.append(population.clone())

            # --- ELITISM ---
            # Preserve the best few individuals unchanged
            n_elite = 10
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
                parent1 = parents[torch.randint(0, n_parents, (1,)).item()]
                parent2 = parents[torch.randint(0, n_parents, (1,)).item()]

                # Crossover to create two children
                child1, child2 = self.crossover(parent1, parent2)

                # Mutate both children
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)

                # Add them to the new population
                new_population[i] = child1
                if i + 1 < self.pop_size:
                    new_population[i + 1] = child2

            # Replace old population with the new one
            population = new_population

            # Display progress every 10 generations
            if generation % 10 == 0:
                best_individual = population[torch.argmax(fitness_values)]
                print(f"Generation {generation}: "
                      f"Fitness = {best_fitness:.2f}, "
                      f"eta_comp = {best_individual[0]:.4f}, "
                      f"eta_turb = {best_individual[1]:.4f}")

        # Final evaluation after evolution
        final_fitness = self.evaluate_population(population)
        best_idx = torch.argmax(final_fitness)
        best_solution = population[best_idx]

        return best_solution, self.best_fitness_history




#======================================================================================
#----------------------------------VERSION 2
#======================================================================================
"""
Parameterization + test with only 1 measurements +fixed of outbounds
"""
class GeneticAlgorithmv2:
  """
  Generic, parameterized Genetic Algorithm (GA)
  for solving inverse problems.

  You can customize:
  - Number of genes
  - Population size
  - Mutation rate
  - Crossover rate
  - Tournament size
  - Elitism rate
  - Search bounds
  """

  def __init__(self,
               simulator: Callable,
               target_measurements: List[float],
               sensors: List[str],
               n_genes: int = 2,
               pop_size: int = 100,
               n_generations: int = 50,
               bounds: Optional[List[Tuple[float, float]]] = None,
               mutation_rate: float = 0.1,
               crossover_rate: float = 0.9,
               tournament_size: int = 3,
               n_elite: int = 10,
               mutation_std_ratio: float = 0.05):
    """
    Parameters
    ----------
    simulator : function
        Takes list of genes and returns simulated sensor readings.
    target_measurements : list of float
        Observed target values to match.
    sensors : list of str
        List of sensor names used.
    n_genes : int, default=2
        Number of genes (parameters to optimize).
    pop_size : int, default=100
        Number of individuals per generation.
    n_generations : int, default=50
        Total number of evolution cycles.
    bounds : list of tuple, optional
        [(min, max)] for each gene.
    mutation_rate : float, default=0.1
        Probability of mutating a gene.
    crossover_rate : float, default=0.9
        Probability of performing crossover.
    tournament_size : int, default=3
        Number of individuals in tournament selection.
    n_elite: int, default=10
        Fraction of best individuals to keep unchanged.
    mutation_std_ratio : float, default=0.05
        Fraction of search range used as mutation noise std.
    """
    self.simulator = simulator
    self.target_measurements = np.array(target_measurements)
    self.sensors = sensors
    self.n_sensors = len(sensors)
    self.n_genes = n_genes
    self.pop_size = pop_size
    self.n_generations = n_generations
    self.mutation_rate = mutation_rate
    self.crossover_rate = crossover_rate
    self.tournament_size = tournament_size
    self.n_elite = n_elite
    self.mutation_std_ratio = mutation_std_ratio

    # Default bounds if not given
    if bounds is None:
      self.bounds = [(0.95, 1.02)] * n_genes
    else:
      assert len(bounds) == n_genes, "Bounds list must match number of genes"
      self.bounds = bounds

    # Storage for monitoring
    self.best_fitness_history = []
    self.population_history = []


  # =======================================================
  # --- POPULATION INITIALIZATION ---
  # =======================================================


  def initialize_population(self) -> torch.Tensor:
        """
        Create the initial population randomly within given bounds.
        Ensures all individuals remain strictly within [min_val, max_val].
        ach individual = [η_comp, η_turb| or n_comb]
        """
        n_genes = len(self.bounds)
        population = torch.zeros((self.pop_size, n_genes))
    
        for i, (min_val, max_val) in enumerate(self.bounds):
            # Uniform random values
            vals = torch.rand(self.pop_size) * (max_val - min_val) + min_val
    
            # Clamp (safety check)
            vals = torch.clamp(vals, min_val, max_val)
    
            population[:, i] = vals
    
        return population


  # =======================================================
  # --- FITNESS FUNCTION ---
  # =======================================================
  def fitness(self, individual: torch.Tensor) -> float:
    """
    Evaluate one individual (lower error = higher fitness).
    """
    genes = [g.item() for g in individual]
    sim_result = list(self.simulator([genes], S=self.sensors))
      
    simulated_measurements = np.array(sim_result[0]).flatten()

    rel_errors = np.abs(simulated_measurements - self.target_measurements) / np.abs(self.target_measurements)
    total_error = np.mean(rel_errors)

    return -total_error * 1000  # maximize fitness


  def evaluate_population(self, population: torch.Tensor) -> torch.Tensor:
    """Compute fitness for all individuals."""
    return torch.tensor([self.fitness(ind) for ind in population])


  # =======================================================
  # --- SELECTION ---
  # =======================================================
  def selection(self, population: torch.Tensor, fitness_values: torch.Tensor, n_parents: int) -> torch.Tensor:
    """
    Tournament selection of parents.
    """
    parents = torch.zeros((n_parents, self.n_genes))
    for i in range(n_parents):
      candidates_idx = torch.randint(0, self.pop_size, (self.tournament_size,))
      best_idx = candidates_idx[torch.argmax(fitness_values[candidates_idx])]
      parents[i] = population[best_idx]
        
    return parents


  # =======================================================
  # --- CROSSOVER ---
  # =======================================================
  def crossover(self, parent1: torch.Tensor, parent2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Uniform crossover with random mixing factor α.
    """
    if torch.rand(1).item() > self.crossover_rate:
      return parent1.clone(), parent2.clone()

    alpha = torch.rand(1).item()
    child1 = alpha * parent1 + (1 - alpha) * parent2
    child2 = (1 - alpha) * parent1 + alpha * parent2

    for i, (min_val, max_val) in enumerate(self.bounds):
      child1[i] = torch.clamp(child1[i], min_val, max_val)
      child2[i] = torch.clamp(child2[i], min_val, max_val)

    

    return child1, child2


  # =======================================================
  # --- MUTATION ---
  # =======================================================
  def mutate(self, individual: torch.Tensor) -> torch.Tensor:
    """Random Gaussian mutation applied with probability."""
    mutated = individual.clone()
    for i, (min_val, max_val) in enumerate(self.bounds):
      if torch.rand(1).item() < self.mutation_rate:
        noise_std = (max_val - min_val) * self.mutation_std_ratio
        mutated[i] += torch.normal(0, noise_std, (1,)).item()
        mutated[i] = torch.clamp(mutated[i], min_val, max_val)
          
    return mutated


  # =======================================================
  # --- MAIN EVOLUTION LOOP ---
  # =======================================================
  def evolve(self, verbose_interval: int = 10) -> Tuple[torch.Tensor, List[float]]:
    """
    Full GA evolution loop.
    """
    population = self.initialize_population()

    n_elite = self.n_elite# max(1, int(self.elitism_rate * self.pop_size))

    for generation in range(self.n_generations):
      fitness_values = self.evaluate_population(population)

      # Record best fitness
      best_fitness = torch.max(fitness_values).item()
      self.best_fitness_history.append(best_fitness)
      self.population_history.append(population.clone())

      # Elitism
      elite_idx = torch.argsort(fitness_values, descending=True)[:n_elite]
      elite = population[elite_idx]



      # Selection
      n_parents = self.pop_size - n_elite
      parents = self.selection(population, fitness_values, n_parents)

      # New generation
      new_population = torch.zeros((self.pop_size, self.n_genes))
      new_population[:n_elite] = elite

      # Crossover + mutation
      for i in range(n_elite, self.pop_size, 2):
        parent1 = parents[torch.randint(0, n_parents, (1,)).item()]
        parent2 = parents[torch.randint(0, n_parents, (1,)).item()]
        child1, child2 = self.crossover(parent1, parent2)
        new_population[i] = self.mutate(child1)
        if i + 1 < self.pop_size:
          new_population[i + 1] = self.mutate(child2)

      population = new_population

      # Final global clamp to keep everything in physical range
      for j, (low, high) in enumerate(self.bounds):
        population[:, j] = torch.clamp(population[:, j], low, high)


      # Logging
      if generation % verbose_interval == 0 or generation == self.n_generations - 1:
        best_individual = population[torch.argmax(fitness_values)]
        print(f"Gen {generation:3d} | Best Fitness = {best_fitness:8.3f} | Genes = {best_individual.tolist()}")

      # Print estimated measurement for the last  
      if generation == self.n_generations - 1:
         
          best_individual = population[torch.argmax(fitness_values)]
          # print(f"Gen {generation:3d} | Best Fitness = {best_fitness:8.3f} | Genes = {best_individual.tolist()}")

          sim_result = list(self.simulator([best_individual], S=self.sensors))
      
          simulated_measurements = np.array(sim_result[0]).flatten()

          print (f"simulated measurement : {simulated_measurements}")
      
        

    # Final best
    final_fitness = self.evaluate_population(population)
    best_idx = torch.argmax(final_fitness)
    best_solution = population[best_idx]
    return best_solution, self.best_fitness_history




#======================================================================================
#----------------------------------VERSION 3
#======================================================================================
"""
L2-norm + adaptive mutation 
"""
class GeneticAlgorithmv3:
    

    def __init__(self,
                 simulator: Callable,
                 target_measurements: List[float],
                 sensors: List[str],
                 pop_size: int = 100,
                 n_generations: int = 50,
                 bounds: Optional[List[Tuple[float, float]]] = None,
                 target_fitness: float = -5.0,
                 stagnation_generations: int = 10,
                 min_improvement: float = 0.1,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.9,
                 tournament_size: int = 3,
                 n_elite: int = 10,
                 mutation_std_ratio: float = 0.05):
        
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
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.n_elite = n_elite
        self.mutation_std_ratio = mutation_std_ratio

        if bounds is None:
            self.bounds = [(0.95, 1.02), (0.95, 1.02)]
        else:
            self.bounds = bounds
            
        self.n_genes = len(self.bounds)

        # Storage for monitoring
        self.best_fitness_history = []
        self.population_history = []
        self.population_sd_history = []
        self.avg_fitness_history = []



  # =======================================================
  # --- POPULATION INITIALIZATION ---
  # =======================================================



    def initialize_population(self) -> torch.Tensor:
        population = torch.zeros((self.pop_size, self.n_genes))
        for i in range(self.n_genes):
            min_val, max_val = self.bounds[i]
            population[:, i] = torch.rand(self.pop_size) * (max_val - min_val) + min_val
        return population
# =======================================================
# --- FITNESS FUNCTION ---
# =======================================================

    def fitness(self, individual: torch.Tensor) -> float:
        genes = [g.item() for g in individual] 
        sim_result = list(self.simulator([genes], S=self.sensors))
        simulated_measurements = np.array(sim_result[0]).flatten()
        
        relative_errors = (simulated_measurements - self.target_measurements) / self.target_measurements
        l2_norm_error = np.linalg.norm(relative_errors)
        
        epsilon = 1e-12 
        fitness_value = -np.log(l2_norm_error + epsilon)
        
        return float(fitness_value)


    def evaluate_population(self, population: torch.Tensor) -> torch.Tensor:
        return torch.tensor([self.fitness(ind) for ind in population])


    def compute_population_metrics(self, population: torch.Tensor, fitness_values: torch.Tensor):
        sd_vals = population.std(dim=0).cpu().numpy()
        self.population_sd_history.append(np.mean(sd_vals).item()) 
        self.avg_fitness_history.append(torch.mean(fitness_values).item())

  # =======================================================
  # --- SELECTION ---
  # =======================================================
    def selection(self, population: torch.Tensor, fitness_values: torch.Tensor, n_parents: int) -> torch.Tensor:
        parents = torch.zeros((n_parents, self.n_genes))
        for i in range(n_parents):
            candidates_idx = torch.randint(0, self.pop_size, (self.tournament_size,))
            best_idx = candidates_idx[torch.argmax(fitness_values[candidates_idx])]
            parents[i] = population[best_idx]
        return parents

 # =======================================================
 # --- CROSSOVER ---
 # =======================================================
    def crossover(self, parent1: torch.Tensor, parent2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        child1, child2 = parent1.clone(), parent2.clone()
        if torch.rand(1).item() < self.crossover_rate:
            alpha = torch.rand(1).item() 
            child1 = alpha * parent1 + (1 - alpha) * parent2
            child2 = (1 - alpha) * parent1 + alpha * parent2
            
        for i in range(self.n_genes):
            min_val, max_val = self.bounds[i]
            child1[i] = torch.clamp(child1[i], min_val, max_val)
            child2[i] = torch.clamp(child2[i], min_val, max_val)
            
        return child1, child2

# =======================================================
# --- MUTATION ---
# =======================================================
    def mutate(self, individual: torch.Tensor, generation: int) -> torch.Tensor:
        mutated = individual.clone() 
        range_factor = self.mutation_std_ratio 
        decay_factor = max(1e-4, 1.0 - (generation / self.n_generations)) 

        for i in range(self.n_genes):
            if torch.rand(1).item() < self.mutation_rate:
                min_val, max_val = self.bounds[i]
                noise_std = (max_val - min_val) * range_factor * decay_factor
                mutated[i] += torch.normal(0, noise_std, (1,)).item()
                mutated[i] = torch.clamp(mutated[i], min_val, max_val)
        return mutated


    def check_stagnation(self, current_best_fitness: float, generation: int) -> bool:
        if generation < self.stagnation_generations:
            return False
        
    # Compares Gen 'N' with Gen 'N - stagnation_generations'
        previous_best_fitness = self.best_fitness_history[generation - self.stagnation_generations]
        improvement = current_best_fitness - previous_best_fitness
    
        if improvement < self.min_improvement:
            print(f"\nSTOPPING: Stagnation detected. Improvement ({improvement:.4f}) over {self.stagnation_generations} generations is below threshold ({self.min_improvement:.4f}).")
            return True
        return False
# =======================================================
# --- MAIN EVOLUTION LOOP ---
# =======================================================

    def evolve(self, verbose_interval: int = 10) -> Tuple[torch.Tensor, List[float]]:
        population = self.initialize_population()
        
        for generation in range(self.n_generations):
            fitness_values = self.evaluate_population(population)
            current_best_fitness = torch.max(fitness_values).item()
            best_individual = population[torch.argmax(fitness_values)]

            self.compute_population_metrics(population, fitness_values)
            self.best_fitness_history.append(current_best_fitness)
            self.population_history.append(population.clone())

            if current_best_fitness >= self.target_fitness:
                print(f"\nSTOPPING: Target fitness ({self.target_fitness:.2f}) achieved at Generation {generation}.")
                break
                
            if self.check_stagnation(current_best_fitness, generation):
                break

            # --- INDENTATION FIXED HERE ---
            elite_idx = torch.argsort(fitness_values, descending=True)[:self.n_elite]
            elite = population[elite_idx]
            
            n_parents = self.pop_size - self.n_elite
            parents = self.selection(population, fitness_values, n_parents)
            
            new_population = torch.zeros((self.pop_size, self.n_genes))
            new_population[:self.n_elite] = elite

            for i in range(self.n_elite, self.pop_size, 2):
                parent1 = parents[torch.randint(0, n_parents, (1,)).item()]
                parent2 = parents[torch.randint(0, n_parents, (1,)).item()]

                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1, generation)
                child2 = self.mutate(child2, generation)

                new_population[i] = child1
                if i + 1 < self.pop_size:
                    new_population[i + 1] = child2

            population = new_population

            if generation % verbose_interval == 0:
                gene_display = best_individual.tolist() 
                print(f"Gen {generation}: "
                      f"Best Fit = {current_best_fitness:.2f}, "
                      f"Avg Fit = {self.avg_fitness_history[-1]:.2f}, "
                      f"Pop SD = {self.population_sd_history[-1]:.4f}, "
                      f"Genes = [{', '.join(f'{g:.4f}' for g in gene_display)}]")

        final_fitness = self.evaluate_population(population)
        best_idx = torch.argmax(final_fitness)
        best_solution = population[best_idx]

        return best_solution, self.best_fitness_history

# =======================================================
# --- PLOTTING UTILITIES ---
# =======================================================
# 2D plot
# def plot_ga_population(ga, best_solution, true_health=None):
#   """Visualize all generations in 2D (works for 2 genes)."""
#   if ga.n_genes != 2:
#     print("--Plot skipped: only supported for 2-gene problems.")
#     return

#   data = []
#   for gen_idx, pop in enumerate(ga.population_history):
#     for x, y in zip(pop[:, 0].numpy(), pop[:, 1].numpy()):
#       data.append({"eta_comp": x, "eta_turb": y, "generation": gen_idx})

#   df = pd.DataFrame(data)
#   plt.figure(figsize=(7, 6))
#   sns.scatterplot(data=df, x="eta_comp", y="eta_turb", hue="generation",
#                   palette="viridis", s=60, alpha=0.8, edgecolor="white")

#   if true_health is not None:
#     plt.scatter(true_health[0], true_health[1], c="lime", s=120, marker="X", label="True")
#   plt.scatter(best_solution[0].item(), best_solution[1].item(),
#               c="red", s=120, marker="*", label="Best")

#   plt.xlabel("η_comp (Compressor)")
#   plt.ylabel("η_turb (Turbine)")
#   plt.title(f"Population Evolution (Sensors: {ga.sensors})")
#   plt.legend()
#   plt.grid(True)
#   plt.show()


#===========================================================================
# ----------------PLOT
#==============================================================================

#-----------------------------------------------------------------
#-------------------------For plot
#-----------------------------------------------------------------


def plot_ga_seaborn(ga, best_solution, true_health=None):
  """
  Plot all individuals from all generations in a single Seaborn scatter plot.
  The color indicates the generation number.
  """

  data = []

  # Collect data from each generation
  for gen_idx, population in enumerate(ga.population_history):
    eta_comp = population[:, 0].cpu().numpy()
    eta_turb = population[:, 1].cpu().numpy()
    for x, y in zip(eta_comp, eta_turb):
      data.append({"eta_comp": x, "eta_turb": y, "generation": gen_idx})

  df = pd.DataFrame(data)

  # Create scatter plot
  plt.figure(figsize=(7, 6))
  sns.scatterplot(
      data=df,
      x="eta_comp", y="eta_turb",
      hue="generation",
      palette="viridis",
      s=60, alpha=0.8, edgecolor="white"
  )

  # Plot the true and best solutions
  if true_health is not None:
    plt.scatter(true_health[0], true_health[1], c="lime", s=120, marker="X", label="True")
  plt.scatter(best_solution[0].item(), best_solution[1].item(),
              c="red", s=120, marker="*", label="Best")

  plt.xlabel("η_comp (Compressor Health)")
  plt.ylabel("η_turb (Turbine Health)")
  plt.title(f"Population Evolution over 5 Generations (Sensors: {ga.sensors})")
  plt.legend()
  plt.grid(True)
  plt.show()



# def plot_convergence_seaborn(ga):
#   """
#   Plot the convergence of best fitness across generations using Seaborn.
#   """

#   generations = list(range(len(ga.best_fitness_history)))
#   df = pd.DataFrame({
#       "Generation": generations,
#       "Best Fitness": ga.best_fitness_history
#   })

#   plt.figure(figsize=(7, 4))
#   sns.lineplot(data=df, x="Generation", y="Best Fitness", marker="o", linewidth=2.5, color="royalblue")
#   sns.despine()
#   plt.grid(True, alpha=0.3)

#   plt.title("Fitness Convergence over Generations", fontsize=13, weight="bold")
#   plt.xlabel("Generation")
#   plt.ylabel("Best Fitness (Higher is Better)")

#   plt.show()


def plot_convergence_log(ga):
    df = pd.DataFrame({
        "Generation": range(len(ga.best_fitness_history)),
        "Best Fitness": ga.best_fitness_history
    })

    plt.figure(figsize=(7, 4))
    sns.lineplot(data=df, x="Generation", y="Best Fitness",
                 marker="o", linewidth=2.5, color="royalblue")

    # log-like scale that supports negatives:
    plt.yscale("symlog", linthresh=1e-6)
    ymin = min(ga.best_fitness_history)
    plt.ylim(ymin, 0)     # upper bound = 0 because fitness → 0

    plt.grid(True, which="both", alpha=0.3)
    plt.title("Fitness Convergence")
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness (symlog)")
    plt.show()
    

#used in v2
def plot_convergence(ga):
  """Plot fitness convergence curve."""
  df = pd.DataFrame({"Generation": range(len(ga.best_fitness_history)),
                     "Best Fitness": ga.best_fitness_history})
  plt.figure(figsize=(7, 4))
  sns.lineplot(data=df, x="Generation", y="Best Fitness", marker="o", linewidth=2.5, color="royalblue")
  plt.grid(True, alpha=0.3)
  plt.title("Fitness Convergence")
  plt.xlabel("Generation")
  plt.ylabel("Best Fitness")
  plt.show()






# PLot for 2 and 3 D
def plot_ga_population(ga, best_solution, true_health=None):
    """
    Visualize GA population evolution.
    - 2D scatter if 2 genes (η_comp, η_turb)
    - 3D scatter if 3 genes (η_comp, η_turb, η_comb)
    """

    n_genes = ga.n_genes

    if n_genes not in [2, 3]:
        print(f"--Plot skipped: only supported for 2 or 3 genes (got {n_genes}).")
        return

    # Collect all population data across generations
    data = []
    for gen_idx, pop in enumerate(ga.population_history):
        row = {"generation": gen_idx}
        if n_genes >= 1:
            row["eta_comp"] = pop[:, 0].numpy()
        if n_genes >= 2:
            row["eta_turb"] = pop[:, 1].numpy()
        if n_genes == 3:
            row["eta_comb"] = pop[:, 2].numpy()

        df_gen = pd.DataFrame(row)
        data.append(df_gen)

    df = pd.concat(data, ignore_index=True)

    # ========== 2D case ==========
    if n_genes == 2:
        plt.figure(figsize=(7, 6))
        sns.scatterplot(
            data=df, x="eta_comp", y="eta_turb",
            hue="generation", palette="viridis",
            s=60, alpha=0.8, edgecolor="white"
        )

        if true_health is not None:
            plt.scatter(true_health[0], true_health[1],
                        c="lime", s=120, marker="X", label="True")
        plt.scatter(best_solution[0].item(), best_solution[1].item(),
                    c="red", s=120, marker="*", label="Best")

        plt.xlabel("η_comp (Compressor)")
        plt.ylabel("η_turb (Turbine)")
        plt.title(f"Population Evolution (Sensors: {ga.sensors})")
        plt.legend()
        plt.grid(True)
        plt.show()

    # ========== 3D case ==========
    elif n_genes == 3:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        sc = ax.scatter(
            df["eta_comp"], df["eta_turb"], df["eta_comb"],
            c=df["generation"], cmap="viridis", s=40, alpha=0.8
        )

        # Add markers for true and best solutions
        if true_health is not None:
            ax.scatter(true_health[0], true_health[1], true_health[2],
                       c="lime", s=100, marker="X", label="True")
        ax.scatter(best_solution[0].item(), best_solution[1].item(), best_solution[2].item(),
                   c="red", s=100, marker="*", label="Best")

        ax.set_xlabel("η_comp (Compressor)")
        ax.set_ylabel("η_turb (Turbine)")
        ax.set_zlabel("η_comb (Combustion)")
        ax.set_title(f"Population Evolution (Sensors: {ga.sensors})")

        plt.legend()
        fig.colorbar(sc, ax=ax, label="Generation")
        plt.show()



### Update plot to 3D version
def plot_ga_population_3d(ga, best_solution, true_health=None):
    """
    Interactive 3D visualization of population evolution for 3 genes.
    Works dynamically with Plotly (compatible marker symbols).
    """
    if ga.n_genes != 3:
        print("--Plot skipped: only supported for 3-gene problems.")
        return

    # Prepare data for all generations
    data = []
    for gen_idx, pop in enumerate(ga.population_history):
        for vals in pop:
            data.append({
                "η_comp": vals[0].item(),
                "η_turb": vals[1].item(),
                "η_comb": vals[2].item(),
                "Generation": gen_idx
            })
    df = pd.DataFrame(data)

    # 3D scatter for population
    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=df["η_comp"],
        y=df["η_turb"],
        z=df["η_comb"],
        mode="markers",
        marker=dict(
            size=4,
            color=df["Generation"],
            colorscale="Viridis",
            opacity=0.8,
            colorbar=dict(title="Generation")
        ),
        name="Population"
    ))

    # True point (green X)
    if true_health is not None:
        fig.add_trace(go.Scatter3d(
            x=[true_health[0]],
            y=[true_health[1]],
            z=[true_health[2]],
            mode="markers",
            marker=dict(size=10, color="lime", symbol="x"),
            name="True"
        ))

    # Best solution (red diamond)
    fig.add_trace(go.Scatter3d(
        x=[best_solution[0].item()],
        y=[best_solution[1].item()],
        z=[best_solution[2].item()],
        mode="markers",
        marker=dict(size=10, color="red", symbol="diamond"),
        name="Best"
    ))

    fig.update_layout(
        title=f"3D Population Evolution (Sensors: {ga.sensors})",
        scene=dict(
            xaxis_title="η_comp (Compressor)",
            yaxis_title="η_turb (Turbine)",
            zaxis_title="η_comb (Combustion)",
            xaxis=dict(range=[0.95, 1.02]),
            yaxis=dict(range=[0.95, 1.02]),
            zaxis=dict(range=[0.95, 1.02])
        ),
        width=850,
        height=700,
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(x=0.02, y=0.98)
    )

    fig.show()


