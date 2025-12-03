
import torch
from typing import Tuple, List, Optional, Callable
import numpy as np
import math

class GeneticAlgorithm:
    """
    Fully Parameterized Genetic Algorithm with Log-L2 Fitness and Adaptive Mutation.
    """

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


    def initialize_population(self) -> torch.Tensor:
        population = torch.zeros((self.pop_size, self.n_genes))
        for i in range(self.n_genes):
            min_val, max_val = self.bounds[i]
            population[:, i] = torch.rand(self.pop_size) * (max_val - min_val) + min_val
        return population


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


    def selection(self, population: torch.Tensor, fitness_values: torch.Tensor, n_parents: int) -> torch.Tensor:
        parents = torch.zeros((n_parents, self.n_genes))
        for i in range(n_parents):
            candidates_idx = torch.randint(0, self.pop_size, (self.tournament_size,))
            best_idx = candidates_idx[torch.argmax(fitness_values[candidates_idx])]
            parents[i] = population[best_idx]
        return parents


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
            
        previous_best_fitness = self.best_fitness_history[generation - self.stagnation_generations]
        improvement = current_best_fitness - previous_best_fitness
        
        if improvement < self.min_improvement:
            print(f"\nSTOPPING: Stagnation detected. Improvement ({improvement:.4f}) over {self.stagnation_generations} generations is below threshold ({self.min_improvement:.4f}).")
            return True
        return False


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
                gene_display = best_individual[:3].tolist() 
                print(f"Gen {generation}: "
                      f"Best Fit = {current_best_fitness:.2f}, "
                      f"Avg Fit = {self.avg_fitness_history[-1]:.2f}, "
                      f"Pop SD = {self.population_sd_history[-1]:.4f}, "
                      f"Genes = [{', '.join(f'{g:.4f}' for g in gene_display)}]")

        final_fitness = self.evaluate_population(population)
        best_idx = torch.argmax(final_fitness)
        best_solution = population[best_idx]

        return best_solution, self.best_fitness_history

print("✅ GA.py overwritten with corrected indentation.")
