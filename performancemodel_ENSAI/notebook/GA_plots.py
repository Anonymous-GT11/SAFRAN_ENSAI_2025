import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Optional
from mpl_toolkits.mplot3d import Axes3D
import math


class GeneticAlgorithmPlaceholder:
    def __init__(self, *args, **kwargs):
        self.best_fitness_history = []
        self.population_history = []
        self.population_sd_history = []
        self.avg_fitness_history = []
        self.sensors = []
        self.n_genes = 2
    def evaluate_population(self, population: torch.Tensor) -> torch.Tensor:
        
        return torch.zeros(population.shape[0])


# POPULATION EVOLUTION (2D and 3D)
def plot_ga_population_evolution(ga: 'GeneticAlgorithmPlaceholder', best_solution: torch.Tensor, true_health: Optional[Tuple] = None):
    """
    Visualize GA population evolution across generations.
    - 2D scatter if 2 genes (e.g., η_comp, η_turb)
    - 3D scatter if 3 genes (e.g., η_comp, η_turb, η_comb)
    """

    n_genes = ga.n_genes

    if n_genes not in [2, 3]:
        print(f"--Plot skipped: Evolution visualization only supports 2 or 3 genes (got {n_genes}).")
        return

    # Collect all population data across generations
    data = []
    gene_labels = [f"η_g{i+1}" for i in range(n_genes)]
    if n_genes == 2:
        gene_labels = ["η_comp", "η_turb"]
    elif n_genes == 3:
        gene_labels = ["η_comp", "η_turb", "η_comb"]

    for gen_idx, pop in enumerate(ga.population_history):
        pop_data = {"generation": gen_idx}
        for i in range(n_genes):
            pop_data[gene_labels[i]] = pop[:, i].cpu().numpy()
        
        df_gen = pd.DataFrame(pop_data)
        data.append(df_gen)

    df = pd.concat(data, ignore_index=True)

    # 2D 
    if n_genes == 2:
        plt.figure(figsize=(8, 7))
        sns.scatterplot(
            data=df, x=gene_labels[0], y=gene_labels[1],
            hue="generation", palette="viridis",
            s=60, alpha=0.8, edgecolor="white", legend=False
        )

        if true_health is not None:
            plt.scatter(true_health[0], true_health[1], c="lime", s=150, marker="X", label="True Health", edgecolor="black")
        plt.scatter(best_solution[0].item(), best_solution[1].item(),
                    c="red", s=150, marker="*", label="Best GA Solution", edgecolor="black")

        plt.xlabel(r"$" + gene_labels[0] + r"$")
        plt.ylabel(r"$" + gene_labels[1] + r"$")
        plt.title(f"Population Evolution (Generations 0 to {len(ga.population_history)-1})")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.show()

    # 3D 
    elif n_genes == 3:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        # Display of 3D data helps visualize convergence in multi-dimensional search spaces 



        sc = ax.scatter(
            df[gene_labels[0]], df[gene_labels[1]], df[gene_labels[2]],
            c=df["generation"], cmap="viridis", s=40, alpha=0.7
        )

        # Add markers for true and best solutions
        if true_health is not None and len(true_health) >= 3:
            ax.scatter(true_health[0], true_health[1], true_health[2],
                        c="lime", s=150, marker="X", label="True Health", edgecolor="black")
        ax.scatter(best_solution[0].item(), best_solution[1].item(), best_solution[2].item(),
                    c="red", s=150, marker="*", label="Best GA Solution", edgecolor="black")

        ax.set_xlabel(r"$" + gene_labels[0] + r"$")
        ax.set_ylabel(r"$" + gene_labels[1] + r"$")
        ax.set_zlabel(r"$" + gene_labels[2] + r"$")
        ax.set_title(f"3D Population Evolution (Sensors: {', '.join(ga.sensors)})")

        fig.colorbar(sc, ax=ax, label="Generation")
        ax.legend()
        plt.show()


# FITNESS CONVERGENCE (Best and Average Fitness)
def plot_fitness_convergence(ga: 'GeneticAlgorithmPlaceholder'):
    """
    Plots the convergence of best fitness and average fitness across generations.
    The gap between Best and Average fitness indicates population diversity.
    """

    generations = list(range(len(ga.best_fitness_history)))
    df = pd.DataFrame({
        "Generation": generations,
        "Best Fitness": ga.best_fitness_history,
        "Average Fitness": ga.avg_fitness_history
    })

    plt.figure(figsize=(9, 5))
    
    # Plot Best Fitness
    sns.lineplot(data=df, x="Generation", y="Best Fitness", marker="o", linewidth=2.5, color="royalblue", label="Best Fitness")
    
    # Plot Average Fitness
    sns.lineplot(data=df, x="Generation", y="Average Fitness", marker="^", linewidth=1.5, color="darkorange", linestyle='--', label="Average Fitness")
    
    sns.despine()
    plt.grid(True, alpha=0.3)
    
    plt.title("Fitness Convergence over Generations (Log-L2 Metric)", fontsize=14, weight="bold")
    plt.xlabel("Generation")
    plt.ylabel("Fitness (Higher is Better)")
    plt.legend()
    plt.show()


# DIVERSITY AND FINAL DISTRIBUTION
def plot_diversity_and_distribution(ga: 'GeneticAlgorithmPlaceholder'):
    """
    Plots the convergence of population standard deviation (diversity) 
    and the kernel density estimation (KDE) of the final population distribution.
    """
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # --- Subplot 1: SD Convergence (Diversity Loss) ---
    generations = list(range(len(ga.population_sd_history)))
    df_sd = pd.DataFrame({
        "Generation": generations,
        "Population Standard Deviation": ga.population_sd_history
    })
    
    sns.lineplot(ax=axes[0], data=df_sd, x="Generation", y="Population Standard Deviation", 
                 marker="o", linewidth=2.5, color="darkgreen")
    axes[0].set_title("Population Diversity (Avg SD) over Generations")
    axes[0].set_xlabel("Generation")
    axes[0].set_ylabel("Average Standard Deviation (Lower is Tighter)")
    axes[0].grid(True, alpha=0.3)
    
    # --- Subplot 2: Final Population Distribution (KDE) ---
    if ga.population_history and ga.n_genes <= 3:
        final_population = ga.population_history[-1]
        
        gene_labels = [f"η_g{i+1}" for i in range(ga.n_genes)]
        if ga.n_genes == 2: gene_labels = ["η_comp", "η_turb"]
        elif ga.n_genes == 3: gene_labels = ["η_comp", "η_turb", "η_comb"]
            
        data_dist = {label: final_population[:, i].cpu().numpy() for i, label in enumerate(gene_labels)}
        df_dist = pd.DataFrame(data_dist)

        axes[1].set_title(f"Final Population Distribution (Gen {len(ga.population_history) - 1})")
        axes[1].set_xlabel("Efficiency Value")
        
        # Plot KDE/Histogram for each parameter
        palette = sns.color_palette("Set2", ga.n_genes)
        for i, label in enumerate(gene_labels):
            sns.kdeplot(ax=axes[1], data=df_dist[label], fill=True, label=r"$" + label + r"$", color=palette[i], linewidth=2)
        
        axes[1].legend()

    plt.tight_layout()
    plt.show()


# FINAL FITNESS HEATMAP (2D Only)
def plot_final_fitness_map(ga: 'GeneticAlgorithmPlaceholder', best_solution: torch.Tensor, true_health: Optional[Tuple] = None):
    """
    Plots the final population in the 2D search space, colored and sized by their fitness.
    Only supported for n_genes=2.
    """
    if ga.n_genes != 2:
        print(f"--Plot skipped: Final Fitness Map only supports 2 genes (got {ga.n_genes}).")
        return
        
    if not ga.population_history:
        print("Error: Population history is empty. Run the evolve method first.")
        return

    final_population = ga.population_history[-1]
    # Re-evaluate fitness to ensure consistency (needed if plotting is done outside the evolve loop)
    final_fitness_values = ga.evaluate_population(final_population)

    gene_labels = ["η_comp", "η_turb"]

    data = {
        gene_labels[0]: final_population[:, 0].cpu().numpy(),
        gene_labels[1]: final_population[:, 1].cpu().numpy(),
        "Fitness": final_fitness_values.cpu().numpy()
    }
    df = pd.DataFrame(data)

    plt.figure(figsize=(8, 7))

    sns.scatterplot(
        data=df,
        x=gene_labels[0],
        y=gene_labels[1],
        hue="Fitness",
        palette="magma",
        size="Fitness",
        sizes=(20, 300), 
        alpha=0.8,
        edgecolor="black",
        legend=True
    )

    if true_health is not None and len(true_health) >= 2:
        plt.scatter(true_health[0], true_health[1], c="cyan", s=250, marker="X", label="True Health", edgecolor="black", linewidth=1.5)

    plt.scatter(best_solution[0].item(), best_solution[1].item(),
                c="gold", s=300, marker="*", label="Best GA Solution", edgecolor="black", linewidth=1.5)

    plt.xlabel(r"$" + gene_labels[0] + r"$ (Compressor Efficiency)")
    plt.ylabel(r"$" + gene_labels[1] + r"$ (Turbine Efficiency)")
    plt.title(f"Final Population Colored by Fitness (Generation {len(ga.population_history) - 1})")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()
