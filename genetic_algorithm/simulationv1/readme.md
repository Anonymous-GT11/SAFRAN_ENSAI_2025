
# Genetic Algorithm for Gas Turbine Health Indicator Estimation

Experiment Framework on a single measurment.

This notebook implements a **Genetic Algorithm (GA)** to solve the **inverse problem** in gas turbine engine health monitoring.


### Systematic Experiments with Multiple Scenarios

This notebook includes:
1. **All original plots** (convergence, 2D/3D search space, ablation)
2. **Multi-scenario experiments** (1-4 contexts × 1-3 sensors)
3. **Clean and noisy data** comparison
4. **Uses existing original and noisy CSV file**

---

## 0. The Problem Formulation
- **Given**: Sensor measurements (temperatures, pressures)
- **Find**: Internal component health indicators (efficiencies, mass flows)


###  0.1. Inverse Problem 

In gas turbine health monitoring, we face an **inverse problem**:

| Direction | Problem | Difficulty |
|----------|---------|------------|
| **Forward** | Health indicators → Sensor measurements | Easy (simulator available) |
| **Inverse** | Sensor measurements → Health indicators | Difficult (ill-posed, non-linear) |

---

### 0.2 Mathematical Formulation

#### Objective
Find health indicators $\mathbf{x}$ that minimize the difference between predicted and measured sensor values.


In other words,  estimate the vector of health indicators  
$\mathbf{x} = (x_1, \dots, x_n)$   that best explains the observed sensor measurements.

Let $m$ denote the total number of compared measurement components  
(e.g. **number of sensors $\times$ number of operating contexts**).

We define the objective function as the **normalized root mean squared error (RMSE)** of the standardized residuals:
<!-- This is the latex block
$$
\min_{\mathbf{x} \in \mathcal{X}}
f(\mathbf{x})
=
\sqrt{
\frac{1}{m}
\sum_{i=1}^{m}
\left(
\frac{
y_{\text{sim},i}(\mathbf{x}) - y_{\text{meas},i}
}{\sigma_i}
\right)^2
}
$$
-->

![objective-function](https://latex.codecogs.com/svg.image?\min_{\mathbf{x}\in\mathcal{X}}f(\mathbf{x})\sqrt{\frac{1}{m}\sum_{i=1}^{m}\left(\frac{y_{\text{sim},i}(\mathbf{x})-y_{\text{meas},i}}{\sigma_i}\right)^2})



#### Compact Vector Form

Equivalently, the objective can be written using the Euclidean norm:
<!--
$$
f(\mathbf{x})
=
\sqrt{\frac{1}{m}}
\left\|
\frac{
\mathbf{y}_{\text{sim}}(\mathbf{x}) - \mathbf{y}_{\text{meas}}
}{
\boldsymbol{\sigma}
}
\right\|_2
$$
-->
![objective-function](https://latex.codecogs.com/svg.image?f(\mathbf{x})=\sqrt{\frac{1}{m}}\left\|\frac{\mathbf{y}_{\text{sim}}(\mathbf{x})-\mathbf{y}_{\text{meas}}}{\boldsymbol{\sigma}}\right\|_2)



### where

- $\mathbf{x} = (x_1, \dots, x_n)$: health indicators to estimate  
- $\mathbf{y}_{\text{sim}}(\mathbf{x})$: sensor values predicted by the simulator  
- $\mathbf{y}_{\text{meas}}$: measured sensor values  
- $\boldsymbol{\sigma}$: per-sensor normalization factors (e.g. standard deviations),   used to handle heterogeneous sensor scales (temperatures K, pressures pa)  
- $m$: total number of residual terms (sensors $\times$ contexts)  
- $\mathcal{X}$: feasible parameter space  



### Feasible Region (Bounds)
<!--
$$
x_j^{\text{lower}} \le x_j \le x_j^{\text{upper}},
\quad j = 1, \dots, n
$$
-->
![bounds-constraint](https://latex.codecogs.com/svg.image?x_j^{\text{lower}}\le%20x_j\le%20x_j^{\text{upper}},\quad%20j=1,\dots,n)


The bounds are enforced directly by the genetic algorithm through candidate  clipping.


---

### 0.3. Why an L2-Based Objective?

The objective function is based on a **scaled Euclidean (L2) norm** of the residuals between simulated and measured sensor values.

| Property | Benefit |
|--------|--------|
| Smooth and continuous | Produces a well-behaved optimization landscape |
| Quadratic error accumulation | Large mismatches are penalized more strongly than small ones |
| Rotation invariant | No directional bias in the residual space |
| Physically interpretable | Measures a distance in (normalized) measurement space |
| Statistical grounding | Corresponds to maximum likelihood under Gaussian noise assumptions |



### 0.4. Why Is the Inverse Problem Difficult?

Despite a well-defined objective function, the inversion remains challenging due to several intrinsic properties of gas turbine systems:

1. **Non-invertible physics**  
   The thermodynamic and aerodynamic relations embedded in the simulator do not admit a closed-form inverse mapping from sensor space to health space.

2. **Non-uniqueness of solutions**  
   Different combinations of health indicator degradations can produce very similar sensor measurements, leading to multiple admissible solutions.

3. **Sensor uncertainty and noise**  
   Measurement noise and modeling uncertainties blur the mapping between health states and sensor responses, further complicating the inversion.

These factors make the inverse problem **ill-posed and highly non-linear**, motivating the use of global, derivative-free optimization methods such as genetic algorithms.

---


### 0.5. Why Genetic Algorithm?
| Advantage | Explanation |
|-----------|-------------|
| Gradient-free | Simulator is a black box - no derivatives |
| Global search | Population explores multiple solutions |
| Escapes local minima | Crossover/mutation provide exploration |
| Handles constraints | Easy to enforce bounds |


### 0.6 Hyperparameter Choices and Tuning

| Parameter | Value | Explanation | How to Tune |
|-----------|-------|-------------|-------------|
| `population_size` | 40-100 | Larger = more exploration, slower | If stuck in local minima, increase |
| `tournament_size` | 0.1 | Selection pressure (higher = more greedy) | 10% .If diversity too low, decrease to 2 |
| `elitism_rate` | 0.01-0.05 | Best individuals preserved | Keep 2-5% of population |
| `crossover_rate` | 0.85 | Probability of combining parents | 0.7-0.9 typical |
| `blx_alpha` | 0.5 | Exploration range in crossover | Higher = more exploration |
| `mutation_rate` | 0.3 | Probability per gene | 0.1-0.5 typical |
| `mutation_strength_initial` | 0.15 | Starting σ (fraction of range) | If no progress, increase |
| `mutation_decay` | 0.98 | σ_t = σ_0 × λ^t | See detailed explanation below |
| `mutation_strength_min` | 0.001 | Floor for mutation | Should be ~1e-3 to 1e-4 for fine-tuning |
| `early_stop_generations` | 15-50 | Stop if no improvement | Higher if expecting slow convergence |




## 1 Our 3 Target Indicators

| Indicator | Component | Physical Meaning | Degradation Range |
|-----------|-----------|------------------|---------------------|
| `deg_CmpFan_s_mapWc_in` | Fan | Mass flow capacity | [-5%, +3%] |
| `deg_CmpH_s_mapEff_in` | HPC |  Compressor efficiency | [-5%, 0%] |
| `deg_TrbH_s_mapEff_in` | HPT | High Pressure Turbine efficiency | [-5%, 0%] |




**Why these three?**
- **Fan mass flow**: Affected by blade erosion, tip clearance increase, FOD damage
- **HPC efficiency**: Degrades due to fouling, blade erosion, seal wear
- **HPT efficiency**: Most critical - operates at highest temperatures, subject to creep, oxidation, thermal fatigue

## 2 Sensor Selection

We use **3 sensors at different condition**:

| Sensor | Description | Why Selected |
|--------|-------------|-------------|
| `HPC_Tin` | Compressor inlet temperature | Reflects upstream conditions |
| `LPT_Tin` | Turbine Low PRESSURE (inlet) temperature | Most sensitive to combustion/HPT changes |
| `HPC_Pout_st` | Compressor Hight Pressure (outlet) pressure | Indicates compression efficiency |




## 3. Simulator Cache Mechanism 

### Motivation
Evaluating the fitness of an individual requires running the physical simulator
(`decksmr_1forall`), which is computationally expensive.
During a Genetic Algorithm (GA) run, the **same individuals are often evaluated multiple times**
(elitism, convergence, duplicate offspring).

### Cache Principle
A cache is introduced to store simulator outputs keyed by the health state:


```python
key = tuple(np.round(individual, 10))
if key not in cache:
    cache[key] = run_simulator(individual)
```
As a result, repeated evaluations 

- do not increase the number of simulator calls.

- The total number of simulator executions therefore depends on the number of
unique health states explored, not on the total number of fitness evaluations.

### Effect on Simulator Calls

Let:

- **P** = population size  
- **G** = number of generations  
- **U** = number of **unique** individuals evaluated  

---

#### Without cache

Every fitness evaluation runs the simulator.
<!--
\[
\text{Calls}_{\text{no cache}} = P \times G 
\]
-->

![calls-no-cache](https://latex.codecogs.com/svg.image?\text{Calls}_{\text{nocache}}=P\times G)


**Example**  

- $P = 80$
- $G = 50$

<!--
\[
\text{Calls}_{\text{no cache}} = 80 \times 50 = 4000 $
\]
-->
![calls-no-cache-numeric](https://latex.codecogs.com/svg.image?\text{Calls}_{\text{nocache}}=80\times50=4000)


---

#### With cache

The simulator is executed **only once per unique individual**.

<!--
\[
\text{Calls}_{\text{cache}} = U \quad \text{with } U \le P \times G
\]
-->

![calls-cache-bound](https://latex.codecogs.com/svg.image?\text{Calls}_{\text{cache}}=U,\;U\le P\times G)


Where:
- duplicates caused by **elitism**
- repeated individuals due to **convergence**
- identical offspring

do **not** trigger new simulator runs.

---

#### Typical behavior in practice

As the GA converges:

- Population diversity decreases  
- Many individuals are re-evaluations of the same states  

So usually:
<!--
\[
U \ll P \times G
\]
-->
![u-much-less](https://latex.codecogs.com/svg.image?U\ll P\times G)


**Example (realistic)**  

- $P = 80$ , $G = 50$ 

- Unique individuals explored: \(U \approx 800\)

<!--
\[
\text{Calls}_{\text{cache}} \approx 800 \quad \text{instead of } 4000
\]
-->
![calls-cache-approx](https://latex.codecogs.com/svg.image?\text{Calls}_{\text{cache}}\approx800\;\ll\;4000)


---

#### Key takeaway

The cache changes the computational cost from:

<!--
\[
\mathcal{O}(P \times G) \;\; \rightarrow \;\; \mathcal{O}(U)
\]
-->
![complexity-reduction](https://latex.codecogs.com/svg.image?\mathcal{O}(P\times G)\rightarrow\mathcal{O}(U))


making the GA **significantly faster** once the population starts converging.
