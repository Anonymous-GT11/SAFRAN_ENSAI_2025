import numpy as np
import pandas as pd

import random


class Scenario:
    """
    General Class Scenario.
    A Scenario is actually a degradation scenario of the health indicator of a turboreactor. It consists in a list of health indicators.

    General Attributes:
        speed : str, speed of degradation, can be slow or fast
        indicators : str, which module are affected by the degradation. Can be combustion, compressor or both.
        list_compressor: List(float), list of degradation of the health indicator for the compressor
        list_combustion: List(float), list of degradation of the health indicator for the combustion chamber
    """

    def __init__(self, speed: str = "slow", indicators: str = "both"):
        """
        Initialization function

        Parameters:
            speed : str, speed of degradation, can be slow or fast
            indicators : str, which module are affected by the degradation. Can be combustion, compressor or both.

        Output:
            None
        """
        possible_speed = ["slow", "fast"]
        possible_indicators = ["both", "compressor", "combustion"]

        if speed not in possible_speed:
            raise ValueError(f"Wrong speed. Possible values are 'slow' or 'fast'")
        if indicators not in possible_indicators:
            raise ValueError(
                f"Wrong indicators. Possible values are 'both', 'compressor' or 'combustion'"
            )

        self.speed = speed
        self.indicators = indicators

    def generate_degradation(
        self, length: int = 10, init_compressor: float = 1, init_combustion: float = 1
    ):
        """
        Function that generate a list of health indicators for the compressor, the combustion chamber or both depending
        on the value of self.indicators

        Parameters:
            length : int, length of the list of degradation we want to output. Default to 10
            init_compressor : float, initial value for the health of the compressor. Default to 1
            init_combustion : float, initial value for the health of the combustion chamber. Default to 1

        Output:
            None
        """
        if length < 0:
            raise ValueError(f"Negative length")
        if not 0.95 <= init_compressor <= 1.05:
            raise ValueError(
                f"init_compressor too low or too high. It must be between 0.95 and 1.05"
            )
        if not 0.95 <= init_combustion <= 1.05:
            raise ValueError(
                f"init_combustion too low or too high. It must be between 0.95 and 1.05"
            )

        if self.speed == "slow":
            step = 0.00005
        elif self.speed == "fast":
            step = 0.001

        if self.indicators == "compressor":
            self.list_compressor = [init_compressor - i * step for i in range(length)]

            return self.list_compressor

        elif self.indicators == "combustion":
            self.list_combustion = [init_combustion - i * step for i in range(length)]

            return self.list_combustion

        elif self.indicators == "both":
            self.list_compressor = [init_compressor - i * step for i in range(length)]
            self.list_combustion = [init_combustion - i * step for i in range(length)]

            return self.list_compressor, self.list_combustion

        return None


def simulate_degradation_trajectory(
    speed='normal',        # 'slow', 'normal', 'fast'
    waterwash='none',      # 'none', 'rare', 'frequent'
    size=100,                 # longueur de la trajectoire
    seed=None              # pour reproductibilité
):
    if seed is not None:
        np.random.seed(seed)

    # Vitesse de dégradation
    speed_params = {
        'slow':   {'mean_slope': -0.00005, 'std_slope': 0.00001},
        'normal': {'mean_slope': -0.0001, 'std_slope': 0.00002},
        'fast':   {'mean_slope': -0.0004, 'std_slope': 0.00005}
    }
    slope_mean = speed_params[speed]['mean_slope']
    slope_std = speed_params[speed]['std_slope']

    # Probabilité de waterwash
    waterwash_probs = {
        'none': 0.0,
        'rare': 0.0002,
        'frequent': 0.01
    }
    waterwash_prob = waterwash_probs[waterwash]

    # Récupération liée au waterwash
    recovery_mean = 0.0015
    recovery_std = 0.00002

    # Initialisation de la trajectoire
    current_val1 = np.random.normal(1.0, 0.005)
    current_val2 = np.random.normal(1.0, 0.005)
    trajectory = [[current_val1, current_val2]]

    for _ in range(1, size):
        # Appliquer une pente avec bruit
        slope1 = np.random.normal(slope_mean, slope_std)
        slope2 = np.random.normal(slope_mean, slope_std)

        # Mise à jour avec bruit
        current_val1 += slope1 + np.random.normal(0, 0.0002)
        current_val2 += slope2 + np.random.normal(0, 0.0002)

        # Vérification de waterwash
        if np.random.rand() < waterwash_prob:
            current_val1 += np.random.normal(recovery_mean, recovery_std)
            current_val2 += np.random.normal(recovery_mean, recovery_std)

        # Clamp les valeurs (optionnel)
        current_val1 = np.clip(current_val1, 0.95, 1.02)
        current_val2 = np.clip(current_val2, 0.95, 1.02)

        trajectory.append([current_val1, current_val2])

    return trajectory


def generate_multiple_trajectories(n_trajectories=100, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    speeds = ['slow', 'normal', 'fast']
    waterwashes = ['none', 'rare', 'frequent']
    trajectories = []

    for _ in range(n_trajectories):
        speed = random.choice(speeds)
        waterwash = random.choice(waterwashes)
        size = random.randint(80, 120)  # longueur aléatoire autour de 100

        traj = simulate_degradation_trajectory(speed=speed, waterwash=waterwash, size=size)
        trajectories.append({
            'speed': speed,
            'waterwash': waterwash,
            'size': size,
            'trajectory': traj
        })

    return trajectories

