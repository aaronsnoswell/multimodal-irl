import numpy as np
from scipy.stats import norm


def mean_ci(values, confidence_level=0.95):
    """Compute mean and symmetric confidence interval for a list of values
    
    Args:
        values (list): List of float
        confidence_level (float): Confidence level
    
    Returns:
        (float): Lower confidence interval
        (float): Mean value
        (float): Upper confidence interval
    """
    mean = np.mean(values)
    std = np.std(values)
    num_repeats = len(values)

    # Compute Z-factor
    critical_value = 1.0 - confidence_level
    z_factor = norm().ppf(1 - critical_value / 2)

    # Compute CI
    ci = z_factor * std / np.sqrt(num_repeats)

    return mean - ci, mean, mean + ci

