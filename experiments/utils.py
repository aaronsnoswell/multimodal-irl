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


def median_ci(values, confidence_level=0.95):
    """Compute median and approximate confidence interval for a list of values
    
    The method of computing the CI is taken from
    https://www.ucl.ac.uk/child-health/short-courses-events/about-statistical-courses/research-methods-and-statistics/chapter-8-content-8
    
    Args:
        values (list): List of float
        confidence_level (float): Confidence level
    
    Returns:
        (float): Lower approximate confidence interval
        (float): Median value
        (float): Upper approximate confidence interval
    """
    median = np.median(values)
    num_repeats = len(values)

    # Compute Z-factor
    critical_value = 1.0 - confidence_level
    z_factor = norm().ppf(1 - critical_value / 2)

    # Compute CI rankings
    low_ci_rank = int(round(num_repeats / 2 - z_factor * np.sqrt(num_repeats) / 2))
    high_ci_rank = int(round(1 + num_repeats / 2 + z_factor * np.sqrt(num_repeats) / 2))
    values_sorted = sorted(values)

    return values_sorted[low_ci_rank], median, values_sorted[high_ci_rank]
