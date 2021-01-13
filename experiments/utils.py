import os
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


def get_num_workers(num_jobs, requested_workers=None):
    """Figure out how many workers to use
    
    Args:
        num_jobs (int): Number of jobs we have to complete
        requested_workers (int): Number of workers we would like to spin up
    
    Returns:
        (int): Number of workers it is feasible to spin up based on OS scheduling,
            and/or the size of our workload
    """

    # Try and determine how many CPUs we are allowed to use
    num_cpus = (
        len(os.sched_getaffinity(0))
        # Ask the (linux) OS how many CPUs wer are scheduled to use
        if "sched_getaffinity" in dir(os)
        # If we can't find our scheduled number of CPUs, just use one less than the
        # system's physical socket count - leave one for GUI, bookkeeping etc.
        else os.cpu_count() - 1
    )

    print(f"META: {num_cpus} CPUs available")
    if requested_workers is not None:
        num_workers = min(num_cpus, num_jobs, requested_workers)
    else:
        num_workers = min(num_cpus, num_jobs)

    return num_workers


def replicate_config(cfg, num_times=1):
    """Replicate a config dictionary some number of times
    
    Args:
        cfg (dict): Base config dictionary
        num_times (int): Number of repeats
    
    Returns:
        (list): List of duplicated config dictionaries, with an added 'replicate' field.
    """
    # Repeat configuration settings across replicates
    configs = []
    for replicate in range(num_times):
        _config = cfg.copy()
        _config.update({"replicate": replicate})
        configs.append(_config)
    return configs


def mongo_config(mongodb_config_file="mongodb-config.txt"):
    """Attempt to load the MongoDB URL from a text file
    
    Args:
        mongodb_config_file (str): Path to a text file containing the MongoDB URL
    
    Returns:
        (str): MongoDB URL, or default URL if path wasn't found
    """
    mongodb_url = "localhost:27017"
    if os.path.exists(mongodb_config_file):
        with open(mongodb_config_file, "r") as file:
            mongodb_url = file.readline()
    return mongodb_url


def gt_responsibility_matrix(k, rpm):
    """Helper to get a ground truth responsibility matrix
    
    Args:
        k (int): Number of clusters
        rpm (int): Number of rollouts per cluster
    
    Returns:
        (numpy array): Block-diagonal ground truth responsibility matrix
    """
    return np.concatenate([np.repeat([np.eye(k)[r, :]], rpm, 0) for r in range(k)], 0,)
