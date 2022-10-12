import os
import numpy as np


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
        (str): MongoDB URL, or 'None' if the path wasn't found
    """
    mongodb_url = None
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
    return np.concatenate(
        [np.repeat([np.eye(k)[r, :]], rpm, 0) for r in range(k)],
        0,
    )


def geometric_distribution(p, num_points):
    """Compute a normalized geometric distribution

    Args:
        p (float): Geometric distribution parameter on range [0, 1]. 0 asymptotically
            approaches a uniform distribution, 1.0 approaches a point-mass distribution
            at the first point.
        num_points (int): Size of the support for this distribution

    Returns:
        (numpy array): Normalized probability of each of the num_points.
    """

    if p == 0.0:
        # Handle asymptotic edge case: p=0 is a uniform distribution
        return np.ones(num_points) / num_points

    # Prepare discrete X-axis samples
    x = np.arange(1, num_points + 1)

    # Apply geometric distribution to those points
    geom = lambda x: ((1 - p) ** (x - 1)) * p
    y = geom(x)

    # Normalize the distribution
    y /= np.sum(y)

    return y
