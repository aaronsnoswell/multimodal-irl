import copy
import warnings
import numpy as np
import itertools as it

from itertools import combinations, permutations

from explicit_env.soln import (
    value_iteration,
    q_from_v,
    OptimalPolicy,
    policy_evaluation,
)
from unimodal_irl.experiments.metrics import ile_evd


def exclusive_pairings(l1, l2):
    """Generate exclusive pairings of items from two sets
    
    N.b. sets do not need to be the same size
    
    Args:
        l1 (list): First set of items
        l2 (list): Second set of items
    
    Yields:
        (tuple): Tuple containing the next pairing of items from the list
    """

    num_elements = min(len(l1), len(l2))
    combs = list(combinations(l1, num_elements))
    perms = list(permutations(l2, num_elements))

    for c in combs:
        for p in perms:
            yield tuple(zip(c, p))



def soft_contingency_table(resp1, resp2):
    """Compute the soft contingency table for two responsibility matrices
    
    Args:
        resp1 (numpy array): N x K_1 responsibility matrix - each row is a probability
            vector for one of the N items belonging to each of K_1 modes
        resp1 (numpy array): N x K_2 responsibility matrix - each row is a probability
            vector for one of the N items belonging to each of K_2 modes
    
    Returns:
        (numpy array): K_1 x K_2 contingency table for soft clustering - defined as
            resp1^T resp2
    """
    return resp1.T @ resp2


def responsibility_entropy(resp):
    """Compute the entropy of a responsibility matrix
    
    Computes H(U), where U is a soft clustering, defined by a responsibility matrix.
    
    Args:
        resp (numpy array): N x K responsibility matrix - each row is a probability
            vector for one of the N items belonging to each of the K modes
    
    Returns:
        (float): Entropy of the responsibilty matrix
    """
    N, K = resp.shape
    np.testing.assert_almost_equal(
        np.sum(resp), N
    ), "Responsibility matrix is not normalized"
    cluster_masses = np.sum(resp, axis=0)
    return -1.0 * np.sum(
        [a_i / N * np.log(a_i / N) if a_i > 0.0 else 0.0 for a_i in cluster_masses]
    )


def contingency_relative_entropy(cont):
    """Computes the relative entropy of one clustering wrt. another
    
    Computes H(U | V), where U is the clustering with clusters along rows of the table,
    and V is the clustering along the columns of the table. Clusters can be hard or
    soft.
    
    Args:
        cont (numpy array): K_1 x K_2 contingency table
    
    Returns:
        (float): Relative entropy of U given V.
    """
    K1, K2 = cont.shape
    N = np.sum(cont)
    cluster_2_masses = np.sum(cont, axis=0)
    return -1.0 * np.sum(
        [
            cont[i, j] / N * np.log((cont[i, j] / N) / (cluster_2_masses[j] / N))
            if cont[i, j] > 0.0
            else 0.0
            for j in range(K2)
            for i in range(K1)
        ]
    )


def contingency_joint_entropy(cont):
    """Computes the joint entropy of two clusterings
    
    Computes H(U, V), where U, V are two clusterings corresponding to the rows/columns
    of the contingency table. Clusters can be hard or soft.
    
    Args:
        cont (numpy array): K_1 x K_2 contingency table
    
    Returns:
        (float): Joint entropy of U, V
    """
    K1, K2 = cont.shape
    N = np.sum(cont)
    return -1.0 * np.sum(
        [
            cont[i, j] / N * np.log(cont[i, j] / N) if cont[i, j] > 0.0 else 0.0
            for j in range(K2)
            for i in range(K1)
        ]
    )


def contingency_mutual_info(cont):
    """Computes the mutual information between two clusterings
    
    Computes I(U, V), where U, V are two clusterings corresponding to the rows/columns
    of the contingency table. Clusters can be hard or soft.
    
    Args:
        cont (numpy array): K_1 x K_2 contingency table
    
    Returns:
        (float): Mutual information between U and V.
    """
    K1, K2 = cont.shape
    N = np.sum(cont)
    cluster_1_masses = np.sum(cont, axis=1)
    cluster_2_masses = np.sum(cont, axis=0)
    return np.sum(
        [
            cont[i, j]
            / N
            * np.log(
                (cont[i, j] / N) / (cluster_1_masses[i] * cluster_2_masses[j] / N ** 2)
            )
            if cont[i, j] > 0.0
            else 0.0
            for j in range(K2)
            for i in range(K1)
        ]
    )


def normalized_information_distance(resp1, resp2):
    """Compute Normalized Information Distance (NID) between soft clusterings
    
    Args:
        resp1 (numpy array): N x K_1 responsibility matrix
        resp2 (numpy array): N x K_2 responsibility matrix
    
    Returns:
        (float): Normalized information distance on range [0, 1]. Lower values indicate
            more agreement between the two clusterings.
    """
    cont = soft_contingency_table(resp1, resp2)
    return 1.0 - (
        contingency_mutual_info(cont)
        / max(responsibility_entropy(resp1), responsibility_entropy(resp2))
    )


def adjusted_normalized_information_distance(resp1, resp2, num_samples=1000):
    """Compute Adjusted Normalized Information Distance (aNID) between soft clusterings
    
    This metric is the same as the NID, however it controls for random chance, removing
    the positive correlations NID α K_1, NID α K_2. We use a random clustering model
    that samples rows of each responsibility matrix from uniform Dirichlet
    distributions.
    
    Args:
        resp1 (numpy array): N x K_1 responsibility matrix
        resp2 (numpy array): N x K_2 responsibility matrix
        num_samples (int): Number of samples to use for computing expectations
    
    Returns:
        (float): Adjusted Normalized information distance stochastically normalized to
            the range [0, 1]. Lower values indicate more agreement between the two
            clusterings.
    """

    N, K1 = resp1.shape
    _, K2 = resp2.shape

    # Find expectations with fixed N, K1, K2
    mis = []
    for _ in range(num_samples):
        r1 = np.random.dirichlet([1.0 / K1] * K1, size=N)
        r2 = np.random.dirichlet([1.0 / K2] * K2, size=N)
        cont = soft_contingency_table(r1, r2)
        mis.append(contingency_mutual_info(cont))
    e_mi = np.mean(mis)

    cont = soft_contingency_table(resp1, resp2)
    return 1.0 - (
        (contingency_mutual_info(cont) - e_mi)
        / (max(responsibility_entropy(resp1), responsibility_entropy(resp2)) - e_mi)
    )
