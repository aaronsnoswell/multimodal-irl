import copy
import warnings
import numpy as np
import itertools as it

from itertools import combinations, permutations

from sklearn.metrics import adjusted_mutual_info_score


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
    l1_combs = list(combinations(l1, num_elements))
    l2_perms = list(permutations(l2, num_elements))

    for c in l1_combs:
        for p in l2_perms:
            yield tuple(zip(p, c))


def ami(zij_a, zij_b):
    """Find Adjusted Mutual Information - a metric quantifying the clustering quality"""
    warnings.warn("TODO: Implement AMI for soft clusterings!")
    labels_a = np.argmax(zij_a, axis=1)
    labels_b = np.argmax(zij_b, axis=1)
    return adjusted_mutual_info_score(labels_a, labels_b)
