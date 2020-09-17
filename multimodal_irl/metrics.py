import copy
import warnings
import numpy as np
import networkx as nx
import itertools as it

from itertools import combinations, permutations

from explicit_env.soln import (
    value_iteration,
    q_from_v,
    OptimalPolicy,
    policy_evaluation,
)
from unimodal_irl.experiments.metrics import ile_evd


def miles(w1, w2, ile, use_int=True, significant_figures=5):
    """MILES metric
    
    Reference for min cost flow solver: https://networkx.github.io/documentation/stable/reference/algorithms/generated/networkx.algorithms.flow.min_cost_flow.html?highlight=min_cost_flow
    
    Args:
        w1 (1D float array): Weights of the learned models
        w2 (1D float array): Weights of the ground-truth models
        ile (2D float array: ile[i, j] is the ILE for the learned model i wrt true
            model j
        
        use_int (bool): If true, convert parameters to integers using significant_figures
            to avoid floating point rounding errors that can prevent convergence.
        significant_figures (int): The solver for the underlying min cost flow network
            problem may not converge for certain floating point weights and capacities.
            A solution is to convert weights and capacities to integers by scaling them
            by a large constant. This achieves convergence at the cost of some accuracy.
            This arguments sets the significant figures (the order of magnitude used for
            the integer scaling). Values too large may lead to float overflow.
    
    Returns:
        (float): The MILESv2 metric
        (dict): Flow dictionary describing how the score is computed
    """
    scale = 10 ** significant_figures if use_int else 1.0
    ile = np.array(ile) * scale
    w1 = np.array(w1) * scale
    w2 = np.array(w2) * scale

    # Convert edge values to integers to ensure convergence
    if use_int:
        ile = ile.astype(np.uint64)
        w1 = w1.astype(np.uint64)
        w2 = w2.astype(np.uint64)

        # Compensate for rounding errors
        w1_sum, w2_sum = w1.sum(), w2.sum()
        if w1_sum < w2_sum:
            w1[0] += w2_sum - w1_sum
        else:
            w2[0] += w1_sum - w2_sum

    start_demand = -1.0 * w1.sum()
    terminal_demand = 1.0 * w2.sum()
    dense_edge_capacities = 1.0 * w1.sum()

    lnames = ["l%d" % (i) for i in range(len(w1))]
    gnames = ["g%d" % (j) for j in range(len(w2))]

    G = nx.DiGraph()
    G.add_node("s", demand=start_demand)
    G.add_node("t", demand=terminal_demand)

    for i in range(len(w1)):
        G.add_edge("s", lnames[i], weight=0, capacity=w1[i])
    for j in range(len(w2)):
        G.add_edge(gnames[j], "t", weight=0, capacity=w2[j])

    for i in range(len(w1)):
        for j in range(len(w2)):
            G.add_edge(
                lnames[i], gnames[j], weight=ile[i][j], capacity=dense_edge_capacities
            )

    flow_dict = nx.min_cost_flow(G)

    if use_int:
        # Scale flowdict back to floating point
        for n2, d in flow_dict.items():
            for n2 in d:
                d[n2] /= scale

    cost = nx.cost_of_flow(G, flow_dict) / scale
    return cost, flow_dict


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


def main():
    """Main function"""

    m2, flowdict = miles([0.7, 0.3], [0.3, 0.7], [[1, 4], [2, 1]])
    print(m2, flowdict)
    print()

    m2, flowdict = miles([0.7, 0.3], [0.3, 0.7], [[1, 4], [2, 1]], use_int=True)
    print(m2, flowdict)
    print()


if __name__ == "__main__":
    main()
