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


def mean_error_metric(resp_learned, resp_gt, error_mat):
    """Average an error metric (learned wrt. GT) across all demonstration path
    
    This metric is used by Choi and Kim, 2012 in
    Nonparametric Bayesian Inverse Reinforcement Learning for Multiple Reward Functions.
    
    Args:
        resp_learned (numpy array): |N|x|K_1| Learned path responsibility matrix
        resp_gt (numpy array): |N|x|K_2| Ground Truth path responsibility matrix
        
        error_mat (numpy array): evd[i, j] is the error metric for the learned model i
            wrt. true model j.
    
    Returns:
        (float): The average of the EVD for each trajectory
    """

    resp_learned = np.array(resp_learned)
    resp_gt = np.array(resp_gt)
    error_mat = np.array(error_mat)

    assert len(resp_gt) == len(
        resp_learned
    ), "Number of paths (rows in responsibility matrices) is not consistent."

    errors = []
    for path_idx in range(len(resp_gt)):
        learned_mode_weights = resp_learned[path_idx]
        gt_mode_weights = resp_gt[path_idx]

        err_sum = 0
        for (li, lw), (gi, gw) in it.product(
            enumerate(learned_mode_weights), enumerate(gt_mode_weights)
        ):
            err_sum += lw * gw * error_mat[li, gi]
        errors.append(err_sum)

    err_avg = np.mean(errors)
    return err_avg


def min_cost_flow_error_metric(
    resp_learned, resp_gt, error_mat, use_int=True, significant_figures=5
):
    """Minimum Cost Flow error metric proposed by us
    
    Reference for min cost flow solver: https://networkx.github.io/documentation/stable/reference/algorithms/generated/networkx.algorithms.flow.min_cost_flow.html?highlight=min_cost_flow
    
    Args:
        w1 (1D float array): Weights of the learned models
        w2 (1D float array): Weights of the ground-truth models
        error_mat (2D float array: ile[i, j] is the error for the learned model i wrt
            true model j
        
        use_int (bool): If true, convert parameters to integers using significant_figures
            to avoid floating point rounding errors that can prevent convergence.
        significant_figures (int): The solver for the underlying min cost flow network
            problem may not converge for certain floating point weights and capacities.
            A solution is to convert weights and capacities to integers by scaling them
            by a large constant. This achieves convergence at the cost of some accuracy.
            This arguments sets the significant figures (the order of magnitude used for
            the integer scaling). Values too large may lead to float overflow.
    
    Returns:
        (float): The Minimum cost flow error metric
        (dict): Flow dictionary describing how the score is computed
    """
    resp_learned = np.array(resp_learned)
    w_learned = np.sum(resp_learned, axis=0) / len(resp_learned)
    resp_gt = np.array(resp_gt)
    w_gt = np.sum(resp_gt, axis=0) / len(resp_gt)

    scale = 10 ** significant_figures if use_int else 1.0
    error_mat = np.array(error_mat) * scale
    w_learned = np.array(w_learned) * scale
    w_gt = np.array(w_gt) * scale

    # Convert edge values to integers to ensure convergence
    if use_int:
        error_mat = error_mat.astype(np.uint64)
        w_learned = w_learned.astype(np.uint64)
        w_gt = w_gt.astype(np.uint64)

        # Compensate for rounding errors
        w_learned_sum, w_gt_sum = w_learned.sum(), w_gt.sum()
        if w_learned_sum < w_gt_sum:
            w_learned[0] += w_gt_sum - w_learned_sum
        else:
            w_gt[0] += w_learned_sum - w_gt_sum

    start_demand = -1.0 * w_learned.sum()
    terminal_demand = 1.0 * w_gt.sum()
    dense_edge_capacities = 1.0 * w_learned.sum()

    lnames = ["l%d" % (i) for i in range(len(w_learned))]
    gnames = ["g%d" % (j) for j in range(len(w_gt))]

    G = nx.DiGraph()
    G.add_node("s", demand=start_demand)
    G.add_node("t", demand=terminal_demand)

    for i in range(len(w_learned)):
        G.add_edge("s", lnames[i], weight=0, capacity=w_learned[i])
    for j in range(len(w_gt)):
        G.add_edge(gnames[j], "t", weight=0, capacity=w_gt[j])

    for i in range(len(w_learned)):
        for j in range(len(w_gt)):
            G.add_edge(
                lnames[i],
                gnames[j],
                weight=error_mat[i][j],
                capacity=dense_edge_capacities,
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
    """Test metrics"""

    resp_gt = [
        [1.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 1.0],
    ]

    resp_learned = [
        [0.5, 0.5, 0.0],
        [0.5, 0.5, 0.0],
        [0.1, 0.5, 0.4],
        [0.1, 0.5, 0.4],
    ]

    # Learned wrt. Ground Truth
    error_mat = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]

    min_cost_flow, _ = min_cost_flow_error_metric(
        resp_learned, resp_gt, error_mat, use_int=True
    )
    print(min_cost_flow)

    mean_error = mean_error_metric(resp_learned, resp_gt, error_mat)
    print(mean_error)


if __name__ == "__main__":
    main()
