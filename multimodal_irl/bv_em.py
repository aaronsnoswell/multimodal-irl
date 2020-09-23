"""Implements Babes-Vroman style EM MM-IRL Algorithms"""


import copy
import warnings
import numpy as np
import itertools as it

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from explicit_env.soln import value_iteration, q_from_v, BoltzmannExplorationPolicy
from unimodal_irl import sw_maxent_irl
from unimodal_irl.utils import pad_terminal_mdp, empirical_feature_expectations
from unimodal_irl.sw_maxent_irl import maxent_path_logprobs, maxent_log_partition, r_tau


def responsibilty_matrix_maxent(env, rollouts, reward_weights, mode_weights=None):
    """Compute responsibility matrix using MaxEnt distribution
    
    Args:
        env (gym.Env): Environment to solve for
        rollouts (list): List of rollouts
        reward_weights (list): List of reward weight vectors
        
        mode_weights (numpy array): Optional list of mode weights
    """

    zij = np.ones((len(rollouts), len(reward_weights)))

    if mode_weights is None:
        mode_weights = np.ones(len(reward_weights), dtype=float)

    for m, r in enumerate(reward_weights):
        # Use Maximum Entropy distribution for computing path likelihoods
        env._state_rewards = reward_weights[m]
        env_padded, rollouts_padded = pad_terminal_mdp(env, rollouts=rollouts)
        zij[:, m] = mode_weights[m] * np.exp(
            maxent_path_logprobs(
                env_padded,
                rollouts_padded,
                theta_s=env_padded.state_rewards,
                with_dummy_state=True,
            )
        )

    # Each path (row of the assignment matrix) gets a mass of 1 to allocate between
    # modes
    zij /= np.sum(zij, axis=1, keepdims=True)

    return zij


def mixture_ll_maxent(env, rollouts, zij, reward_weights):
    """Find the log-likelihood of a MaxEnt MM-IRL mixture mode"""

    env = copy.deepcopy(env)

    num_paths, num_modes = zij.shape

    num_states = len(env.states)
    num_actions = len(env.actions)

    # Find max path length
    if len(rollouts) == 1:
        max_path_length = min_path_length = len(rollouts[0])
    else:
        max_path_length = max(*[len(r) for r in rollouts])

    # Pre-compute the partition values for each reward parameter
    mode_log_partition_values = []
    for r in reward_weights:
        env._state_rewards = r
        env_padded, rollouts_padded = pad_terminal_mdp(env, rollouts=rollouts)
        mode_log_partition_values.append(
            maxent_log_partition(
                env_padded, max_path_length, env_padded.state_rewards, None, None, True,
            )
        )

    total_ll = 0
    # Sweep all paths
    for n in range(num_paths):

        path = rollouts[n]
        path_mode_probs = []
        for k in range(num_modes):

            mode_weight = np.sum(zij[:, k])
            mode_reward = reward_weights[k]
            mode_log_parition = mode_log_partition_values[k]

            env._state_rewards = mode_reward

            # Find likelihood of this path under this reward
            path_prob = np.exp(
                env.path_log_probability(path) + r_tau(env, path) - mode_log_parition
            )

            path_mode_probs.append(mode_weight * path_prob)

        total_ll += np.log(np.sum(path_mode_probs))

    return total_ll


def bv_em_maxent(
    env,
    rollouts,
    num_modes,
    initial_mode_weights=None,
    initial_reward_weights=None,
    max_iterations=100,
    min_weight_change=1e-6,
    after_estep=None,
    after_mstep=None,
    verbose=False,
):
    """Solve a multi-modal IRL problem using the Babes-Vroman EM alg with MaxEnt IRL
    
    Args:
        env (explicit_env.envs.IExplicitEnv): Environment defining the dynamics of the
            multi-modal IRL problem and the .reward_range parameter
        rollouts (list): List of demonstration trajectories, each a list of (s ,a)
            tuples
        num_modes (int): Number of modes to solve for - this is a hyper parameter of
            the algorithm
        
        initial_mode_weights (numpy array): Optional list of initial mode weights -
            if not provided, defaults to uniform.
        initial_reward_weights (list): List of initial mode reward weights (numpy
            arrays) - one for each mode. If not provided, these weights are sampled from
            a uniform distribution over possible reward values.
        max_iterations (int): Maximum number of EM iterations to perform
        min_weight_change (float): Stop EM iterations when the change in mode weights
            fall below this value
        after_mstep (function): Optional callback called after each E-Step. Function
            should accept as arguments the current responsibility matrix and reward
            weights
        after_mstep (function): Optional callback called after each M-Step. Function
            should accept as arguments the current responsibility matrix and reward
            weights
        verbose (bool): Print progress information
    
    Returns:
        (int): Number of iterations performed until convergence
        (numpy array): N x K numpy array representing path assignments to modes. One row
            for each path in the demonstration data, one column for each mode
        (numpy array): List of weights, one for each reward mode - this is equal to the
            normalized sum along the columns of the assignment matrix, and is provided
            for convenience only.
        (numpy array): K x |S| matrix of recovered state reward weights one for each
            mode
    """

    # Work with a copy of the env
    env = copy.deepcopy(env)

    env_padded, rollouts_padded = pad_terminal_mdp(env, rollouts=rollouts)

    # How many paths have we got?
    num_paths = len(rollouts)

    if initial_mode_weights is None:
        mode_weights = np.ones(num_modes, dtype=float) / num_modes
    else:
        assert (
            len(initial_mode_weights) == num_modes
        ), "Wrong number of initial mode weights passed"
        mode_weights = initial_mode_weights

    if initial_reward_weights is None:
        # Use uniform initial weights over range of valid reward values
        reward_weights = [
            np.random.uniform(
                low=env.reward_range[0], high=env.reward_range[1], size=len(env.states)
            )
            for _ in range(num_modes)
        ]
    else:
        assert (
            len(initial_reward_weights) == num_modes
        ), "Wrong number of initial reward weights passed"
        reward_weights = initial_reward_weights
        m2 = np.clip(reward_weights, *env.reward_range)
        if not np.all(m2 == reward_weights):
            warnings.warn(
                "Initial reward weights are outside valid reward ranges - clipping"
            )
        reward_weights = m2

    for _it in it.count():
        if verbose:
            print("EM Iteration {}".format(_it + 1), end="", flush=True)
            print(", Weights:{}".format(mode_weights), end="", flush=True)

        # E-step: Solve for responsibility matrix
        zij = responsibilty_matrix_maxent(env, rollouts, reward_weights, mode_weights)

        if after_estep is not None:
            after_estep(zij, reward_weights)

        # M-step: Update mode weights and reward estimates
        old_mode_weights = copy.copy(mode_weights)
        mode_weights = np.sum(zij, axis=0) / num_paths

        for m in range(num_modes):
            # Use MaxEnt IRL for computing reward functions
            theta_s, _, _ = sw_maxent_irl(
                rollouts_padded,
                env_padded,
                rs=True,
                rbound=env.reward_range,
                with_dummy_state=True,
                grad_twopoint=True,
                path_weights=zij[:, m],
            )
            reward_weights[m] = theta_s[:-1]

        delta = np.max(np.abs(old_mode_weights - mode_weights))
        if verbose:
            print(", Δ:{}".format(delta), flush=True)

        if after_mstep is not None:
            after_mstep(zij, reward_weights)

        if delta <= min_weight_change:
            if verbose:
                print("EM mode wights have converged, stopping", flush=True)
            break

        if _it == max_iterations - 1:
            if verbose:
                print("Reached maximum number of EM iterations, stopping", flush=True)
            break

    return (_it + 1), zij, mode_weights, np.array(reward_weights)


def init_from_feature_clusters(env, rollouts, num_modes, init="uniform", verbose=False):
    """Compute a MM-IRL initialization by clustering in feature space
    
    For now, only supports state-based reward features
    
    Args:
        env (explicit_env.IExplicitEnv): Environment defining dynamics
        rollouts (list): List of state-action demonstration rollouts
        num_modes (int): Number of modes to cluster into
        init (str): Clustering method, one of 'uniform', 'gmm', 'kmeans'
        verbose (bool): Print progress information
    
    Returns:
        (numpy array): |K| Vector of initial mode weights
        (numpy array): |K|x|S| Array of initial reward weights
    """

    # How many times to restart the initialization method?
    NUM_INIT_RESTARTS = 5000

    # Build feature matrix
    rollout_features = np.array(
        [empirical_feature_expectations(env, [r])[0] for r in rollouts]
    )

    if init == "uniform":

        # Don't do any up-front clustering, let the MM-IRL algorithm
        # determine a uniform initial clustering
        # TODO replace with actual soft clustering
        initial_reward_weights = None

        initial_mode_weights = np.ones(num_modes) / num_modes
    else:

        if init == "kmeans":

            # Initialize mode weights with K-Means (hard) clustering
            km = KMeans(n_clusters=num_modes, n_init=NUM_INIT_RESTARTS)
            hard_initial_clusters = km.fit_predict(rollout_features)
            soft_initial_clusters = np.zeros((len(rollouts), num_modes))
            for idx, clstr in enumerate(hard_initial_clusters):
                soft_initial_clusters[idx, clstr] = 1.0

        elif init == "gmm":

            # Initialize mode weights with GMM (soft) clustering
            gmm = GaussianMixture(n_components=num_modes, n_init=NUM_INIT_RESTARTS,)
            gmm.fit(rollout_features)
            soft_initial_clusters = gmm.predict_proba(rollout_features)

        else:
            raise ValueError(f"Unknown argument for init: {init}")

        if verbose:
            print("Initial clusters:", flush=True)
            print(soft_initial_clusters, flush=True)

        # Compute initial mode weights from soft clustering
        initial_mode_weights = np.sum(soft_initial_clusters, axis=0) / len(rollouts)

        # Compute initial reward weights from up-front clustering
        env_padded, rollouts_padded = pad_terminal_mdp(env, rollouts=rollouts)
        initial_reward_weights = []
        for m in range(num_modes):
            initial_reward_weights.append(
                sw_maxent_irl(
                    rollouts_padded,
                    env_padded,
                    rs=env.state_rewards is not None,
                    rsa=env.state_action_rewards is not None,
                    rsas=env.state_action_state_rewards is not None,
                    rbound=env.reward_range,
                    with_dummy_state=True,
                    grad_twopoint=True,
                    path_weights=soft_initial_clusters[:, m],
                )[0][:-1]
            )

    return initial_mode_weights, initial_reward_weights
