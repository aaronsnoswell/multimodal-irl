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


def responsibilty_matrix_maxent(
    env,
    rollouts,
    num_modes,
    state_reward_weights=None,
    state_action_reward_weights=None,
    state_action_state_reward_weights=None,
    mode_weights=None,
):
    """Compute responsibility matrix using MaxEnt distribution
    
    Args:
        env (gym.Env): Environment providing dynamics
        rollouts (list): List of rollouts
        num_modes (int): Number of modes
        state_reward_weights (list): List of state reward weight vectors
        state_action_reward_weights (list): List of state-action reward weight vectors
        state_action_state_reward_weights (list): List of state-action-state reward weight vectors
        
        mode_weights (numpy array): Optional list of mode weights, defaults to uniform
    
    Returns:
        (numpy array): NxK responsibility matrix (a soft clustering of the provided
            paths under the given MaxEnt mixture model)
    """

    env = copy.deepcopy(env)

    assert (
        state_reward_weights is not None
        or state_action_reward_weights is not None
        or state_action_state_reward_weights is not None
    ), "Must provided at least one reward weight parameter"

    resp = np.ones((len(rollouts), num_modes))

    if mode_weights is None:
        mode_weights = np.ones(num_modes, dtype=float) / num_modes

    for m in range(num_modes):
        # Use Maximum Entropy distribution for computing path likelihoods
        if state_reward_weights is not None:
            env._state_rewards = state_reward_weights[m]
        if state_action_reward_weights is not None:
            env._state_action_rewards = state_action_reward_weights[m]
        if state_action_state_reward_weights is not None:
            env._state_action_state_rewards = state_action_state_reward_weights[m]

        # Pad the environment
        env_padded, rollouts_padded = pad_terminal_mdp(env, rollouts=rollouts)

        # Compute path likelihoods
        resp[:, m] = mode_weights[m] * np.exp(
            maxent_path_logprobs(
                env_padded,
                rollouts_padded,
                theta_s=env_padded.state_rewards,
                theta_sa=env_padded.state_action_rewards,
                theta_sas=env_padded.state_action_state_rewards,
                with_dummy_state=True,
            )
        )

    # Each path (row of the assignment matrix) gets a mass of 1 to allocate between
    # modes
    resp /= np.sum(resp, axis=1, keepdims=True)

    return resp


def mixture_ll_maxent(
    env,
    rollouts,
    mode_weights,
    mode_state_reward_parameters=None,
    mode_state_action_reward_parameters=None,
    mode_state_action_state_reward_parameters=None,
):
    """Find the log-likelihood of a MaxEnt mixture model
    
    This is the average over all paths of the log-likelihood of each path. That is
    
    $$
        \mathcal{L}(D \mid \Theta) =
            \frac{1}{|D|}
            \sum_{\tau \in \Data}
            \log
            \sum_{k=1}^K
            \alpha_k
            ~
            p(\tau \mid \theta_k)
    $$
    
    Where \alpha_k is the weight of mixture component k, and \theta_k is the reward weights
    for that mixture component
    
    Args:
        env (explicit_env.IExplicitEnv): Environment defining dynamics
        rollouts (list): List of state-action rollouts
        mode_weights (list): List of prior probabilities for each mode
        
        mode_state_reward_parameters (list): List of |S| state reward parameter vectors
        mode_state_reward_parameters (list): List of |S|x|A| state-action reward parameter vectors
        mode_state_reward_parameters (list): List of |S|x|A|x|S| state-action-state reward parameter vectors
    
    Returns:
        (float): Log Likelihood of the rollouts under the given mixture model
    """

    assert (
        mode_state_reward_parameters is not None
        or mode_state_action_reward_parameters is not None
        or mode_state_action_state_reward_parameters is not None
    ), "Must provide at least one reward parameter collection"

    env = copy.deepcopy(env)
    num_modes = len(mode_weights)

    # Convert missing reward parameter arguments to lists of 'None'
    if mode_state_reward_parameters is None:
        mode_state_reward_parameters = [None for _ in range(num_modes)]
    if mode_state_action_reward_parameters is None:
        mode_state_action_reward_parameters = [None for _ in range(num_modes)]
    if mode_state_action_state_reward_parameters is None:
        mode_state_action_state_reward_parameters = [None for _ in range(num_modes)]

    assert (
        len(mode_state_reward_parameters)
        == len(mode_state_action_reward_parameters)
        == len(mode_state_action_state_reward_parameters)
        == num_modes
    ), "Provided number of reward parameters does not match number of modes in responsibility matrix"

    # Pre-compute the partition values for each reward parameter
    max_path_length = (
        len(rollouts[0]) if len(rollouts) == 1 else max(*[len(r) for r in rollouts])
    )
    mode_log_partition_values = []
    for mode in range(num_modes):
        env._state_rewards = mode_state_reward_parameters[mode]
        env._state_action_rewards = mode_state_action_reward_parameters[mode]
        env._state_action_state_rewards = mode_state_action_state_reward_parameters[
            mode
        ]
        env_padded, rollouts_padded = pad_terminal_mdp(env, rollouts=rollouts)
        mode_log_partition_values.append(
            maxent_log_partition(
                env_padded,
                max_path_length,
                env_padded.state_rewards,
                env_padded.state_action_rewards,
                env_padded.state_action_state_rewards,
                True,
            )
        )

    # Model Log Likelihood is a sum over rollouts
    ll = 0
    for ri, r in enumerate(rollouts):

        # Path Likelihood is the log of a sum over modes
        l_rollout_permode = []
        for mode in range(num_modes):

            env._state_rewards = mode_state_reward_parameters[mode]
            env._state_action_rewards = mode_state_action_reward_parameters[mode]
            env._state_action_state_rewards = mode_state_action_state_reward_parameters[
                mode
            ]

            q = np.exp(env.path_log_probability(r))

            mode_weight = mode_weights[mode]
            path_prob_me = np.exp(r_tau(env, r))
            z_theta = np.exp(mode_log_partition_values[mode])
            l_rollout_mode = mode_weight * q * path_prob_me / z_theta
            l_rollout_permode.append(l_rollout_mode)
        l_rollout = np.sum(l_rollout_permode)
        ll_rollout = np.log(l_rollout)
        ll += ll_rollout / len(rollouts)

    return ll


def bv_em_maxent(
    env,
    rollouts,
    num_modes,
    rs=False,
    rsa=False,
    rsas=False,
    initial_mode_weights=None,
    initial_state_reward_weights=None,
    initial_state_action_reward_weights=None,
    initial_state_action_state_reward_weights=None,
    max_iterations=100,
    min_weight_change=1e-6,
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
        
        rs (bool): Solve for state rewards?
        rsa (bool): Solve for state-action rewards?
        rsas (bool): Solve for state-action-state rewards?
        initial_mode_weights (numpy array): Optional list of initial mode weights -
            if not provided, defaults to uniform.
        initial_state_reward_weights (list): List of initial mode state reward weights (numpy
            arrays) - one for each mode. If not provided, these weights are sampled from
            a uniform distribution over possible reward values.
        initial_state_action_reward_weights (list): List of initial mode state-action reward weights (numpy
            arrays) - one for each mode. If not provided, these weights are sampled from
            a uniform distribution over possible reward values.
        initial_state_action_state_reward_weights (list): List of initial mode state-action-state reward weights (numpy
            arrays) - one for each mode. If not provided, these weights are sampled from
            a uniform distribution over possible reward values.
        max_iterations (int): Maximum number of EM iterations to perform
        min_weight_change (float): Stop EM iterations when the change in mode weights
            fall below this value
        verbose (bool): Print progress information
    
    Returns:
        (int): Number of iterations performed until convergence
        (numpy array): N x K numpy array representing path assignments to modes. One row
            for each path in the demonstration data, one column for each mode
        (numpy array): K x |S| array of recovered state reward weights for each mode
        (numpy array): K x |S|x|A| array of recovered state-action reward weights for each mode
        (numpy array): K x |S|x|A|x|S| array of recovered state-aciton-state reward weights for each mode
    """

    assert rs or rsa or rsas, "Must specify one of rs, rsa, or rsas!"

    # Work with a copy of the env
    env = copy.deepcopy(env)

    env_padded, rollouts_padded = pad_terminal_mdp(env, rollouts=rollouts)

    # How many paths have we got?
    num_paths = len(rollouts)

    # Initialize mode weights
    if initial_mode_weights is None:
        mode_weights = np.ones(num_modes, dtype=float) / num_modes
    else:
        assert (
            len(initial_mode_weights) == num_modes
        ), "Wrong number of initial mode weights passed"
        mode_weights = initial_mode_weights

    if rs:
        # Initialize state rewards
        if initial_state_reward_weights is None:
            # Use uniform initial weights over range of valid reward values
            state_reward_weights = [
                np.random.uniform(
                    low=env.reward_range[0],
                    high=env.reward_range[1],
                    size=len(env.states),
                )
                for _ in range(num_modes)
            ]
        else:
            assert (
                len(initial_state_reward_weights) == num_modes
            ), "Wrong number of initial reward weights passed"
            state_reward_weights = initial_state_reward_weights
            state_reward_weights_clipped = np.clip(
                state_reward_weights, *env.reward_range
            )
            if not np.all(state_reward_weights_clipped == state_reward_weights):
                warnings.warn(
                    "Initial reward weights are outside valid reward ranges - clipping"
                )
            state_reward_weights = state_reward_weights_clipped
    else:
        state_reward_weights = None
    if rsa:
        # Initialize state-action rewards
        if initial_state_action_reward_weights is None:
            # Use uniform initial weights over range of valid reward values
            state_action_reward_weights = [
                np.random.uniform(
                    low=env.reward_range[0],
                    high=env.reward_range[1],
                    size=(len(env.states), len(env.actions)),
                )
                for _ in range(num_modes)
            ]
        else:
            assert (
                len(initial_state_action_reward_weights) == num_modes
            ), "Wrong number of initial reward weights passed"
            state_action_reward_weights = initial_state_action_reward_weights
            state_action_reward_weights_clipped = np.clip(
                state_action_reward_weights, *env.reward_range
            )
            if not np.all(
                state_action_reward_weights_clipped == state_action_reward_weights
            ):
                warnings.warn(
                    "Initial reward weights are outside valid reward ranges - clipping"
                )
            state_action_reward_weights = state_action_reward_weights_clipped
    else:
        state_action_reward_weights = None
    if rsas:
        # Initialize state-action-state rewards
        if initial_state_action_state_reward_weights is None:
            # Use uniform initial weights over range of valid reward values
            state_action_state_reward_weights = [
                np.random.uniform(
                    low=env.reward_range[0],
                    high=env.reward_range[1],
                    size=(len(env.states), len(env.actions), len(env.states)),
                )
                for _ in range(num_modes)
            ]
        else:
            assert (
                len(initial_state_action_state_reward_weights) == num_modes
            ), "Wrong number of initial reward weights passed"
            state_action_state_reward_weights = (
                initial_state_action_state_reward_weights
            )
            state_action_state_reward_weights_clipped = np.clip(
                state_action_state_reward_weights, *env.reward_range
            )
            if not np.all(
                state_action_state_reward_weights_clipped
                == state_action_state_reward_weights
            ):
                warnings.warn(
                    "Initial reward weights are outside valid reward ranges - clipping"
                )
            state_action_state_reward_weights = (
                state_action_state_reward_weights_clipped
            )
    else:
        state_action_state_reward_weights = None

    for _it in it.count():
        if verbose:
            print("EM Iteration {}".format(_it + 1), end="", flush=True)
            print(", Weights:{}".format(mode_weights), end="", flush=True)

        # E-step: Solve for responsibility matrix
        zij = responsibilty_matrix_maxent(
            env,
            rollouts,
            num_modes,
            state_reward_weights,
            state_action_reward_weights,
            state_action_state_reward_weights,
            mode_weights,
        )

        # M-step: Update mode weights and reward estimates
        old_mode_weights = copy.copy(mode_weights)
        mode_weights = np.sum(zij, axis=0) / num_paths

        for m in range(num_modes):
            # Use MaxEnt IRL for computing reward functions
            theta_s, theta_sa, theta_sas = sw_maxent_irl(
                rollouts_padded,
                env_padded,
                rs=rs,
                rsa=rsa,
                rsas=rsas,
                rbound=env.reward_range,
                with_dummy_state=True,
                grad_twopoint=True,
                path_weights=zij[:, m],
            )
            if rs:
                state_reward_weights[m] = theta_s[:-1]
            if rsa:
                state_action_reward_weights[m] = theta_sa[:-1, :-1]
            if rsas:
                state_action_state_reward_weights[m] = theta_sas[:-1, :-1, :-1]

        delta = np.max(np.abs(old_mode_weights - mode_weights))
        if verbose:
            print(", Δ:{}".format(delta), flush=True)

        if delta <= min_weight_change:
            if verbose:
                print("EM mode wights have converged, stopping", flush=True)
            break

        if _it == max_iterations - 1:
            if verbose:
                print("Reached maximum number of EM iterations, stopping", flush=True)
            break

    if rs:
        state_reward_weights = np.array(state_reward_weights)
    if rsa:
        state_action_reward_weights = np.array(state_action_reward_weights)
    if rsas:
        state_action_state_reward_weights = np.array(state_action_state_reward_weights)

    return (
        (_it + 1),
        zij,
        state_reward_weights,
        state_action_reward_weights,
        state_action_state_reward_weights,
    )


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
        (numpy array): NxK initial responsibility matrix
        (numpy array): K Vector of initial mode weights
        (numpy array): K list of |S| arrays of initial state reward weights
        (numpy array): K list of |S|x|A| arrays of initial state-action reward weights
        (numpy array): K list of |S|x|A|x|S| arrays of initial state-action-state reward weights
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
        initial_state_reward_weights = None
        initial_state_action_reward_weights = None
        initial_state_action_state_reward_weights = None

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
        initial_state_reward_weights = []
        initial_state_action_reward_weights = []
        initial_state_action_state_reward_weights = []
        for m in range(num_modes):
            r_s, r_sa, r_sas = sw_maxent_irl(
                rollouts_padded,
                env_padded,
                rs=env.state_rewards is not None,
                rsa=env.state_action_rewards is not None,
                rsas=env.state_action_state_rewards is not None,
                rbound=env.reward_range,
                with_dummy_state=True,
                grad_twopoint=True,
                path_weights=soft_initial_clusters[:, m],
            )

            # Drop dummy states
            if env.state_rewards is not None:
                r_s = r_s[:-1]
            if env.state_action_rewards is not None:
                r_sa = r_sa[:-1, :-1]
            if env.state_action_state_rewards is not None:
                r_sas = r_sas[:-1, :-1, :-1]

            initial_state_reward_weights.append(r_s)
            initial_state_action_reward_weights.append(r_sa)
            initial_state_action_state_reward_weights.append(r_sas)

    return (
        initial_mode_weights,
        initial_state_reward_weights,
        initial_state_action_reward_weights,
        initial_state_action_state_reward_weights,
    )
