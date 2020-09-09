"""Implements Babes-Vroman style EM MM-IRL Algorithms"""


import copy
import warnings
import numpy as np
import itertools as it


from explicit_env.soln import value_iteration, q_from_v, BoltzmannExplorationPolicy
from unimodal_irl import sw_maxent_irl
from unimodal_irl.utils import pad_terminal_mdp
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
    mode_partition_values = []
    for r in reward_weights:
        env._state_rewards = r
        env_padded, rollouts_padded = pad_terminal_mdp(env, rollouts=rollouts)
        mode_partition_values.append(
            maxent_log_partition(
                env_padded, max_path_length, env_padded.state_rewards, None, None, True,
            )
        )

    total_ll = 0
    # Sweep all paths
    for r in rollouts:

        # Find the likelihood of htis
        path_likelihood = 0
        for mode in range(num_modes):
            # Slice out mode params
            mode_weight = np.sum(zij[mode])
            mode_reward = reward_weights[mode]
            mode_partition = mode_partition_values[mode]

            env_padded._state_reward = mode_reward

            # Get log probability of this path under this mode
            maxent_path_logprob = (
                env.path_log_probability(r) + r_tau(env_padded, r) - mode_partition
            )

            # Find the likelihood of this path under this mode
            path_mode_likelihood = mode_weight * np.exp(maxent_path_logprob)
            path_likelihood += path_mode_likelihood

        # This path's log probability is the log of the sum of it's probabilites under each mode
        total_ll += np.log(path_likelihood)

    return total_ll


def bv_em_maxent(
    env,
    rollouts,
    num_modes,
    initial_mode_weights=None,
    initial_reward_weights=None,
    max_iterations=100,
    min_weight_change=1e-6,
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
        print("EM Iteration {}".format(_it + 1), end="")
        print(", Weights:{}".format(mode_weights), end="")

        # E-step: Solve for responsibility matrix
        zij = responsibilty_matrix_maxent(env, rollouts, reward_weights, mode_weights)

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
        print(", Δ:{}".format(delta))

        if delta <= min_weight_change:
            print("EM mode wights have converged, stopping")
            break

        if _it == max_iterations - 1:
            print("Reached maximum number of EM iterations, stopping")
            break

    return (_it + 1), zij, mode_weights, np.array(reward_weights)
