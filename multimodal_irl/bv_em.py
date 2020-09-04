"""Implements Babes-Vroman style EM MM-IRL Algorithms"""


import copy
import warnings
import numpy as np
import itertools as it


from explicit_env.soln import value_iteration, q_from_v, BoltzmannExplorationPolicy
from unimodal_irl import sw_maxent_irl
from unimodal_irl.utils import pad_terminal_mdp
from unimodal_irl.sw_maxent_irl import maxent_path_logprobs


# def bv_mm_irl_mlirl(
#     env,
#     rollouts,
#     num_modes,
#     initial_mode_weights=None,
#     initial_reward_weights=None,
#     max_iterations=100,
#     min_weight_change=1e-6,
#     boltzman_scale=1.0,
# ):
#     """Solve an MM-IRL problem using the Babes-Vroman EM alg with Max Likelihood IRL
#
#     Args:
#         env (explicit_env.envs.IExplicitEnv): Environment defining the dynamics of the
#             multi-modal IRL problem and the .reward_range parameter
#         rollouts (list): List of demonstration trajectories, each a list of (s ,a)
#             tuples
#         num_modes (int): Number of modes to solve for - this is a hyper parameter of
#             the algorithm
#
#         initial_mode_weights (numpy array): Optional initial mode weight vector. If not
#             provided, this will be initialized in a near-uniform fashion.
#         initial_reward_weights (list): List of initial mode reward weights (numpy
#             arrays) - one for each mode. If not provided, these weights are sampled from
#             a uniform distribution over possible reward values.
#         max_iterations (int): Maximum number of EM iterations to perform
#         min_weight_change (float): Stop EM iterations when the change in mode weights
#             fall below this value
#         boltzman_scale (float): Scale parameter for the Boltzman exploration policies
#             used for computing path likelihoods at each E-step
#
#     Returns:
#         (int): Number of iterations performed until convergence
#         (numpy array): N x K numpy array representing path assignments to modes. One row
#             for each path in the demonstration data, one column for each mode
#         (numpy array): List of weights, one for each reward mode - this is equal to the
#             normalized sum along the columns of the assignment matrix, and is provided
#             for convenience only.
#         (list): List of recovered reward weights (numpy arrays), one for each reward
#             mode
#     """
#
#     # Work with a copy of the env
#     env = copy.deepcopy(env)
#
#     env_padded, rollouts_padded = pad_terminal_mdp(env, rollouts=rollouts)
#
#     # How many paths have we got?
#     num_paths = len(rollouts)
#
#     # Pick random initial mode weights and mode reward parameters
#     if initial_mode_weights is None:
#         # Initialize weights from uniform Dirichlet sample
#         # mode_weights = np.random.dirichlet(np.ones(num_modes) / num_modes)
#
#         # Initialize mode weights using same method as scikit-learn GaussianMixtureModel
#         # https://github.com/scikit-learn/scikit-learn/blob/0fb307bf39bbdacd6ed713c00724f8f871d60370/sklearn/mixture/_gaussian_mixture.py#L629
#         # This results in weights that are close to balanced, but not quite
#         _zij = np.random.rand(num_paths, num_modes)
#         _zij /= np.sum(_zij, axis=1, keepdims=True)
#         mode_weights = _zij.sum(axis=0) + np.finfo(_zij.dtype).eps * 10
#         mode_weights /= num_paths
#     else:
#         mode_weights = initial_mode_weights
#
#     if initial_reward_weights is None:
#         # Use uniform initial weights over range of valid reward values
#         reward_weights = [
#             np.random.uniform(
#                 low=env.reward_range[0], high=env.reward_range[1], size=len(env.states)
#             )
#             for _ in range(num_modes)
#         ]
#     else:
#         reward_weights = initial_reward_weights
#         m2 = np.clip(reward_weights, *env.reward_range)
#         if not np.all(m2 == reward_weights):
#             warnings.warn(
#                 "Initial reward weights are outside valid reward ranges - clipping"
#             )
#         reward_weights = m2
#
#     for _it in it.count():
#
#         print()
#         print("================== EM Iteration {}".format(_it))
#         print("Current mode weights: {}".format(mode_weights))
#         print("Current mode reward weights:")
#         for mr in reward_weights:
#             print(mr)
#
#         # E-step: Solve for assignment matrix
#         zij = np.ones((num_paths, num_modes))
#         for m in range(num_modes):
#
#             # Use Boltzman policy for computing path likelihoods
#             env._state_rewards = reward_weights[m]
#             v_star = value_iteration(env)
#             q_star = q_from_v(v_star, env)
#
#             # The above uses a vanilla Q function for the Boltzman policy - Babes Vroman
#             # calls for a special blended Q function Q_{A_\theta}
#             raise NotImplementedError(
#                 "Need to implement Babes-Vroman style Q-function!"
#             )
#
#             pi_b = BoltzmannExplorationPolicy(q_star, scale=boltzman_scale)
#
#             # Measure how much each path agrees with the policy
#             for pi, p in enumerate(rollouts):
#                 zij[pi, m] = mode_weights[m] * np.exp(pi_b.path_log_likelihood(p))
#
#         # print("Assignment matrix: ")
#         # print(zij)
#
#         # Each path (row of the assignment matrix) gets a mass of 1 to allocate between
#         # modes
#         zij /= np.sum(zij, axis=1, keepdims=True)
#
#         # M-step: Update mode weights and reward estimates
#         old_mode_weights = mode_weights.copy()
#         mode_weights = np.sum(zij, axis=0) / num_paths
#         for m in range(num_modes):
#             raise NotImplementedError("Haven't yet implemented Maximum Likelihood IRL!")
#             MaxLikelihoodIRL = None
#             theta_s = MaxLikelihoodIRL()
#             reward_weights[m] = theta_s
#
#         delta = np.max(np.abs(old_mode_weights - mode_weights))
#         print("Weight change: {}".format(delta))
#
#         if delta <= min_weight_change:
#             print("EM mode wights have converged, stopping")
#             break
#
#         if _it == max_iterations - 1:
#             print("Reached maximum number of EM iterations, stopping")
#             break
#
#     return _it, zij, mode_weights, reward_weights


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
