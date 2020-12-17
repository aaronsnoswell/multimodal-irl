"""Implements Babeş-Vroman style EM MM-IRL Algorithms"""


import abc
import copy
import warnings
import numpy as np
import itertools as it

from scipy.optimize import minimize
from scipy.stats import dirichlet

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from unimodal_irl import (
    sw_maxent_irl,
    maxent_path_logprobs,
    nb_backward_pass_log,
    log_partition,
    bv_maxlikelihood_irl,
)
from mdp_extras import Linear, trajectory_reward, q_vi, BoltzmannExplorationPolicy, v_vi


class EMSolver(abc.ABC):
    """An abstract base class for a Multi-Modal IRL EM solver"""

    def __init__(self):
        """C-tor"""
        pass

    def estep(self, xtr, phi, mode_weights, rewards, rollouts):
        """Compute responsibility matrix using MaxEnt reward parameters
        
        Args:
            xtr (mdp_extras.DiscreteExplicitExtras): MDP Extras
            phi (mdp_extras.FeatureFunction): Feature function
            mode_weights (numpy array): Weights for each behaviour mode
            rewards (list): List of mdp_extras.Linear reward functions, one for each mode
            rollouts (list): List of (s, a) rollouts
            
        Returns:
            (numpy array): |D|xK responsibility matrix based on the current reward
                parameters and mode weights.
        """
        raise NotImplementedError

    def mstep(self, xtr, phi, resp, rollouts, reward_range=None):
        """Compute reward parameters given responsibility matrix
        
        Args:
            xtr (DiscreteExplicitExtras): Extras object for multi-modal MDP
            phi (FeatureFunction): Feature function for multi-modal MDP
            resp (numpy array): Responsibility matrix
            rollouts (list): Demonstration data
    
        Returns:
            (list): List of Linear reward functions
        """
        raise NotImplementedError

    def mixture_nll(self, xtr, phi, mode_weights, rewards, rollouts):
        """"""
        raise NotImplementedError


class MaxEntEMSolver(EMSolver):
    """Solve an EM MM-IRL problem with MaxEnt IRL"""

    def estep(self, xtr, phi, mode_weights, rewards, rollouts):
        """Compute responsibility matrix using MaxEnt reward parameters
        
        Args:
            xtr (mdp_extras.DiscreteExplicitExtras): MDP Extras
            phi (mdp_extras.FeatureFunction): Feature function
            mode_weights (numpy array): Weights for each behaviour mode
            rewards (list): List of mdp_extras.Linear reward functions, one for each mode
            rollouts (list): List of (s, a) rollouts
            
        Returns:
            (numpy array): |D|xK responsibility matrix based on the current reward
                parameters and mode weights.
        """

        num_modes = len(mode_weights)

        resp = np.ones((len(rollouts), num_modes))
        for mode_idx, (mode_weight, reward) in enumerate(zip(mode_weights, rewards)):
            resp[:, mode_idx] = mode_weight * np.exp(
                maxent_path_logprobs(xtr, phi, reward, rollouts)
            )

        # Each demonstration gets a mass of 1 to allocate between modes
        resp /= np.sum(resp, axis=1, keepdims=True)

        return resp

    def mstep(self, xtr, phi, resp, rollouts, reward_range=None):
        """Compute reward parameters given responsibility matrix
        
        TODO ajs 2/dec/2020 Support re-using previous reward parameters as starting points
        
        Args:
            xtr (DiscreteExplicitExtras): Extras object for multi-modal MDP
            phi (FeatureFunction): Feature function for multi-modal MDP
            resp (numpy array): Responsibility matrix
            rollouts (list): Demonstration data
    
        Returns:
            (list): List of Linear reward functions
        """

        num_rollouts, num_modes = resp.shape

        rewards = []
        for mode_idx in range(num_modes):

            rollout_weights = resp[:, mode_idx]

            theta0 = np.zeros(len(phi))
            method = "BFGS"
            reward_parameter_bounds = None
            if reward_range is not None:
                method = "L-BFGS-B"
                reward_parameter_bounds = tuple(reward_range for _ in range(len(phi)))
            res = minimize(
                sw_maxent_irl,
                theta0,
                args=(xtr, phi, rollouts, rollout_weights),
                method=method,
                jac=True,
                bounds=reward_parameter_bounds,
            )
            rewards.append(Linear(res.x[:]))

        return rewards

    def mixture_nll(self, xtr, phi, mode_weights, rewards, rollouts):
        """Find the average negative log-likelihood of a MaxEnt mixture model
    
        This is the average over all paths of the log-likelihood of each path. That is
    
        $$
            \mathcal{L}(D \mid \Theta) =
                -1 \times \frac{1}{|D|}
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
    
        Returns:
            (float): Log Likelihood of the rollouts under the given mixture model
        """

        max_path_length = max([len(r) for r in rollouts])

        log_partition_values = []
        for mode_idx, (mode_weight, reward) in enumerate(zip(mode_weights, rewards)):
            alpha_log = nb_backward_pass_log(
                xtr.p0s,
                max_path_length,
                xtr.t_mat,
                xtr.gamma,
                *reward.structured(xtr, phi),
            )
            log_partition_values.append(
                log_partition(max_path_length, alpha_log, xtr.padded)
            )

        rollout_lls = []
        for rollout in rollouts:

            # Get path probability under dynamics
            q_tau = np.exp(xtr.path_log_probability(rollout))

            # Accumulate likelihood for this rollout across modes
            rollout_likelihood = 0
            for mode_idx, (mode_weight, reward, log_partition_value) in enumerate(
                zip(mode_weights, rewards, log_partition_values)
            ):
                # Find MaxEnt likelihood of this rollout under this mode
                path_prob_me = np.exp(trajectory_reward(xtr, phi, reward, rollout))
                z_theta = np.exp(log_partition_value)
                l_rollout_mode = mode_weight * q_tau * path_prob_me / z_theta
                rollout_likelihood += l_rollout_mode
            rollout_lls.append(np.log(rollout_likelihood))

        # Find average path negative log likelihood
        nll = -1 * np.mean(rollout_lls)

        return nll


class MaxLikEMSolver(EMSolver):
    """Solve an EM MM-IRL problem with MaxLikelihood IRL"""

    def __init__(self, boltzman_scale=0.5, qge_tol=1e-3):
        """C-tor
        
        Args:
            boltzman_scale (float): Boltzmann policy scale factor
            qge_tol (float): Tolerance for Q-function gradient estimation
        """
        self._boltzman_scale = boltzman_scale
        self._qge_tol = qge_tol

    def estep(self, xtr, phi, mode_weights, rewards, rollouts):
        """Compute responsibility matrix using Max Likelihood reward parameters
        
        Args:
            xtr (mdp_extras.DiscreteExplicitExtras): MDP Extras
            phi (mdp_extras.FeatureFunction): Feature function
            mode_weights (numpy array): Weights for each behaviour mode
            rewards (list): List of mdp_extras.Linear reward functions, one for each mode
            rollouts (list): List of (s, a) rollouts
            
        Returns:
            (numpy array): |D|xK responsibility matrix based on the current reward
                parameters and mode weights.
        """

        num_modes = len(mode_weights)

        resp = np.ones((len(rollouts), num_modes))
        for mode_idx, (mode_weight, reward) in enumerate(zip(mode_weights, rewards)):
            q_star = q_vi(xtr, phi, reward)
            pi = BoltzmannExplorationPolicy(q_star, scale=self._boltzman_scale)
            for rollout_idx, rollout in enumerate(rollouts):
                resp[rollout_idx, mode_idx] = mode_weight * np.exp(
                    pi.path_log_likelihood(rollout)
                )

        # Each demonstration gets a mass of 1 to allocate between modes
        resp /= np.sum(resp, axis=1, keepdims=True)

        return resp

    def mstep(self, xtr, phi, resp, rollouts, reward_range=None):
        """Compute reward parameters given responsibility matrix
        
        TODO ajs 2/dec/2020 Support re-using previous reward parameters as starting points
        
        Args:
            xtr (DiscreteExplicitExtras): Extras object for multi-modal MDP
            phi (FeatureFunction): Feature function for multi-modal MDP
            resp (numpy array): Responsibility matrix
            rollouts (list): Demonstration data
    
        Returns:
            (list): List of Linear reward functions
        """

        num_rollouts, num_modes = resp.shape

        rewards = []
        for mode_idx in range(num_modes):

            rollout_weights = resp[:, mode_idx]

            theta0 = np.zeros(len(phi))
            method = "BFGS"
            reward_parameter_bounds = None
            if reward_range is not None:
                method = "L-BFGS-B"
                reward_parameter_bounds = tuple(reward_range for _ in range(len(phi)))
            res = minimize(
                bv_maxlikelihood_irl,
                theta0,
                args=(
                    xtr,
                    phi,
                    rollouts,
                    rollout_weights,
                    self._boltzman_scale,
                    self._qge_tol,
                ),
                method=method,
                jac=True,
                bounds=reward_parameter_bounds,
            )
            rewards.append(Linear(res.x[:]))

        return rewards

    def mixture_nll(self, xtr, phi, mode_weights, rewards, rollouts):
        """Find the average negative log-likelihood of a MaxEnt mixture model
    
        This is the average over all paths of the log-likelihood of each path. That is
    
        $$
            \mathcal{L}(D \mid \Theta) =
                -1 \times \frac{1}{|D|}
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
    
        Returns:
            (float): Log Likelihood of the rollouts under the given mixture model
        """

        max_path_length = max([len(r) for r in rollouts])

        log_partition_values = []
        for mode_idx, (mode_weight, reward) in enumerate(zip(mode_weights, rewards)):
            alpha_log = nb_backward_pass_log(
                xtr.p0s,
                max_path_length,
                xtr.t_mat,
                xtr.gamma,
                *reward.structured(xtr, phi),
            )
            log_partition_values.append(
                log_partition(max_path_length, alpha_log, xtr.padded)
            )

        rollout_lls = []
        for rollout in rollouts:

            # Get path probability under dynamics
            q_tau = np.exp(xtr.path_log_probability(rollout))

            # Accumulate likelihood for this rollout across modes
            rollout_likelihood = 0
            for mode_idx, (mode_weight, reward, log_partition_value) in enumerate(
                zip(mode_weights, rewards, log_partition_values)
            ):
                # Find MaxEnt likelihood of this rollout under this mode
                path_prob_me = np.exp(trajectory_reward(xtr, phi, reward, rollout))
                z_theta = np.exp(log_partition_value)
                l_rollout_mode = mode_weight * q_tau * path_prob_me / z_theta
                rollout_likelihood += l_rollout_mode
            rollout_lls.append(np.log(rollout_likelihood))

        # Find average path negative log likelihood
        nll = -1 * np.mean(rollout_lls)

        return nll


def bv_em(
    solver,
    xtr,
    phi,
    rollouts,
    num_modes,
    reward_range,
    mode_weights=None,
    rewards=None,
    tolerance=1e-5,
    max_iterations=None,
):
    """
    Expectation Maximization Multi-Modal IRL by Babeş-Vroman et al. 2011
    
    See the paper "Apprenticeship learning about multiple intentions." in ICML 2011.
    
    Args:
        xtr (mpd_extras.DiscreteExplicitExtras): MDP extras
        phi (mdp_extras.FeatureFunction): Feature function
        rollouts (list): List of (s, a) rollouts
        solver (EMSolver): IRL solver to use for EM algorithm
        num_modes (int): Number of behaviour modes to learn
        reward_range (tuple): Low, High bounds for reward function parameters
        
        mode_weights (numpy array): Initial mode weights. If provided, len() must match
            num_modes. If None, weights are uniformly sampled from the (num_modes - 1)
            probability simplex.
        rewards (list): List of initial rewards (mdp_extras.Linear) - if None, reward
            parameters are uniform randomly initialized within reward_range
        tolerance (float): NLL convergence threshold
        max_iterations (int): Maximum number of iterations (alternate stopping criterion)

    Returns:
        (list): List of responsibility matrix (numpy array) at every iteration
        (list): List of cluster weights (numpy array) at every iteration
        (list): List of rewards (mdp_extras.Linear) at every iteration
        (float): The NLL of the starting configuration
        (list): List of NLL (float) at every algorithm iteration

    """

    # Initialize reward parameters and mode weights randomly if not passed
    if mode_weights is None:
        rv = dirichlet([1.0 / num_modes for _ in range(num_modes)])
        mode_weights = rv.rvs(1)[0]

    if rewards is None:
        rewards = [
            Linear(np.random.uniform(*reward_range, len(phi))) for _ in range(num_modes)
        ]

    assert len(mode_weights) == num_modes
    assert len(rewards) == num_modes

    resp_history = []
    mode_weights_history = []
    rewards_history = []
    nll_history = []
    for iteration in it.count():

        # Compute LL
        nll = solver.mixture_nll(xtr, phi, mode_weights, rewards, rollouts)
        nll_history.append(nll)

        if len(nll_history) > 1:
            if nll_history[-1] > nll_history[-2]:
                # We've over-stepped a solution point - go back
                resp_history = resp_history[:-1]
                mode_weights_history = mode_weights_history[:-1]
                rewards_history = rewards_history[:-1]
                nll_history = nll_history[:-1]
                break

            nll_delta = np.abs(nll_history[-2] - nll_history[-1])
            if nll_delta <= tolerance:
                # NLL has converged
                break

        # E-step
        resp = solver.estep(xtr, phi, mode_weights, rewards, rollouts)
        mode_weights = np.sum(resp, axis=0) / len(rollouts)

        # M-step
        rewards = solver.mstep(xtr, phi, resp, rollouts, reward_range=reward_range)

        resp_history.append(resp)
        mode_weights_history.append(mode_weights)
        rewards_history.append(rewards)

        # Check for max iterations stopping condition
        if max_iterations is not None and iteration >= max_iterations:
            break

    return (
        resp_history,
        mode_weights_history,
        rewards_history,
        nll_history[0],
        nll_history[1:],
    )

    )
