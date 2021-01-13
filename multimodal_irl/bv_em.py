"""Implements Babeş-Vroman style EM MM-IRL Algorithms"""


import abc
import copy
import warnings
import numpy as np
import itertools as it

from concurrent import futures
from scipy.stats import dirichlet
from scipy.optimize import minimize

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from unimodal_irl import (
    sw_maxent_irl,
    maxent_path_logprobs,
    nb_backward_pass_log,
    nb_backward_pass_log_deterministic_stateonly,
    log_partition,
    bv_maxlikelihood_irl,
)
from mdp_extras import (
    Linear,
    trajectory_reward,
    q_vi,
    BoltzmannExplorationPolicy,
    v_vi,
    DiscreteExplicitExtras,
    DiscreteImplicitExtras,
)


class EMSolver(abc.ABC):
    """An abstract base class for a Multi-Modal IRL EM solver"""

    def __init__(
        self,
        minimize_kwargs={},
        minimize_options={},
        pre_it=lambda i: None,
        post_it=lambda i: None,
    ):
        """C-tor
        
        Args:
            minimize_kwargs (dict): Optional keyword arguments to scipy.optimize.minimize
            minimize_options (dict): Optional args for the scipy.optimize.minimize
                'options' parameter
            pre_it (callable): Optional function accepting the current iteration - called
                before that iteration commences
            post_it (callable): Optional function accepting the solver, current iteration,
                responsibility matrix, mode weights, and reward objects - called
                after that iteration ends
        """
        self.minimize_kwargs = minimize_kwargs
        self.minimize_options = minimize_options
        self.pre_it = pre_it
        self.post_it = post_it
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
            
            reward_range (tuple): Optional reward parameter min and max values
    
        Returns:
            (list): List of Linear reward functions
        """
        raise NotImplementedError

    def mixture_nll(self, xtr, phi, mode_weights, rewards, rollouts):
        """"""
        raise NotImplementedError

    def init_random(self, phi, num_clusters, reward_range, with_resp=False):
        """Initialize mixture model uniform randomly
        
        Args:
            phi (mdp_extras.FeatureFunction): Feature function
            num_clusters (int): Number of mixture components
            reward_range (tuple): Lower and upper reward function parameter bounds
            
            with_resp (bool): Also return responsibility matrix
        
        Returns:
            (numpy array): Initial mixture component weights
            (list): List of mdp_extras.Linear initial reward functions
        """

        mode_weights = dirichlet([1.0 / num_clusters for _ in range(num_clusters)]).rvs(
            1
        )[0]

        rewards = [
            Linear(np.random.uniform(*reward_range, len(phi)))
            for _ in range(num_clusters)
        ]

        if not with_resp:
            return mode_weights, rewards
        else:
            raise NotImplementedError

    def init_kmeans(
        self,
        xtr,
        phi,
        rollouts,
        num_clusters,
        reward_range,
        num_restarts=5000,
        with_resp=False,
    ):
        """Initialize mixture model with KMeans (hard clustering)
        
        Args:
            xtr (mdp_extras.DiscreteExplicitExtras): MDP Extras
            phi (mdp_extras.FeatureFunction): Feature function
            rollouts (list): List of (s, a) rollouts
            num_clusters (int): Number of mixture components
            reward_range (tuple): Lower and upper reward function parameter bounds
            
            num_restarts (int): Number of random clusterings to perform
            with_resp (bool): Also return responsibility matrix
        
        Returns:
            (numpy array): Initial mixture component weights
            (list): List of mdp_extras.Linear initial reward functions
        """

        feature_mat = np.array([phi.expectation([r], xtr.gamma) for r in rollouts])

        km = KMeans(n_clusters=num_clusters, n_init=num_restarts)
        hard_initial_clusters = km.fit_predict(feature_mat)
        soft_initial_clusters = np.zeros((len(rollouts), num_clusters))
        for idx, clstr in enumerate(hard_initial_clusters):
            soft_initial_clusters[idx, clstr] = 1.0

        # Compute initial mode weights from soft clustering
        mode_weights = np.sum(soft_initial_clusters, axis=0) / len(rollouts)

        # Compute initial rewards
        rewards = self.mstep(
            xtr, phi, soft_initial_clusters, rollouts, reward_range=reward_range
        )

        if not with_resp:
            return mode_weights, rewards
        else:
            return soft_initial_clusters, mode_weights, rewards

    def init_gmm(
        self,
        xtr,
        phi,
        rollouts,
        num_clusters,
        reward_range,
        num_restarts=5000,
        with_resp=False,
    ):
        """Initialize mixture model with GMM (soft clustering)
        
        Args:
            xtr (mdp_extras.DiscreteExplicitExtras): MDP Extras
            phi (mdp_extras.FeatureFunction): Feature function
            rollouts (list): List of (s, a) rollouts
            num_clusters (int): Number of mixture components
            reward_range (tuple): Lower and upper reward function parameter bounds
            
            num_restarts (int): Number of random clusterings to perform
            with_resp (bool): Also return responsibility matrix
        
        Returns:
            (numpy array): Initial mixture component weights
            (list): List of mdp_extras.Linear initial reward functions
        """

        feature_mat = np.array([phi.expectation([r], xtr.gamma) for r in rollouts])

        gmm = GaussianMixture(n_components=num_clusters, n_init=num_restarts)
        gmm.fit(feature_mat)
        soft_initial_clusters = gmm.predict_proba(feature_mat)

        # Compute initial mode weights from soft clustering
        mode_weights = np.sum(soft_initial_clusters, axis=0) / len(rollouts)

        # Compute initial rewards
        rewards = self.mstep(
            xtr, phi, soft_initial_clusters, rollouts, reward_range=reward_range
        )

        if not with_resp:
            return mode_weights, rewards
        else:
            return soft_initial_clusters, mode_weights, rewards


class MaxEntEMSolver(EMSolver):
    """Solve an EM MM-IRL problem with MaxEnt IRL"""

    def __init__(
        self,
        minimize_kwargs={},
        minimize_options={},
        pre_it=lambda i: None,
        post_it=lambda i: None,
        parallel_executor=None,
    ):
        """C-tor
        
        Args:
            minimize_kwargs (dict): Optional keyword arguments to scipy.optimize.minimize
            minimize_options (dict): Optional args for the scipy.optimize.minimize
                'options' parameter
            pre_it (callable): Optional function accepting the current iteration - called
                before that iteration commences
            post_it (callable): Optional function accepting the solver, current iteration,
                responsibility matrix, mode weights, and reward objects - called
                after that iteration ends
            parallel_executor (concurrent.futures.Executor) optional executor object to
                parallelize each the E and M steps across modes.
        """

        super().__init__(minimize_kwargs, minimize_options, pre_it, post_it)

        # The MaxEnt reward learning objective is convex, so we can safely store the
        # previous iteration's reward solutions to use as a super-efficient starting
        # point for the current iteration's reward optimization
        self._prev_rewards = None
        self.parallel_executor = parallel_executor

    def estep(self, xtr, phi, mode_weights, rewards, demonstrations):
        """Compute responsibility matrix using MaxEnt reward parameters
        
        Args:
            xtr (mdp_extras.DiscreteExplicitExtras): MDP Extras
            phi (mdp_extras.FeatureFunction): Feature function
            mode_weights (numpy array): Weights for each behaviour mode
            rewards (list): List of mdp_extras.Linear reward functions, one for each mode
            demonstrations (list): List of (s, a) rollouts
            
        Returns:
            (numpy array): |D|xK responsibility matrix based on the current reward
                parameters and mode weights.
        """

        num_modes = len(mode_weights)

        # Shortcut for K=1
        if num_modes == 1:
            return np.array([np.ones(len(demonstrations))]).T

        weights_rewards = zip(mode_weights, rewards)
        proc_one = (
            lambda xtr, phi, mode_weight, mode_reward, demonstrations: mode_weight
            * np.exp(maxent_path_logprobs(xtr, phi, mode_reward, demonstrations))
        )
        if self.parallel_executor is None:
            resp = np.ones((len(demonstrations), num_modes))
            for mode_idx, (mode_weight, mode_reward) in enumerate(weights_rewards):
                resp[:, mode_idx] = proc_one(
                    xtr, phi, mode_weight, mode_reward, demonstrations
                )
        else:
            # Parallelelize execution over modes
            tasks = {
                self.parallel_executor.submit(
                    proc_one, xtr, phi, mode_weight, mode_reward, demonstrations
                )
                for (mode_weight, mode_reward) in weights_rewards
            }
            resp = []
            for future in futures.as_completed(tasks):
                # Use arg or result here if desired
                # arg = tasks[future]
                resp.append(future.result())
            resp = np.array(resp).T

        # Each demonstration gets a mass of 1 to allocate between modes
        resp /= np.sum(resp, axis=1, keepdims=True)

        return resp

    def mstep(
        self, xtr, phi, resp, demonstrations, reward_range=None,
    ):
        """Compute reward parameters given responsibility matrix
        
        Args:
            xtr (DiscreteExplicitExtras): Extras object for multi-modal MDP
            phi (FeatureFunction): Feature function for multi-modal MDP
            resp (numpy array): Responsibility matrix
            demonstrations (list): Demonstration data
            
            reward_range (tuple): Optional reward parameter min and max values
    
        Returns:
            (list): List of Linear reward functions
        """

        num_rollouts, num_modes = resp.shape

        method = "BFGS"
        reward_parameter_bounds = None
        if reward_range is not None:
            method = "L-BFGS-B"
            reward_parameter_bounds = tuple(reward_range for _ in range(len(phi)))

        theta0 = [np.zeros(len(phi)) for i in range(num_modes)]
        if self._prev_rewards is not None:
            # Because the MaxEnt optimization is convex, we can safely re-use the
            # previous iteration's reward estimates as an efficient starting point
            theta0 = [self._prev_rewards[i].theta for i in range(num_modes)]

        max_path_length = max(*[len(r) for r in demonstrations])

        def proc_one(
            xtr,
            phi,
            max_path_length,
            method,
            demonstrations,
            reward_parameter_bounds,
            minimize_options,
            minimize_kwargs,
            rollout_weights,
            theta0,
        ):
            phi_bar = phi.expectation(
                demonstrations, gamma=xtr.gamma, weights=rollout_weights
            )

            res = minimize(
                sw_maxent_irl,
                theta0,
                args=(xtr, phi, phi_bar, max_path_length),
                method=method,
                jac=True,
                bounds=reward_parameter_bounds,
                options=minimize_options,
                **(minimize_kwargs),
            )
            return Linear(res.x[:])

        demo_weights_theta0s = zip(resp.T, theta0)
        rewards = []
        if self.parallel_executor is None:
            for mode_demo_weights, mode_theta0 in demo_weights_theta0s:
                rewards.append(
                    proc_one(
                        xtr,
                        phi,
                        max_path_length,
                        method,
                        demonstrations,
                        reward_parameter_bounds,
                        self.minimize_options,
                        self.minimize_kwargs,
                        mode_demo_weights,
                        mode_theta0,
                    )
                )
        else:
            tasks = {
                self.parallel_executor.submit(
                    proc_one,
                    xtr,
                    phi,
                    max_path_length,
                    method,
                    demonstrations,
                    reward_parameter_bounds,
                    self.minimize_options,
                    self.minimize_kwargs,
                    mode_demo_weights,
                    mode_theta0,
                )
                for mode_demo_weights, mode_theta0 in demo_weights_theta0s
            }
            for future in futures.as_completed(tasks):
                # Use arg or result here if desired
                # arg = tasks[future]
                rewards.append(future.result())

        # Store current reward estimate for next time
        self._prev_rewards = rewards

        return rewards

    def mixture_nll(self, xtr, phi, mode_weights, rewards, demonstrations):
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
            xtr (): Extras object
            phi (): Features object
            mode_weights (list): List of prior probabilities for each mode
            rewards (): reward
            demonstrations (list): List of state-action rollouts
    
        Returns:
            (float): Log Likelihood of the rollouts under the given mixture model
        """

        weights_rewards = zip(mode_weights, rewards)
        proc_one = (
            lambda xtr, phi, mode_weight, mode_reward, demonstrations: mode_weight
            * np.exp(maxent_path_logprobs(xtr, phi, mode_reward, demonstrations))
        )

        path_likelihoods = []
        if self.parallel_executor is None:
            for mode_weight, mode_reward in weights_rewards:
                mode_path_likelihoods = proc_one
                # Add all finite path likelihoods
                path_likelihoods.extend(
                    proc_one(xtr, phi, mode_weight, mode_reward, demonstrations)
                )
        else:
            # Parallelize across modes
            tasks = {
                self.parallel_executor.submit(
                    proc_one, xtr, phi, mode_weight, mode_reward, demonstrations
                )
                for (mode_weight, mode_reward) in weights_rewards
            }
            for future in futures.as_completed(tasks):
                # Use arg or result here if desired
                # arg = tasks[future]
                path_likelihoods.extend(future.result())

        # Drop -inf values
        path_likelihoods = np.array(path_likelihoods)
        path_likelihoods = path_likelihoods[np.isfinite(path_likelihoods)]

        # Return average over all paths, modes
        return -1.0 * np.mean(path_likelihoods)


class MeanOnlyEMSolver(MaxEntEMSolver):
    """Approximates a MaxEnt solver by picking the feature expectation at every mstep"""

    def mstep(
        self, xtr, phi, resp, demonstrations, reward_range=None,
    ):
        """Compute reward parameters given responsibility matrix
        
        Args:
            xtr (DiscreteExplicitExtras): Extras object for multi-modal MDP
            phi (FeatureFunction): Feature function for multi-modal MDP
            resp (numpy array): Responsibility matrix
            demonstrations (list): Demonstration data
            
            reward_range (tuple): Optional reward parameter min and max values
    
        Returns:
            (list): List of Linear reward functions
        """
        rewards = []
        for mode_demo_weights in resp.T:
            phi_bar = phi.expectation(
                demonstrations, gamma=xtr.gamma, weights=mode_demo_weights
            )
            rewards.append(Linear(phi_bar))
        return rewards


class MaxLikEMSolver(EMSolver):
    """Solve an EM MM-IRL problem with MaxLikelihood IRL"""

    def __init__(
        self,
        boltzman_scale=0.5,
        qge_tol=1e-3,
        minimize_kwargs={},
        minimize_options={},
        pre_it=lambda i: None,
        post_it=lambda i: None,
    ):
        """C-tor
        
        Args:
            boltzman_scale (float): Boltzmann policy scale factor
            qge_tol (float): Tolerance for Q-function gradient estimation
            minimize_kwargs (dict): Optional keyword arguments to scipy.optimize.minimize
            minimize_options (dict): Optional args for the scipy.optimize.minimize
                'options' parameter
            pre_it (callable): Optional function accepting the current iteration - called
                before that iteration commences
            post_it (callable): Optional function accepting the solver, current iteration,
                responsibility matrix, mode weights, and reward objects - called
                after that iteration ends
        """
        super().__init__(minimize_kwargs, minimize_options, pre_it, post_it)

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

        # Shortcut for K=1
        if num_modes == 1:
            return np.array([np.ones(len(rollouts))]).T

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

    def mstep(
        self, xtr, phi, resp, rollouts, reward_range=None,
    ):
        """Compute reward parameters given responsibility matrix
        
        Args:
            xtr (DiscreteExplicitExtras): Extras object for multi-modal MDP
            phi (FeatureFunction): Feature function for multi-modal MDP
            resp (numpy array): Responsibility matrix
            rollouts (list): Demonstration data
            
            reward_range (tuple): Optional reward parameter min and max values
    
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
                options=self.minimize_options,
                **(self.minimize_kwargs),
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
                log_partition(max_path_length, alpha_log, xtr.is_padded)
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
        (int): Number of EM iterations performed
        (list): List of responsibility matrix (numpy array) at every iteration
        (list): List of cluster weights (numpy array) at the start, and for every iteration
        (list): List of rewards (mdp_extras.Linear) at the start, and for every iteration
        (list): List of NLL (float) at the start, and for every algorithm iteration
        (str): Reason for termination
    """

    # Initialize reward parameters and mode weights randomly if not passed
    if mode_weights is None:
        mode_weights, _ = solver.init_random(phi, num_modes, reward_range)

    if rewards is None:
        _, rewards = solver.init_random(phi, num_modes, reward_range)

    assert len(mode_weights) == num_modes
    assert len(rewards) == num_modes

    # Compute LL
    nll = solver.mixture_nll(xtr, phi, mode_weights, rewards, rollouts)

    resp_history = []
    mode_weights_history = [mode_weights]
    rewards_history = [rewards]
    nll_history = [nll]
    reason = ""
    for iteration in it.count():

        # Call user pre-iteration callback
        solver.pre_it(iteration)

        # E-step - update responsibility matrix, mixture component weights
        resp = solver.estep(xtr, phi, mode_weights, rewards, rollouts)
        resp_history.append(resp)
        mode_weights = np.sum(resp, axis=0) / len(rollouts)
        mode_weights_history.append(mode_weights)

        # M-step - solve for new reward parameters
        rewards = solver.mstep(xtr, phi, resp, rollouts, reward_range=reward_range)
        rewards_history.append(rewards)

        # Compute LL
        nll = solver.mixture_nll(xtr, phi, mode_weights, rewards, rollouts)
        nll_history.append(nll)

        # Call user post-iteration callback
        solver.post_it(solver, iteration, resp, mode_weights, rewards, nll)

        # Edge case for only one cluster
        if num_modes == 1:
            reason = "Only one cluster to learn"
            break

        # Check NLL delta
        nll_deltas = np.diff(nll_history)
        if not np.all(nll_deltas <= 0):
            reason = "NLL is not monotonically decreasing - possible loss of accuracy due to numerical rounding"
            break
        elif np.abs(nll_deltas[-1]) <= tolerance:
            # NLL has converged
            reason = "NLL converged: |NLL delta| <= tol"
            break

        # Check for max iterations stopping condition
        if max_iterations is not None and iteration >= max_iterations - 1:
            reason = "Max iterations reached"
            break

    return (
        iteration + 1,
        resp_history,
        mode_weights_history,
        rewards_history,
        nll_history,
        reason,
    )


def main():
    """Main function"""
    pass


if __name__ == "__main__":
    main()
