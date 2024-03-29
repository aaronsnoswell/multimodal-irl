"""Implements Babeş-Vroman style EM MM-IRL Algorithms"""


import abc
import cma
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
    bv_maxlikelihood_irl,
    traj_jacobian,
    form_jacobian,
    pi_gradient_irl,
    optimal_jacobian_mean,
    gradient_path_logprobs,
)
from mdp_extras import (
    Linear,
    vi,
    BoltzmannExplorationPolicy,
    DiscreteExplicitExtras,
    MirrorWrap,
)


class EMSolver(abc.ABC):
    """An abstract base class for a Multi-Modal IRL EM solver"""

    def __init__(
        self,
        minimize_kwargs={},
        minimize_options={},
        pre_it=lambda i: None,
        post_it=lambda solver, iteration, resp, mode_weights, rewards, nll, nll_delta, resp_delta: None,
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

    def init_random(
        self,
        xtr,
        phi,
        rollouts,
        num_clusters,
        reward_range,
        with_resp=False,
    ):
        """Initialize mixture model uniform randomly

        For random initialisation the generative model samples mode weights first from the
            Dirichlet distribution, then samples trajectory weights for each mode. We do
            things in this order, because the reverse leads to a strong bias toward uniform
            mode weights.

            Finally, .mstep() is called to initialise the reward parameters from the
            responsibility matrix.

        Args:
            xtr (mdp_extras.DiscreteExplicitExtras): MDP Extras
            phi (mdp_extras.FeatureFunction): Feature function
            rollouts (list): List of (s, a) rollouts
            num_clusters (int): Number of mixture components
            reward_range (tuple): Lower and upper reward function parameter bounds

            with_resp (bool): Also return responsibility matrix as first return variable

        Returns:
            (numpy array): Initial mixture component weights
            (list): List of mdp_extras.Linear initial reward functions
        """
        """mode_weights = dirichlet([1.0 / num_clusters for _ in range(num_clusters)]).rvs(1)[0]

        soft_initial_clusters = np.zeros((len(rollouts), num_clusters))
        for wi, w in enumerate(mode_weights):            
            trajectory_weights = dirichlet(
                [1.0 / len(rollouts) for _ in range(len(rollouts))]
            ).rvs(1)[0]
            print("trajectory_weights.shape", trajectory_weights.shape)
            print("trajectory_weights", trajectory_weights)
            soft_initial_clusters[:, wi] = w * trajectory_weights            
        soft_initial_clusters /= np.sum(soft_initial_clusters, axis=1, keepdims=True)        

        print("Random\n", soft_initial_clusters, "\n", mode_weights)"""
        alpha = np.ones(num_clusters)

        def rand_simplex(num_samples):
            # Draw samples from unit simplex
            return dirichlet.rvs(size=num_samples, alpha=alpha)

        soft_initial_clusters = np.zeros((len(rollouts), num_clusters))
        for i in range(len(rollouts)):
            soft_initial_clusters[i] = rand_simplex(num_clusters)[0]

        mode_weights = rand_simplex(1)[0]
        print("soft_initial_clusters\n", soft_initial_clusters, "\n", mode_weights)

        # Compute initial rewards
        rewards = self.mstep(
            xtr, phi, soft_initial_clusters, rollouts, reward_range=reward_range
        )

        if not with_resp:
            return mode_weights, rewards
        else:
            return soft_initial_clusters, mode_weights, rewards

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
            with_resp (bool): Also return responsibility matrix as first return variable

        Returns:
            (numpy array): Initial mixture component weights
            (list): List of mdp_extras.Linear initial reward functions
        """

        feature_mat = np.array([phi.onpath(r, xtr.gamma) for r in rollouts])

        km = KMeans(n_clusters=num_clusters, n_init=num_restarts)
        hard_initial_clusters = km.fit_predict(feature_mat)
        soft_initial_clusters = np.zeros((len(rollouts), num_clusters))
        for idx, clstr in enumerate(hard_initial_clusters):
            soft_initial_clusters[idx, clstr] = 1.0

        # Compute initial mode weights from soft clustering
        mode_weights = np.sum(soft_initial_clusters, axis=0) / len(rollouts)

        print("KMeans\n", soft_initial_clusters, "\n", mode_weights)

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
            with_resp (bool): Also return responsibility matrix as first return variable

        Returns:
            (numpy array): Initial mixture component weights
            (list): List of mdp_extras.Linear initial reward functions
        """

        feature_mat = np.array([phi.onpath(r, xtr.gamma) for r in rollouts])

        gmm = GaussianMixture(n_components=num_clusters, n_init=num_restarts)
        gmm.fit(feature_mat)
        soft_initial_clusters = gmm.predict_proba(feature_mat)

        # Compute initial mode weights from soft clustering
        mode_weights = np.sum(soft_initial_clusters, axis=0) / len(rollouts)

        print("GMM\n", soft_initial_clusters, "\n", mode_weights)

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
        post_it=lambda solver, iteration, resp, mode_weights, rewards, nll, nll_delta, resp_delta: None,
        parallel_executor=None,
        method="L-BFGS-B",
        min_path_length=None,
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
            method (str): Optimizer to use. Options are;
                - SLSQP - Seems to work well, even for challenging problems
                - CMA-ES - Seems to work well, even for challenging problems
                - L-BFGS-B - Seems to diverge on difficult problems (line search fails)
                - TNC - Seems to diverge on difficult problems
                - trust-const - Seems to diverge on difficult problems
            min_path_length (int): If provided, use this minimum path length for MaxEnt
                calculations - this will force 2-point gradient estimation, making
                making optimization slower.
        """
        super().__init__(minimize_kwargs, minimize_options, pre_it, post_it)
        self.parallel_executor = parallel_executor
        self.method = method
        self.min_path_length = min_path_length

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
        proc_one = lambda xtr, phi, mode_weight, mode_reward, demonstrations: (
            np.log(mode_weight)
            + maxent_path_logprobs(xtr, phi, mode_reward, demonstrations)
        )
        if self.parallel_executor is None:
            resp = np.ones((len(demonstrations), num_modes))
            for mode_idx, (mode_weight, mode_reward) in enumerate(weights_rewards):
                resp[:, mode_idx] = proc_one(
                    xtr, phi, mode_weight, mode_reward, demonstrations
                )
        else:
            # Parallelize execution over modes
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

        resp_max = np.max(resp, axis=1)
        resp_min = np.min(resp, axis=1)
        resp_range = np.max(resp_max - resp_min)
        dtype_exp_precision = np.abs(np.log(np.finfo(resp.dtype).eps))
        if resp_range > dtype_exp_precision:
            warnings.warn(
                f"Responsibility matrix rows vary in log-magnitude by up to {resp_range}, which is larger than the responsibility matrix's dtype ({resp.dtype}) can consistently represent after exponentiating ({dtype_exp_precision}) - algorithm may be numerically unstable. Consider using shorter path lengths or better reward initializations to get larger path likelihoods"
            )

        # Exponentiate the log weights
        resp = np.exp(resp - np.max(resp, axis=1, keepdims=True))

        # Convert log weights to probabilities with SoftMax
        # Each demonstration gets a mass of 1 to allocate between modes
        resp /= np.sum(resp, axis=1, keepdims=True)

        return resp

    def mstep(self, xtr, phi, resp, demonstrations, reward_range=None):
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

        reward_parameter_bounds = None
        if reward_range is not None:
            reward_parameter_bounds = tuple(reward_range for _ in range(len(phi)))

        theta0 = [np.zeros(len(phi)) for i in range(num_modes)]

        if len(demonstrations) == 1:
            max_path_length = max([len(r) for r in demonstrations])
        else:
            max_path_length = max(*[len(r) for r in demonstrations])

        def proc_one(
            xtr,
            phi,
            max_path_length,
            demonstrations,
            reward_parameter_bounds,
            minimize_options,
            minimize_kwargs,
            rollout_weights,
            theta0,
            method="L-BFGS-B",
            min_path_length=None,
        ):
            phi_bar = phi.demo_average(
                demonstrations,
                gamma=xtr.gamma,
                weights=(rollout_weights / np.sum(rollout_weights)),
            )

            nll_only = False
            jac = True
            if min_path_length is not None and min_path_length != 1:
                warnings.warn(
                    "min_path_length != 1 - reverting to two-point gradient estimation. This will make optimization slower"
                )
                jac = "2-point"
                nll_only = True

            if method == "CG":
                res_cg = minimize(
                    sw_maxent_irl,
                    theta0,
                    args=(
                        xtr,
                        phi,
                        phi_bar,
                        max_path_length,
                        nll_only,
                        min_path_length,
                    ),
                    method="Newton-CG",
                    jac=jac,
                    options={},
                    **(minimize_kwargs),
                )
                x_star = res_cg.x
            elif method == "L-BFGS-B":
                res_lbfgs = minimize(
                    sw_maxent_irl,
                    theta0,
                    args=(
                        xtr,
                        phi,
                        phi_bar,
                        max_path_length,
                        nll_only,
                        min_path_length,
                    ),
                    method="L-BFGS-B",
                    jac=jac,
                    bounds=reward_parameter_bounds,
                    options={},
                    **(minimize_kwargs),
                )
                x_star = res_lbfgs.x
            elif method == "TNC":
                res_tnc = minimize(
                    sw_maxent_irl,
                    theta0,
                    args=(
                        xtr,
                        phi,
                        phi_bar,
                        max_path_length,
                        nll_only,
                        min_path_length,
                    ),
                    method="TNC",
                    jac=jac,
                    bounds=reward_parameter_bounds,
                    options=minimize_options,
                    **(minimize_kwargs),
                )
                x_star = res_tnc.x
            elif method == "SLSQP":
                res_slsqp = minimize(
                    sw_maxent_irl,
                    theta0,
                    args=(
                        xtr,
                        phi,
                        phi_bar,
                        max_path_length,
                        nll_only,
                        min_path_length,
                    ),
                    method="SLSQP",
                    jac=jac,
                    bounds=reward_parameter_bounds,
                    options={},
                    **(minimize_kwargs),
                )
                x_star = res_slsqp.x
            elif method == "trust-const":
                res_trust_const = minimize(
                    sw_maxent_irl,
                    theta0,
                    args=(
                        xtr,
                        phi,
                        phi_bar,
                        max_path_length,
                        nll_only,
                        min_path_length,
                    ),
                    method="trust-constr",
                    jac=jac,
                    bounds=reward_parameter_bounds,
                    options=minimize_options,
                    **(minimize_kwargs),
                )
                x_star = res_trust_const.x
            elif method == "CMA-ES":
                std_dev_init_val = 0.5
                es = cma.CMAEvolutionStrategy(
                    theta0,
                    std_dev_init_val,
                    {"bounds": list(zip(*reward_parameter_bounds)), "verbose": -9},
                )
                es.optimize(
                    sw_maxent_irl,
                    args=(xtr, phi, phi_bar, max_path_length, True, min_path_length),
                )
                x_star = es.result[0]
            else:
                raise ValueError

            return Linear(x_star)

        demo_weights_theta0s = zip(resp.T, theta0)
        rewards = []
        if self.parallel_executor is None:

            i = 0
            for m_idx, (mode_demo_weights, mode_theta0) in enumerate(
                demo_weights_theta0s
            ):
                reward = proc_one(
                    xtr,
                    phi,
                    max_path_length,
                    demonstrations,
                    reward_parameter_bounds,
                    self.minimize_options,
                    self.minimize_kwargs,
                    mode_demo_weights,
                    mode_theta0,
                    method=self.method,
                    min_path_length=self.min_path_length,
                )
                rewards.append(reward)
                i += 1
        else:
            tasks = {
                self.parallel_executor.submit(
                    proc_one,
                    xtr,
                    phi,
                    max_path_length,
                    demonstrations,
                    reward_parameter_bounds,
                    self.minimize_options,
                    self.minimize_kwargs,
                    mode_demo_weights,
                    mode_theta0,
                    self.method,
                    self.min_path_length,
                )
                for mode_demo_weights, mode_theta0 in demo_weights_theta0s
            }
            for future in futures.as_completed(tasks):
                # Use arg or result here if desired
                # arg = tasks[future]
                rewards.append(future.result())

        return rewards

    def mixture_nll(self, xtr, phi, mode_weights, rewards, demonstrations):
        """Find the average negative log-likelihood of a MaxEnt mixture model

        Args:
            xtr (): Extras object
            phi (): Features object
            mode_weights (list): List of prior probabilities for each mode
            rewards (): reward
            demonstrations (list): List of state-action rollouts

        Returns:
            (float): Negative Log Likelihood of the rollouts under the given mixture model
        """

        # Pre-compute path probabilities under each mode (faster this way)
        # We work in log-space and apply the Log-Sum-Exp trick to avoid overflow
        # Shape is (modes) x (paths)
        path_log_probs = np.array(
            [
                np.log(mode_weight)
                + maxent_path_logprobs(xtr, phi, reward, demonstrations)
                for mode_weight, reward in zip(mode_weights, rewards)
            ]
        )

        # Apply LSE trick individually for each path here
        max_logprob_per_path = np.max(path_log_probs, axis=0, keepdims=True)
        max_logprob_per_path_flat = np.max(path_log_probs, axis=0)
        path_mode_probs = np.exp(path_log_probs - max_logprob_per_path)
        mode_probs = np.sum(path_mode_probs, axis=0)
        trajectory_mixture_lls = np.log(mode_probs) + max_logprob_per_path_flat

        # Take the average LL across all trajectories
        mixture_ll = np.mean(trajectory_mixture_lls)

        return -1.0 * mixture_ll


class MaxLikEMSolver(EMSolver):
    """Solve an EM MM-IRL problem with MaxLikelihood IRL"""

    def __init__(
        self,
        boltzman_scale=0.5,
        qge_tol=1e-3,
        minimize_kwargs={},
        minimize_options={},
        pre_it=lambda i: None,
        post_it=lambda solver, iteration, resp, mode_weights, rewards, nll, nll_delta, resp_delta: None,
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
            _, q_star = vi(xtr, phi, reward)
            pi = BoltzmannExplorationPolicy(q_star, scale=self._boltzman_scale)
            for rollout_idx, rollout in enumerate(rollouts):
                resp[rollout_idx, mode_idx] = mode_weight * np.exp(
                    pi.path_log_likelihood(rollout)
                )

        # Each demonstration gets a mass of 1 to allocate between modes
        resp /= np.sum(resp, axis=1, keepdims=True)

        return resp

    def mstep(
        self,
        xtr,
        phi,
        resp,
        rollouts,
        reward_range=None,
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
                    rollout_weights / np.sum(rollout_weights),
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

    def mixture_nll(self, xtr, phi, mode_weights, rewards, demonstrations):
        """Find the average negative log-likelihood of a MaxLikelihood mixture model

        Args:
            env (explicit_env.IExplicitEnv): Environment defining dynamics
            demonstrations (list): List of state-action rollouts
            mode_weights (list): List of prior probabilities for each mode

        Returns:
            (float): Log Likelihood of the rollouts under the given mixture model
        """

        # Compute Boltzman probabilities
        exp_beta_q_stars = [
            np.exp(self._boltzman_scale * vi(xtr, phi, reward)[1]) for reward in rewards
        ]
        boltzman_policy_probs = [
            exp_beta_q_star / np.sum(exp_beta_q_star, axis=1, keepdims=True)
            for exp_beta_q_star in exp_beta_q_stars
        ]

        # Pre-compute path probabilities under each mode (faster this way)
        path_qs = np.exp(
            np.array([xtr.path_log_probability(demo) for demo in demonstrations])
        )

        path_probs = []
        for mode_idx, reward in enumerate(rewards):
            mode_path_probs = []
            for demo_idx, demo in enumerate(demonstrations):
                path_prob = path_qs[demo_idx] * np.product(
                    [boltzman_policy_probs[mode_idx][s, a] for (s, a) in demo[:-1]]
                )
                mode_path_probs.append(path_prob)
            path_probs.append(mode_path_probs)
        path_probs = np.array(path_probs)

        trajectory_mixture_lls = []
        for demo_idx in range(len(demonstrations)):
            # Sum at this level, then take log at this level
            trajectory_mixture_lls.append(
                np.log(
                    np.sum(
                        [
                            mode_weight * path_probs[mode_idx, demo_idx]
                            for mode_idx, mode_weight in enumerate(mode_weights)
                        ]
                    )
                )
            )

        mixture_ll = np.mean(trajectory_mixture_lls)

        return -1.0 * mixture_ll


class SigmaGIRLEMSolver(EMSolver):
    """Solve an EM MM-IRL problem with multiple-intention Sigma-GIRL IRL

    I.e. Algorithm 1 from
     * Ramponi, Giorgia, et al. "Truly Batch Model-Free Inverse Reinforcement Learning about Multiple Intentions."
       International Conference on Artificial Intelligence and Statistics. PMLR, 2020.

    """

    def __init__(
        self,
        PolicyClass,
        mstep_policy_kwargs={},
        mstep_bc_restarts=5,
        mstep_opt_restarts=50,
        mstep_num_bc_epochs=500,
        minimize_kwargs={},
        minimize_options={},
        pre_it=lambda i: None,
        post_it=lambda solver, iteration, resp, mode_weights, rewards, nll, nll_delta, resp_delta: None,
    ):
        """C-tor

        Args:
            PolicyClass (mdp_extras.TorchPolicy): Policy class that can be called to construct new policy
                objects using the mdp_extras.TorchPolicy API.

            mstep_policy_kwargs (dict): Keyword arguments to pass to PolicyClass constructor, e.g. this should
                specify the output (action) dimension of the policy, and the number of hidden units, but *not* the
                input dimension, as this will be over-written in the .mstep() method
            mstep_bc_restarts (int): Number of random re-starts to perform for each mode during the M-Step (behaviour cloning part)
            mstep_opt_restarts (int): Number of random re-starts to perform for each mode during the M-Step (reward optimization part)
            mstep_num_bc_epochs (int): Number of behaviour clonining epochs to perform for each mode during
                the M-Step

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

        self.PolicyClass = PolicyClass
        self.mstep_policy_kwargs = mstep_policy_kwargs
        self.mstep_restarts = mstep_opt_restarts
        self.mstep_bc_restarts = mstep_bc_restarts
        self.mstep_num_bc_epochs = mstep_num_bc_epochs

        # Internal storage for latest policy, optimal jacobian mean, and jacobian covariance matrices
        # Each list holds one item for each behaviour intent
        # These items are updated each time the mstep method is called
        # XXX ajs 28/May/2021 This assumes that the rewards returned from mstep don't get re-ordered outside this class!!!
        self._policies = None
        self._opt_jac_means = None
        self._jac_covs = None

    def estep(self, xtr, phi, mode_weights, rewards, rollouts):
        """Compute responsibility matrix using current SigmaGIRL rewards

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

        # Step 1 - we mirror the feature function
        phi_mirrored = MirrorWrap(phi)

        # XXX ajs 28/May/2021 We assume that these items (computed during the previous .mstep()) correspond to the
        # reward list we are passed as an argument
        policies = self._policies
        opt_jac_means = self._opt_jac_means
        jac_covs = self._jac_covs

        num_modes = len(mode_weights)

        # Shortcut for K=1
        if num_modes == 1:
            return np.array([np.ones(len(rollouts))]).T

        resp = np.ones((len(rollouts), num_modes))
        for mode_idx, (mode_weight, pi, opt_jac_mean, jac_cov) in enumerate(
            zip(mode_weights, policies, opt_jac_means, jac_covs)
        ):
            print(f"Mode {mode_idx + 1} - computing demo jacobians")
            # Compute jacobian for each demonstration trajectory
            demo_jacs = np.array(
                [traj_jacobian(pi, phi_mirrored, d, xtr.gamma) for d in rollouts]
            )

            # Compute path log likelihoods under this reward
            print(f"Mode {mode_idx + 1} - computing demo log likelihoods")
            path_lls = gradient_path_logprobs(opt_jac_mean, jac_cov, demo_jacs)

            # Store this column in the responsibility matrix
            path_probs = mode_weight * np.exp(path_lls)
            resp[:, mode_idx] = path_probs

        # Each demonstration gets a mass of 1 to allocate between modes
        resp /= np.sum(resp, axis=1, keepdims=True)

        # If there were values that recieved 0 probability mass from all modes, set them to uniform prob.
        resp = np.nan_to_num(resp, nan=1.0 / resp.shape[1])

        return resp

    def mstep(
        self,
        xtr,
        phi,
        resp,
        rollouts,
        reward_range=None,
    ):
        """Compute reward parameters given responsibility matrix

        Internally, we mirror and un-mirror the feature function using mdp_extras.MirrorWrap because Sigma-GIRL only
        supports reward parameters on a probability simplex. This process should be transparent to the calling method.

        Args:
            xtr (DiscreteExplicitExtras): Extras object for multi-modal MDP
            phi (FeatureFunction): Feature function for multi-modal MDP
            resp (numpy array): Responsibility matrix
            rollouts (list): Demonstration data

            reward_range (tuple): Optional reward parameter min and max values

        Returns:
            (list): List of Linear reward functions
        """

        # Step 1 - we mirror the feature function
        phi_mirrored = MirrorWrap(phi)

        num_rollouts, num_modes = resp.shape

        # from multimodal_irl.envs import ElementWorldEnv

        policies = []
        rewards_mirrored = []
        opt_jac_means = []
        jac_covs = []
        for mode_idx in range(num_modes):
            print(f"M-Step - Intent {mode_idx+1}")

            # Slice demonstration weights out
            rollout_weights = resp[:, mode_idx]

            # Generate a new policy
            print(f"M-Step - BC")
            self.mstep_policy_kwargs["in_dim"] = len(phi_mirrored)
            pics = []
            losses = []
            for _ in range(self.mstep_bc_restarts):
                pic = self.PolicyClass(**self.mstep_policy_kwargs)
                loss = pic.behaviour_clone(
                    rollouts,
                    phi_mirrored,
                    num_epochs=self.mstep_num_bc_epochs,
                    weights=rollout_weights,
                )
                # print(loss)
                # print(
                #     np.array(
                #         [
                #             ElementWorldEnv(
                #                 num_elements=resp.shape[1]
                #             ).ACTION_SYMBOLS_A2SYM[
                #                 int(pic.predict((phi_mirrored(s)), stoch=False)[0])
                #             ]
                #             for s in xtr.states
                #         ]
                #     ).reshape((int(sum(xtr.terminal_state_mask)), -1))
                # )
                pics.append(pic)
                losses.append(loss)
            pi = pics[np.argmin(losses)]
            policies.append(pi)

            # print("============ Selected policy")
            # print(
            #     np.array(
            #         [
            #             ElementWorldEnv(
            #                 num_elements=resp.shape[1]
            #             ).ACTION_SYMBOLS_A2SYM[
            #                 int(pi.predict((phi_mirrored(s)), stoch=False)[0])
            #             ]
            #             for s in xtr.states
            #         ]
            #     ).reshape((int(sum(xtr.terminal_state_mask)), -1))
            # )

            # Find weighted data jacobian
            print(f"M-Step - form_jacobian")
            jac_mean, jac_cov, p_mat = form_jacobian(pi, phi_mirrored, rollouts)
            jac_covs.append(jac_cov)
            d, q = jac_mean.shape

            # Run optimization here
            print(f"M-Step - optimize")
            evaluations = []
            while len(evaluations) < self.mstep_restarts - 1:
                # Choose random initial guess
                x0 = np.random.uniform(0, 1, q)
                x0 = x0 / np.sum(x0)

                res = minimize(
                    pi_gradient_irl,
                    x0,
                    method="SLSQP",
                    constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1},
                    bounds=[(0.0, 1.0)] * len(x0),
                    args=(p_mat),
                    options={"ftol": 1e-8, "disp": False},
                )
                if res.success:
                    # If the optimization was successful, save it
                    evaluations.append([res.x, res.fun])
            params, losses = zip(*evaluations)
            reward_weights = params[np.argmin(losses)]
            rewards_mirrored.append(Linear(reward_weights))

            # Find weighted data optimal jacobian mean
            print(f"M-Step - optimal_jacobian_mean")
            opt_jac_mean = optimal_jacobian_mean(jac_mean, jac_cov, reward_weights)
            opt_jac_means.append(opt_jac_mean)

        # Store new policies, optimal jacobian means, and jacobian covariances
        del self._policies
        self._policies = policies
        del self._opt_jac_means
        self._opt_jac_means = opt_jac_means
        del self._jac_covs
        self._jac_covs = jac_covs

        # Final step - Un-mirror the learned rewards and re-scale to the requested reward range
        rewards = []
        for r_mirrored in rewards_mirrored:
            r = phi_mirrored.unupdate_reward(r_mirrored)
            r_param = r.theta
            r_param_scaled = np.interp(
                r_param, (r_param.min(), r_param.max()), reward_range
            )
            rewards.append(Linear(r_param_scaled))

        return rewards

    def mixture_nll(
        self,
        xtr,
        phi,
        mode_weights,
        rewards,
        rollouts,
    ):
        """Find the average negative log-likelihood of a MaxLikelihood mixture model

        Args:
            xtr (mdp_extras.DiscreteExplicitExtras): MDP definition
            phi (mdp_extras.FeatureFunction): Feature function
            mode_weights (list): List of prior probabilities, one for each mode
            reawards (list): List of Linear rewards, one for each mode
            rollouts (list): List of state-action rollouts

        Returns:
            (float): Log Likelihood of the rollouts under the given mixture model
        """

        # Step 1 - we mirror the feature function
        phi_mirrored = MirrorWrap(phi)

        # XXX ajs 28/May/2021 We assume that these items correspond to the reward list we are passed
        policies = self._policies
        opt_jac_means = self._opt_jac_means
        jac_covs = self._jac_covs

        # Pre-compute path probabilities under each mode
        path_probs = []
        for pi, opt_jac_mean, jac_cov in zip(policies, opt_jac_means, jac_covs):
            rollout_jacobians = np.array(
                [traj_jacobian(pi, phi_mirrored, traj, xtr.gamma) for traj in rollouts]
            )
            path_probs.append(
                np.exp(gradient_path_logprobs(opt_jac_mean, jac_cov, rollout_jacobians))
            )
        path_probs = np.array(path_probs)

        trajectory_mixture_lls = []
        for demo_idx in range(len(rollouts)):
            # Sum at this level, then take log at this level
            trajectory_mixture_lls.append(
                np.log(
                    np.sum(
                        [
                            mode_weight * path_probs[mode_idx, demo_idx]
                            for mode_idx, mode_weight in enumerate(mode_weights)
                        ]
                    )
                )
            )

        mixture_ll = np.mean(trajectory_mixture_lls)

        return -1.0 * mixture_ll


class MeanOnlyEMSolver(MaxEntEMSolver):
    """Approximates a MaxEnt solver by picking the feature expectation at every mstep"""

    def mstep(
        self,
        xtr,
        phi,
        resp,
        demonstrations,
        reward_range=None,
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

        # First compute the mean over all demonstrations - used for centering
        phi_mean = phi.demo_average(demonstrations, gamma=xtr.gamma)

        feature_vecs = np.array(
            [phi.onpath(d, gamma=xtr.gamma) for d in demonstrations]
        )
        feature_vecs -= phi_mean

        rewards = []
        for mode_demo_weights in resp.T:
            # Now compute mean using cluster weights
            phi_bar = np.average(
                feature_vecs,
                axis=0,
                weights=(mode_demo_weights / np.sum(mode_demo_weights)),
            )
            rewards.append(Linear(phi_bar))
        return rewards


def bv_em(
    solver,
    xtr,
    phi,
    rollouts,
    num_modes,
    reward_range,
    mode_weights=None,
    rewards=None,
    nll_tolerance=1e-5,
    resp_tolerance=1e-5,
    max_iterations=None,
    break_on_nll_increase=True,
):
    """
    Expectation Maximization Multi-Modal IRL by Babeş-Vroman et al. 2011

    See the paper "Apprenticeship learning about multiple intentions." in ICML 2011.

    Stopping criterion is a logical OR of several options, each of which can be set to
    `None' to disable that check
     - NLL tolerance - stop when the NLL change falls below some threshold
     - Responsibility Matrix tolerance - Stop when the sum of responsibilty matrix entry
        differences falls below some threshold
     - Max Iterations - Stop after this many iterations

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
        nll_tolerance (float): NLL convergence threshold - set to zero to disable this check
        resp_tolerance (float): Responsibility matrix convergence threshold
        max_iterations (int): Maximum number of iterations (alternate stopping criterion)
        break_on_nll_increase (bool): The Mixture NLL can sometimes increase (instead of monotonically decreasing as
            the theory predicts), due to numerical rounding errors, especially with the IRL gradients. If this is set to
            true, the optimization will stop as soon as this occurs. Otherwise, a warning will be raised, but the
            optimization will continue until some other stopping condition is reached.

    Returns:
        (int): Number of EM iterations performed
        (list): List of responsibility matrix (numpy array) at every iteration
        (list): List of cluster weights (numpy array) at the start, and for every iteration
        (list): List of rewards (mdp_extras.Linear) at the start, and for every iteration
        (list): List of NLL (float) at the start, and for every algorithm iteration
        (str): Reason for termination
    """

    # Initialize reward parameters and/or mode weights randomly if not passed
    if mode_weights is None and rewards is None:
        mode_weights, rewards = solver.init_random(
            xtr, phi, rollouts, num_modes, reward_range
        )
    elif mode_weights is None:
        mode_weights, _ = solver.init_random(
            xtr, phi, rollouts, num_modes, reward_range
        )
    elif rewards is None:
        _, rewards = solver.init_random(xtr, phi, rollouts, num_modes, reward_range)

    assert len(mode_weights) == num_modes
    assert len(rewards) == num_modes

    resp_history = []
    resp_delta_history = []
    mode_weights_history = [mode_weights]
    rewards_history = [rewards]
    nll_history = []
    reason = ""

    for iteration in it.count():

        # Call user pre-iteration callback
        solver.pre_it(iteration)

        # Compute NLL
        nll = solver.mixture_nll(xtr, phi, mode_weights, rewards, rollouts)
        nll_history.append(nll)

        nll_delta = np.nan
        if len(nll_history) >= 2 and nll_tolerance != 0.0:
            # Check NLL delta
            nll_delta = np.diff(nll_history)[-1]

        # E-step - update responsibility matrix, mixture component weights
        # print("E-Step")
        resp = solver.estep(xtr, phi, mode_weights, rewards, rollouts)
        resp_history.append(resp)
        # print(resp)

        resp_delta = np.nan
        if len(resp_history) >= 2:
            # Check Responsibility matrix delta
            resp_delta = np.sum(np.abs(resp_history[-1] - resp_history[-2]))
        resp_delta_history.append(resp_delta)

        mode_weights = np.sum(resp, axis=0) / len(rollouts)
        # print("mode_weights", mode_weights)
        mode_weights_history.append(mode_weights)

        # M-step - solve for new reward parameters
        # print("M-Step")
        rewards = solver.mstep(xtr, phi, resp, rollouts, reward_range=reward_range)
        rewards_history.append(rewards)

        # Call user post-iteration callback
        # print(f"End of iteration {len(rewards_history)}")
        solver.post_it(
            solver, iteration, resp, mode_weights, rewards, nll, nll_delta, resp_delta
        )

        # Edge case for only one cluster
        if num_modes == 1:
            reason = "Only one cluster to learn"
            break

        if len(resp_history) >= 2:
            # Check Responsibility matrix delta
            resp_delta = np.sum(np.abs(resp_history[-1] - resp_history[-2]))
            # print("resp_tolerance, resp_delta", (resp_tolerance, resp_delta))

            if resp_tolerance is not None and resp_delta <= resp_tolerance:
                # Responsibility matrix has converged (solution is epsilon optimal)
                reason = "Responsibility matrix has converged: \sum_i, sum_k |u_{ik}^{t+1} - u_{ik}^t| <= tol"
                break

        if len(nll_history) >= 2 and nll_tolerance != 0.0:
            # Check NLL delta
            nll_delta = np.diff(nll_history)[-1]
            # print("nll, nll_tolerance, nll_delta", (nll, nll_tolerance, nll_delta))

            if nll_tolerance is not None and np.abs(nll_delta) <= nll_tolerance:
                # NLL has converged
                reason = "NLL converged: |NLL delta| <= tol"
                break

            if not nll_delta <= 0.0:
                warnings.warn(
                    f"NLL is not monotonically decreasing - possible loss of accuracy due to numerical rounding. NLL Delta = {nll_delta}"
                )
                if break_on_nll_increase:
                    reason = "NLL is not monotonically decreasing"
                    iteration -= 1
                    resp_history = resp_history[:-1]
                    mode_weights_history = mode_weights_history[:-1]
                    rewards_history = rewards_history[:-1]
                    nll_history = nll_history[:-1]
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


import torch
from mdp_extras import TorchPolicy


class EWSimplePolicy(TorchPolicy):
    """A very simple policy that can solve ElementWorld if the params are set right"""

    def __init__(self, in_dim, out_dim, learning_rate=0.01):
        """C-tor

        Args:
            in_dim (int): Input (feature vector) size
            out_dim (int): Output (action vector) size
            hidden_size (int): Size of the hidden layer
            learning_rate (float): Learning rate for optimizer used for training
        """
        super().__init__(in_dim, out_dim, hidden_size=None, learning_rate=learning_rate)

        # self.fc1 = torch.nn.Linear(in_dim, out_dim, bias=False)
        self.out_dim = out_dim

        self.w_start_right = torch.nn.Parameter(torch.rand(1))
        self.w_el_right = torch.nn.Parameter(torch.rand(1))
        self.w_e2_right = torch.nn.Parameter(torch.rand(1))
        self.w_el_up = torch.nn.Parameter(torch.rand(1))
        self.w_e2_up = torch.nn.Parameter(torch.rand(1))

        self.loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")
        self.loss_target_type = torch.long
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        # Input is feature vector phi(s, a, s')
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        x = x.float()
        # x = self.fc1(x)

        if len(x.shape) == 1:
            x = x.reshape((1, -1))

        #
        start = x[:, 0]
        goal = x[:, 1]
        e1 = x[:, 2]
        e2 = x[:, 2]
        up, down, left, right = 0, 1, 2, 3

        #
        y = torch.zeros(x.shape[0], self.out_dim, requires_grad=True) + 1.0
        y[:, right] = (
            self.w_start_right * start + self.w_el_right * e1 + self.w_e2_right * e2
        )
        y[:, up] = self.w_el_right * e1 + self.w_e2_right * e2

        # Output is vector of categorical log probabilities from which we sample an action
        return y

    def predict(self, x, stoch=True):
        """Predict next action and distribution over states

        N.b. This function matches the API of the stabe-baselines policies.

        Args:
            x (int): Input feature vector

            stoch (bool): If true, sample action stochastically

        Returns:
            (int): Sampled action
            (None): Placeholder to ensure this function matches the stable-baselines
                policy interface. Some policies use this argument to return a prediction
                over future state distributions - we do not.
        """
        probs = torch.exp(self(x))
        if stoch:
            dist = torch.distributions.Categorical(probs)
            a = dist.sample()
        else:
            a = torch.argmax(probs)
        return a, None

    def log_prob_for_state(self, x):
        """Get the action log probability vector for the given feature vector

        Args:
            x (numpy array): Current feature vector

        Returns:
            (numpy array): Log probability distribution over actions
        """
        return self(x)

    def log_prob_for_state_action(self, x, a):
        """Get the log probability for the given state, action

        Args:
            x (int): Current feature vector phi(s, a, s')
            a (int): Chosen action

        Returns:
            (float): Log probability of choosing a from phi(s, a, s')
        """
        return self(x)[int(a.item())]


def main():
    """Main function"""

    # Construct env
    from mdp_extras import OptimalPolicy, vi, MLPCategoricalPolicy
    from multimodal_irl.envs import ElementWorldEnv, element_world_extras

    num_elements = 2
    width = 6
    reward_range = (-10.0, 0.0)
    env = ElementWorldEnv(
        num_elements=num_elements, rotate=False, element_zone_size=3, width=width
    )
    xtr, phi, gt_rewards = element_world_extras(env)

    print("GT Rewards:")
    print(gt_rewards[0].theta.reshape((-1, width)))
    print(gt_rewards[1].theta.reshape((-1, width)))

    env.reset()
    print(env.render())

    print("Collecting data")
    # Collect dataset of demonstration (s, a) trajectories from expert
    num_rollouts_per_mode = 5
    rollouts = []
    for reward in gt_rewards:
        _, q_star = vi(xtr, phi, reward)
        pi_star = OptimalPolicy(q_star, stochastic=True)
        rollouts.extend(pi_star.get_rollouts(env, num_rollouts_per_mode))

    def post_it(
        solver, iteration, resp, mode_weights, rewards, nll, nll_delta, resp_delta
    ):
        print(f"Iteration {iteration}")
        print(mode_weights)
        print(nll, nll_delta)
        print(resp)
        print(resp_delta)

    # Prepare solver
    solver = SigmaGIRLEMSolver(
        MLPCategoricalPolicy,
        mstep_policy_kwargs=dict(out_dim=len(xtr.actions), hidden_size=30),
        mstep_num_bc_epochs=1000,
        post_it=post_it,
    )

    mode_weights, rewards = solver.init_kmeans(
        xtr, phi, rollouts, num_elements, reward_range=reward_range
    )
    print("Rewards:")
    print(np.round(rewards[0].theta.reshape((-1, width)), 3))
    print(np.round(rewards[1].theta.reshape((-1, width)), 3))

    bv_em(
        solver,
        xtr,
        phi,
        rollouts,
        num_elements,
        reward_range,
        mode_weights=mode_weights,
        rewards=rewards,
        max_iterations=None,
        nll_tolerance=0.0,
        break_on_nll_increase=True,
    )

    # print("Policy 1")
    # print(
    #     np.array(
    #         [
    #             env.ACTION_SYMBOLS_A2SYM[
    #                 int(policies[0].predict((phi(s)), stoch=False)[0])
    #             ]
    #             for s in xtr.states
    #         ]
    #     ).reshape(-1, 6)
    # )

    print("Here")


if __name__ == "__main__":
    main()
