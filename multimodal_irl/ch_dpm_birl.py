"""Implementation of Dirichlet Process Mixture Bayesian IRL by Choi and Kim, 2012"""

import random
import numpy as np

from scipy.stats import dirichlet, multivariate_normal, multinomial, norm

from mdp_extras import vi, BoltzmannExplorationPolicy, Linear, q_grad_fpi


def cluster_compaction(clusters, *args):
    """Re-name clusters, dropping any empty ones

    Args:
        clusters (list): List of cluster ID for each demonstration

        *args: Any number of enumerable objects of length len(clusters), that will also
            be re-ordered and returned

    Returns:
        (list): List of cluster IDs for each demonstration - clusters will have been
            renamed to start at 0 and not have any empty clusters
    """
    old_clusters = clusters.copy()
    num_actual_clusters = len(set(clusters))

    # Compute renaming map
    cluster_rename_map = {
        k: v for k, v in zip(sorted(set(clusters)), range(num_actual_clusters))
    }
    cluster_rename_reverse_map = {v: k for k, v in cluster_rename_map.items()}

    # Re-name clusters
    clusters = np.array([cluster_rename_map[c] for c in old_clusters])

    # Also re-order any other passed lists
    new_args = []
    for arg in args:
        new_arg = []
        for cluster in sorted(set(clusters)):
            new_arg.append(arg[cluster_rename_reverse_map[cluster]])
        if isinstance(arg, np.ndarray):
            new_arg = np.array(new_arg)
        new_args.append(new_arg)

    if len(args) == 0:
        return clusters
    else:
        return [clusters] + new_args


def log_urpd(
    xtr,
    reward,
    q_star,
    demos,
    reward_prior_log_pdf,
    reward_prior_confidence,
):
    """Compute the log of the un-normalized reward posterior distribution

    Args:
        xtr (mdp_extras.Extras): MDP Extras
        reward (numpy array): Reward vector
        q_star (numpy array): Optimal Q function for reward parameters \theta
        demos (list): List of (s, a) demos
        reward_prior_log_pdf (callable):
        reward_prior_confidence (float): Boltzmann policy confidence factor

    Returns:
        (float): Un-normalized reward
    """
    rlp_nonorm = 0.0

    # Add contribution from demos in this cluster
    for cluster_demo in demos:
        # XXX ajs Skipping end state means we don't catch reward contribution from goal states?
        for t, (st, at) in enumerate(cluster_demo[:-1]):
            aprime_norm_factor = np.log(
                np.sum(
                    [
                        np.exp(reward_prior_confidence * q_star[st, aprime])
                        for aprime in xtr.actions
                    ]
                )
            )
            rlp_nonorm += reward_prior_confidence * q_star[st, at] - aprime_norm_factor

    # Add contribution from reward prior
    rlp_nonorm += reward_prior_log_pdf(reward)

    return rlp_nonorm


def log_urpd_grad(
    xtr,
    reward,
    q_grad,
    demos,
    reward_prior_log_pdf_grad,
    reward_prior_confidence,
):
    """Compute the gradient of the log of the un-normalized reward posterior distribution

    Args:
        xtr (mdp_extras.Extras): MDP Extras
        reward (numpy array): Reward vector
        q_grad (numpy array): Gradient of Q function wrt. reward parameters \theta
        demos (list): List of (s, a) demos
        reward_prior_log_pdf_grad (callable): Callable that returns the reward prior log pdf gradient wrt. reward
            parameters
        reward_prior_confidence (float): Boltzmann policy confidence factor

    Returns:
        (numpy array): Gradient of the log of the un-normalized reward posterior distribution,
            with respect to reward parameters \theta
    """

    log_urpd_grad = np.zeros(len(reward))
    with np.errstate(divide="raise"):

        # Add contribution from demos in this cluster
        for cluster_demo in demos:
            # XXX ajs Skipping end state means we don't catch reward contribution from goal states?
            for t, (st, at) in enumerate(cluster_demo[:-1]):
                aprime_norm_factor = [
                    reward_prior_confidence * q_grad[st, aprime, :]
                    for aprime in xtr.actions
                ]
                aprime_norm_factor = np.sum(aprime_norm_factor, axis=0)
                contrib = (
                    reward_prior_confidence * q_grad[st, at, :] - aprime_norm_factor
                )
                log_urpd_grad += contrib

    # Add contributions from reward prior
    log_urpd_grad += reward_prior_log_pdf_grad(reward)

    return log_urpd_grad


def log_g(r1, log_urpd_grad_r1, r2, scale):
    """The log of the g(x, y) fn from the original Roberts and Rosenthal paper

    Choi and Kim get this equation wrong in the paper - they mistakenly subtract r2 from r1, but it should be
    the other way around, according to the original Langevin paper by Roberts and Rosenthal

    Args:
        r1 (numpy array): First reward vector
        log_urpd_grad_r1 (numpy array): Gradient of the log of the un-normalized reward
            posterior distribution, evaluated at the first reward vector r1.
        r2 (numpy array): Second reward vector
        scale (float): Scale parameter for langevin step size

    Returns:
        (float):
    """
    return (np.log(1.0 / (2 * np.pi * scale ** 2) ** (len(r1) / 2))) + (
        -1.0
        / (2 * scale ** 2)
        * (np.linalg.norm(r2 - r1 - 0.5 * (scale ** 2) * log_urpd_grad_r1) ** 2)
    )


def reward_improvement_step(reward, reward_log_gradient, scale):
    """Take a single step of IRL to improve a reward function

    Args:

    Returns:
        (numpy array):
    """
    reward_new = (
        reward
        + scale ** 2 / 2.0 * reward_log_gradient
        + scale * norm(0, 1).rvs(len(reward))
    )
    return reward_new


def demo_reallocation_probabilities(all_clusters, current_cluster, concentration=1.0):
    """Get vector of cluster re-allocation probabilities

    Args:
        all_clusters (list): The list of all demonstration clusters (one int per demo)
        current_cluster (int): The cluster of the trajectory we are currently re-allocating
        concentration (float): Dirichlet concentration parameter 'alpha'
    """
    cluster_probs = [
        np.sum(all_clusters == cluster_idx) for cluster_idx in set(all_clusters)
    ]
    cluster_probs[current_cluster] -= 1
    cluster_probs.append(concentration)
    cluster_probs = np.array(cluster_probs) / np.sum(cluster_probs)
    return cluster_probs


def ch_dpm_birl(
    xtr,
    phi,
    demonstrations,
    initial_clusters,
    initial_rewards,
    max_iterations,
    reward_prior_sample,
    reward_prior_log_pdf,
    reward_prior_log_pdf_grad,
    dirichlet_prior_concentration=1.0,
    reward_prior_confidence=1.0,
    langevin_scale_param=0.001,
    reward_bounds=(None, None),
):
    """A Metropolis Hastings algorithm for DPM-BIRL

    TODO check this implementation against the following;
     * https://github.com/erensezener/IRL_Algorithms/tree/ea19532d61f229f03e254f1ba938deb814e2aca7/irl_multiple_experts/DPM_BIRL
     * https://github.com/s-arora-1987/sawyer_i2rl_project_workspace

    TODO ajs 3/Feb/21 This function leaks memory - after 3000 iterations it is hogging 15Gb.

    Args:
        xtr (mdp_extras.Extras): MDP Extras
        phi (mdp_extras.FeatureFunction): Feature function object
        demonstrations (list): List of (s, a) demonstration trajectories
        initial_clusters (list): Initial cluster allocations (one int for each demonstration)
        initial_rewards (numpy array): Array of initial reward parameters - one row for each initial reward function
        max_iterations (int): Maximum number of MH iterations
        reward_prior_sample (callable): A lambda that returns a single sample from the reward prior
        reward_prior_log_pdf (callable): A lambda that returns the reward prior log pdf for a reward parameter vector
        reward_prior_log_pdf_grad (callable): A lambda that returns the reward prior log pdf gradient wrt. a reward
            vector

        dirichlet_prior_concentration (float): Dirichlet distribution concentration
            parameter - set to 1.0 for uninformed prior
        reward_prior_confidence (float): Boltzmann confidence parameter for MaxLikelihood IRL
            model
        langevin_scale_param (float): Positive scale for the Langevin dynamics step size - should be
            tuned down and/or up until the average acceptance probability for the MH process is ~0.574.
            Acceptance probabilities are inversely proportional to this parameter's size.
        reward_bounds (tuple): Tuple of low, high reward parameter bounds. Set the respective entry to None to
            remove that reward bound.

    Returns:
        (numpy array): Cluster indices for each demonstration
        (numpy array): Array of learned reward functions
    """

    assert (
        len(initial_rewards.shape) > 1
    ), "Passed initial rewards must be two-dimensional"

    # THis is the optimal acceptance ratio for Langevin dynamics MH-MCMC (see paper by R & R).
    TARGET_REWARD_ACCEPTANCE_PROB_RATIO = 0.574

    # Sanitize initial clusters and rewards
    clusters = cluster_compaction(initial_clusters)
    print("Initial clusters:")
    print(clusters)

    rewards = np.clip(initial_rewards, *reward_bounds)
    print("Initial Rewards:")
    print(rewards)

    # Solve for Boltzmann Policy for each reward parameter
    boltzmann_policies = [
        BoltzmannExplorationPolicy(vi(xtr, phi, Linear(r))[1], reward_prior_confidence)
        for r in rewards
    ]

    # Loop until max number of iterations
    log_acceptance_probs = []
    for t in range(max_iterations):

        print(f"Iteration {t+1}")

        # Loop over each demonstration
        # We assume the list of clusters is always compact
        # print("Updating demonstration memberships...")
        for demo_idx, demo in enumerate(demonstrations):
            demo_cluster = clusters[demo_idx]
            demo_cluster_boltzmann_policy = boltzmann_policies[demo_cluster]

            # TODO ajs sample new cluster assignment from the full conditional posterior

            # Sample a new cluster for this trajectory from Eq 5
            cluster_probs = demo_reallocation_probabilities(
                clusters, demo_cluster, dirichlet_prior_concentration
            )
            demo_cluster_new = np.random.choice(
                list(range(len(cluster_probs))), p=cluster_probs
            )

            if demo_cluster_new == demo_cluster:
                # We didn't move the demonstration - nothing doing
                continue
            elif demo_cluster_new not in clusters:
                # We selected a new/empty cluster, sample a new reward function from the prior and apply bounds
                demo_cluster_new_reward = reward_prior_sample()
                demo_cluster_new_reward = np.clip(
                    demo_cluster_new_reward, *reward_bounds
                )
                demo_cluster_boltzmann_policy_new = BoltzmannExplorationPolicy(
                    vi(xtr, phi, Linear(demo_cluster_new_reward))[1],
                    reward_prior_confidence,
                )
            else:
                # We selected to move the demonstration to an existing cluster
                demo_cluster_new_reward = rewards[demo_cluster_new]
                demo_cluster_boltzmann_policy_new = boltzmann_policies[demo_cluster_new]

            # Compute acceptance probability under Eq. 5 (in log space)
            if cluster_probs[demo_cluster_new] == 0.0:
                loga = -np.inf
            else:
                loga = np.log(
                    cluster_probs[demo_cluster_new]
                ) + demo_cluster_boltzmann_policy_new.path_log_action_probability(demo)

            if cluster_probs[demo_cluster] == 0.0:
                logb = -np.inf
            else:
                logb = np.log(
                    cluster_probs[demo_cluster]
                ) + demo_cluster_boltzmann_policy.path_log_action_probability(demo)

            if np.isneginf(logb):
                acceptance_logprob_ratio = np.inf
            else:
                acceptance_logprob_ratio = loga - logb
            acceptance_logprob = min(np.log(1.0), acceptance_logprob_ratio)
            acceptance_prob = np.exp(acceptance_logprob)

            # Accept/reject the new cluster+reward assignment
            if np.random.rand() <= acceptance_prob:

                if demo_cluster_new not in clusters:
                    # Accept the new cluster and reward
                    clusters[demo_idx] = demo_cluster_new

                    # We spawned a new cluster - add it to the reward list
                    rewards = np.concatenate(
                        (rewards, [demo_cluster_new_reward]), axis=0
                    )
                    boltzmann_policies.append(demo_cluster_boltzmann_policy_new)
                else:
                    # We added this trajectory to an existing cluster
                    # Accept the new cluster and reward
                    clusters[demo_idx] = demo_cluster_new

                # Run a compaction step, removing any empty clusters
                clusters, rewards, boltzmann_policies = cluster_compaction(
                    clusters, rewards, boltzmann_policies
                )
            else:
                # Reject the new cluster and reward - don't change anything
                continue

        print("Current clusters:", clusters)

        # print("Updating rewards...")
        for cluster_idx in range(len(set(clusters))):
            # Update reward function estimates based on current clusters
            reward = rewards[cluster_idx]
            policy = boltzmann_policies[cluster_idx]
            q_star = policy.q
            demos = [
                demonstrations[idx]
                for idx, c in enumerate(clusters)
                if c == cluster_idx
            ]

            # Take a single IRL step to update this reward function
            q_star_grad = q_grad_fpi(reward, xtr, phi)
            reward_log_grad = log_urpd_grad(
                xtr,
                reward,
                q_star_grad,
                demos,
                reward_prior_log_pdf_grad,
                reward_prior_confidence,
            )
            new_reward = reward_improvement_step(
                reward,
                reward_log_grad,
                langevin_scale_param,
            )

            # Apply reward constraints here - projection into a box constraint is truncation
            new_reward = np.clip(new_reward, *reward_bounds)

            # Compute acceptance probability of new reward
            _, new_q_star = vi(xtr, phi, Linear(new_reward))
            new_q_grad = q_grad_fpi(new_reward, xtr, phi)
            new_reward_log_urpd = log_urpd(
                xtr,
                new_reward,
                new_q_star,
                demos,
                reward_prior_log_pdf,
                reward_prior_confidence,
            )
            new_reward_log_grad = log_urpd_grad(
                xtr,
                new_reward,
                new_q_grad,
                demos,
                reward_prior_log_pdf_grad,
                reward_prior_confidence,
            )
            new_reward_log_g = log_g(
                new_reward, new_reward_log_grad, reward, langevin_scale_param
            )

            reward_log_urpd = log_urpd(
                xtr,
                reward,
                q_star,
                demos,
                reward_prior_log_pdf,
                reward_prior_confidence,
            )
            reward_log_g = log_g(
                reward, reward_log_grad, new_reward, langevin_scale_param
            )

            log_ratio = (
                new_reward_log_urpd
                + new_reward_log_g
                - (reward_log_urpd + reward_log_g)
            )
            log_acceptance_probs.append(log_ratio)
            accept_logprob = min(np.log(1.0), log_ratio)
            accept_prob = np.exp(accept_logprob)

            # print(f"R_{cluster_idx}: Log ratio is {log_ratio} ({np.exp(log_ratio)})")

            # Accept/reject new reward
            if np.random.rand() <= accept_prob:
                # Accept new reward
                # print(
                #     f"Accepting R_{cluster_idx} change from\n{reward} to\n{new_reward}"
                # )
                rewards[cluster_idx] = new_reward

                # Solve for policy under new cluster reward
                cluster_policy_new = BoltzmannExplorationPolicy(
                    new_q_star, reward_prior_confidence
                )
                boltzmann_policies[cluster_idx] = cluster_policy_new
            else:
                # Reject new reward
                # print(f"Rejecting change of R_{cluster_idx}")
                continue

        # Run a compaction step, removing any empty clusters
        clusters, rewards, boltzmann_policies = cluster_compaction(
            clusters, rewards, boltzmann_policies
        )

        print("Current rewards:")
        print(rewards)

        print(
            f"Mean acceptance ratio is {np.exp(np.mean(log_acceptance_probs))} - target is {TARGET_REWARD_ACCEPTANCE_PROB_RATIO}"
        )

    # Return learned reward ensemble
    return clusters, rewards


def main():
    """Main function"""

    from mdp_extras import vi, OptimalPolicy
    from multimodal_irl.envs.element_world import ElementWorldEnv, element_world_extras

    demos_per_mode = 10

    num_elements = 4
    env = ElementWorldEnv(num_elements=num_elements, wind=0.1)
    xtr, phi, rewards_gt = element_world_extras(env)
    demos = []
    for reward in rewards_gt:
        _, q_star = vi(xtr, phi, reward)
        pi_star = OptimalPolicy(q_star)
        cluster_demos = pi_star.get_rollouts(env, demos_per_mode)
        demos.extend(cluster_demos)

    print("GT Rewards:")
    print(np.array([r.theta for r in rewards_gt]))

    # Algorithm hyper-params
    concentration = 10.0
    max_initial_clusters = num_elements
    reward_mean = np.mean(rewards_gt[0].theta) * np.ones_like(rewards_gt[0].theta)
    reward_covariance = np.var(rewards_gt[0].theta) * np.ones_like(rewards_gt[0].theta)
    max_iterations = 10000
    reward_bounds = (-10.0, 0.0)

    # Initialise clusters
    initial_clusters = []
    for cluster_idx, cluster_size in enumerate(
        multinomial(
            len(demos),
            *dirichlet(
                np.ones(max_initial_clusters) * concentration / max_initial_clusters
            ).rvs(1),
        ).rvs(1)[0]
    ):
        for _ in range(cluster_size):
            initial_clusters.append(cluster_idx)
    initial_clusters = cluster_compaction(np.array(initial_clusters))
    num_initial_clusters = len(set(initial_clusters))

    # Define the reward prior and associated access functions
    reward_prior = multivariate_normal(mean=reward_mean, cov=reward_covariance)
    reward_prior_sample = lambda: reward_prior.rvs(1)
    reward_prior_log_pdf = lambda theta: reward_prior.logpdf(theta)
    reward_prior_log_pdf_grad = lambda theta: np.array(
        [
            (1.0 / norm(mu_d, sigma_d).pdf(theta_d))
            * (-1.0 / sigma_d / np.sqrt(2 * np.pi))
            * ((theta_d - mu_d) / sigma_d)
            * (np.exp(-0.5 * ((theta_d - mu_d) / sigma_d) ** 2))
            for reward_dim, (theta_d, mu_d, sigma_d) in enumerate(
                zip(theta, reward_mean, reward_covariance)
            )
        ]
    )  # Believe it or not, this computes the derivative of the multivariate normal log pdf (with diagonal covariance)

    # Initialise rewards
    initial_rewards = np.clip(reward_prior.rvs(num_initial_clusters), *reward_bounds)
    if len(set(initial_clusters)) == 1:
        initial_rewards = np.array([initial_rewards])

    # Set initial clusters and rewards to ground truth
    # print("Initialising with ground truth rewards and clusters")
    # initial_clusters = np.concatenate(
    #     [[i] * demos_per_mode for i in range(num_elements)]
    # )
    # initial_rewards = np.array([r.theta for r in rewards_gt])

    with np.errstate(divide="raise", over="raise", invalid="raise"):
        rewards_learned = ch_dpm_birl(
            xtr,
            phi,
            demos,
            initial_clusters,
            initial_rewards,
            max_iterations,
            reward_prior_sample,
            reward_prior_log_pdf,
            reward_prior_log_pdf_grad,
            dirichlet_prior_concentration=concentration,
            reward_bounds=reward_bounds,
        )

    print("Learned Rewards:")
    print(rewards_learned)

    print("Done")


if __name__ == "__main__":
    main()
