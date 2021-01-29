"""Implementation of Dirichlet Process Mixture Bayesian IRL by Choi and Kim, 2012"""

import random
import numpy as np

from scipy.stats import dirichlet, multivariate_normal, multinomial, norm

from mdp_extras import q_vi, BoltzmannExplorationPolicy, Linear, q_grad_fpi


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

    new_args = list(args)
    if max(clusters) != num_actual_clusters - 1:

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


def reward_log_posterior_grad(
    xtr,
    reward,
    q_grad,
    demos,
    reward_mean,
    reward_covariance_diag,
    reward_prior_confidence,
):
    """Compute gradient of MH update target distribution (un-normalized reward posterior)"""

    with np.errstate(divide="raise"):
        log_grad = np.zeros(len(reward))

        # Add contribution from demos in this cluster
        for cluster_demo in demos:
            # XXX ajs Skipping end state means we don't catch reward contribution from goal states!!!
            for t, (st, at) in enumerate(cluster_demo[:-1]):
                aprime_norm_factor = [
                    reward_prior_confidence * q_grad[st, aprime, :]
                    for aprime in xtr.actions
                ]
                aprime_norm_factor = np.sum(aprime_norm_factor, axis=0)
                contrib = (
                    reward_prior_confidence * q_grad[st, at, :] - aprime_norm_factor
                )
                log_grad += contrib

        # Add contribution from reward prior
        for reward_dim in range(len(reward)):
            theta_d = reward[reward_dim]
            mu_d = reward_mean[reward_dim]
            sigma_d = reward_covariance_diag[reward_dim]

            g_of_x = norm(mu_d, sigma_d).pdf(theta_d)
            f_prime = 1.0 / g_of_x

            a = -1.0 / sigma_d / np.sqrt(2 * np.pi)
            b = (theta_d - mu_d) / sigma_d
            c = np.exp(-0.5 * ((theta_d - mu_d) / sigma_d) ** 2)
            g_prime = a * b * c

            contrib = f_prime * g_prime

            log_grad[reward_dim] += contrib
    return log_grad


def reward_log_posterior(
    xtr,
    reward,
    q_star,
    demos,
    reward_mean,
    reward_covariance_diag,
    reward_prior_confidence,
):
    """Evaluate un-normalized reward log posterior"""
    rlp_nonorm = 0.0

    # Add contribution from demos in this cluster
    for cluster_demo in demos:
        # XXX ajs Skipping end state means we don't catch reward contribution from goal states!!!
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
    for reward_dim in range(len(reward)):
        theta_d = reward[reward_dim]
        mu_d = reward_mean[reward_dim]
        sigma_d = reward_covariance_diag[reward_dim]
        rlp_nonorm += np.log(norm(mu_d, sigma_d).pdf(theta_d))

    return rlp_nonorm


def logg(r1, r1_log_grad, r2, langevin_scale_param):
    """The log of the g(x, y) fn from the paper

    Choi and Kim get this equation wrong in the paper - they mistakenly subtract r2 from r1, but it should be
    the other way around, according to the original Langevin paper by Roberts and Rosenthal
    """
    a = np.log(1.0 / (2 * np.pi * langevin_scale_param ** 2) ** (len(r1) / 2))
    b = (
        -1.0
        / (2 * langevin_scale_param ** 2)
        * (
            np.linalg.norm(r1 - r2 - 0.5 * (langevin_scale_param ** 2) * r1_log_grad)
            ** 2
        )
    )
    return a + b


def logg_roberts_rosenthal(r1, r1_log_grad, r2, langevin_scale_param):
    """The log of the g(x, y) fn from the original Roberts and Rosenthal paper"""
    a = np.log(1.0 / (2 * np.pi * langevin_scale_param ** 2) ** (len(r1) / 2))
    b = (
        -1.0
        / (2 * langevin_scale_param ** 2)
        * (
            np.linalg.norm(r2 - r1 - 0.5 * (langevin_scale_param ** 2) * r1_log_grad)
            ** 2
        )
    )
    return a + b


TARGET_REWARD_ACCEPTANCE_PROB_RATIO = 0.574


def ch_dpm_birl(
    xtr,
    phi,
    demonstrations,
    max_clusters,
    reward_dimension,
    max_iterations,
    cluster_prior_concentration=1.0,
    reward_prior_mean=0.0,
    reward_prior_covariance=1.0,
    reward_prior_confidence=1.0,
    langevin_scale_param=0.001,
    reward_bounds=(None, None),
):
    """A Metropolis Hastings algorithm for DPM-BIRL
    
    Args:
        demonstrations (list): List of (s, a) demonstration trajectories
        max_clusters (int): Maximum number of clusters to learn
        reward_dimension (int): Number of of reward function parameters
        max_iterations (int): Maximum number of MH iterations
        
        cluster_prior_concentration (float): Dirichlet distribution concentration
            parameter - set to 1.0 for uninformed prior
        reward_prior_mean (float): Reward prior mean
        reward_prior_covariance (float): Reward prior covariance
        reward_prior_confidence (float): Boltzmann confidence parameter for MaxLikelihood IRL
            model
        langevin_scale_param (float): Positive scale for the Langevin dynamics step size - should be
            tuned down and/or up until the average acceptance probability for the MH process is ~0.574.
            Acceptance probabilities are inversely proportional to this parameter's size.
        reward_bounds (tuple): Tuple of low, high reward parameter bounds. Set the respective entry to None to
            remove that reward bound.
    
    Returns:
        TODO
    """

    num_demos = len(demonstrations)

    # TODO ajs 28/Jan/21 Initialize clusters and rewards arbitrarily (possibly from an argument)
    # For MCMC algorithms, convergence is guarunteed, regardless of initialisation

    # Initialize clusters from prior
    cluster_prior = dirichlet(
        np.ones(max_clusters) * cluster_prior_concentration / max_clusters
    )
    _cluster_sizes = multinomial(num_demos, *cluster_prior.rvs(1)).rvs(1)[0]
    clusters = []
    for cluster_idx, cluster_size in enumerate(_cluster_sizes):
        for _ in range(cluster_size):
            clusters.append(cluster_idx)
    clusters = np.array(clusters)

    # Drop any empty clusters for efficiency
    clusters = cluster_compaction(clusters)

    print("Initial clusters:")
    print(clusters)

    # Initialise rewards from prior
    reward_mean = np.ones(reward_dimension)
    reward_prior_mean = np.array(reward_prior_mean)
    if len(reward_prior_mean.shape) == 0:
        # Interpret as constant vector value
        reward_mean *= reward_prior_mean
    elif len(reward_prior_covariance.shape) == 1:
        # Interpret as actual mean vector
        assert len(reward_prior_mean) == reward_dimension
        reward_mean = reward_prior_mean
    else:
        raise ValueError

    reward_covariance = np.eye(reward_dimension)
    reward_prior_covariance = np.array(reward_prior_covariance)
    if len(reward_prior_covariance.shape) == 0:
        # Interpret covariance parameter as a diagonal cov matrix scale factor
        reward_covariance *= reward_prior_covariance
    elif len(reward_prior_covariance.shape) == 1:
        # Interpret covariance parameter as a covariance matrix diagonal
        assert len(reward_prior_covariance) == reward_dimension
        reward_covariance = np.diag(reward_prior_covariance)
    elif len(reward_prior_covariance.shape) == 2:
        # Interpret covariance parameter as full covariance matrix
        assert reward_prior_covariance.shape == (reward_dimension, reward_dimension)
        raise NotImplementedError(
            "Langevin gradient calculations not supported for covariance matrices"
        )
        reward_covariance = reward_prior_covariance
    else:
        raise ValueError
    reward_prior = multivariate_normal(mean=reward_mean, cov=reward_covariance)
    reward_covariance_diag = np.diag(reward_covariance)
    rewards = reward_prior.rvs(len(set(clusters)))
    if len(set(clusters)) == 1:
        rewards = np.array([rewards])

    # Apply reward constraints here - projection into a box constraint is truncation
    rewards = np.clip(rewards, *reward_bounds)
    print("Current Rewards:")
    print(rewards)

    # Solve for Boltzmann Policy for each reward parameter
    boltzmann_policies = [
        BoltzmannExplorationPolicy(q_vi(xtr, phi, Linear(r)), reward_prior_confidence)
        for r in rewards
    ]

    # Loop until max number of iterations
    log_acceptance_probs = []
    for t in range(max_iterations):

        print(f"Iteration {t}")

        # Loop over each demonstration
        # We assume the list of clusters is always compact
        print("Updating demonstration memberships...")
        for demo_idx, demo in enumerate(demonstrations):
            # Update cluster assignment for this demo
            demo_cluster = clusters[demo_idx]
            demo_cluster_boltzmann_policy = boltzmann_policies[demo_cluster]

            # Sample a new cluster for this trajectory from Eq 5
            # Count how many trajectories are in each cluster, excluding the current trajectory
            cluster_counts = [
                np.sum(clusters == cluster_idx) for cluster_idx in set(clusters)
            ]
            cluster_counts[demo_cluster] -= 1

            # Convert cluster counts to cluster probabilities, including the possibility of a new cluster
            cluster_probs = cluster_counts.copy()
            cluster_probs.append(cluster_prior_concentration)
            cluster_probs = np.array(cluster_probs)
            cluster_probs /= np.sum(cluster_probs)

            # Sample new cluster assignment
            # TODO ajs sample new cluster assignment from the full conditional posterior - it's
            # not that much more expensive, and will converge faster
            demo_cluster_new = np.random.choice(
                list(range(len(cluster_probs))), p=cluster_probs
            )

            if demo_cluster_new == demo_cluster:
                # We didn't move the demonstration - nothing doing
                continue
            elif demo_cluster_new not in clusters:
                # We selected a new/empty cluster, sample a new reward function from the prior
                demo_cluster_new_reward = reward_prior.rvs(1)

                # Solve for new reward's boltzmann policy
                demo_cluster_boltzmann_policy_new = BoltzmannExplorationPolicy(
                    q_vi(xtr, phi, Linear(demo_cluster_new_reward)),
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
                ) + demo_cluster_boltzmann_policy_new.path_log_likelihood(demo)

            if cluster_probs[demo_cluster] == 0.0:
                logb = -np.inf
            else:
                logb = np.log(
                    cluster_probs[demo_cluster]
                ) + demo_cluster_boltzmann_policy.path_log_likelihood(demo)

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

        print("Clusters:", clusters)

        print("Updating rewards...")
        for cluster_idx in range(len(set(clusters))):
            # Update reward function estimates based on current clusters
            cluster_reward = rewards[cluster_idx]
            cluster_policy = boltzmann_policies[cluster_idx]
            cluster_q_star = cluster_policy.q
            cluster_demos = [
                demonstrations[idx]
                for idx, c in enumerate(clusters)
                if c == cluster_idx
            ]

            # Sample epsilon
            eps = norm(0, 1).rvs(len(cluster_reward))

            # Estimate Q gradient (indexed by s, a, phi dimension)
            cluster_q_grad = q_grad_fpi(cluster_reward, xtr, phi)

            log_grad = reward_log_posterior_grad(
                xtr,
                cluster_reward,
                cluster_q_grad,
                cluster_demos,
                reward_mean,
                reward_covariance_diag,
                reward_prior_confidence,
            )

            # Compute r_k^*
            cluster_reward_new = (
                cluster_reward
                + langevin_scale_param ** 2 / 2.0 * log_grad
                + langevin_scale_param * eps
            )

            # Apply reward constraints here - projection into a box constraint is truncation
            cluster_reward_new = np.clip(cluster_reward_new, *reward_bounds)

            cluster_q_grad_new = q_grad_fpi(cluster_reward_new, xtr, phi)

            # Compute gradient of new reward
            log_grad_new = reward_log_posterior_grad(
                xtr,
                cluster_reward_new,
                cluster_q_grad_new,
                cluster_demos,
                reward_mean,
                reward_covariance_diag,
                reward_prior_confidence,
            )

            # Compute r_k^* acceptance probability in log space
            cluster_q_star_new = q_vi(xtr, phi, Linear(cluster_reward_new))
            logp_reward_new = reward_log_posterior(
                xtr,
                cluster_reward_new,
                cluster_q_star_new,
                cluster_demos,
                reward_mean,
                reward_covariance_diag,
                reward_prior_confidence,
            )
            # XXX ajs 28/jan/21 Mistake in Choi + Kim paper equation g(x, y) here
            # logg_reward_new = logg(
            #     cluster_reward_new, log_grad_new, cluster_reward, langevin_scale_param
            # )
            logg_reward_new = logg_roberts_rosenthal(
                cluster_reward_new, log_grad_new, cluster_reward, langevin_scale_param
            )

            logp_reward = reward_log_posterior(
                xtr,
                cluster_reward,
                cluster_q_star,
                cluster_demos,
                reward_mean,
                reward_covariance_diag,
                reward_prior_confidence,
            )
            # logg_reward = logg(
            #     cluster_reward, log_grad, cluster_reward_new, langevin_scale_param
            # )
            logg_reward = logg_roberts_rosenthal(
                cluster_reward, log_grad, cluster_reward_new, langevin_scale_param
            )

            log_ratio = logp_reward_new + logg_reward_new - (logp_reward + logg_reward)
            log_acceptance_probs.append(log_ratio)
            print(f"R_{cluster_idx}: Log ratio is {log_ratio} ({np.exp(log_ratio)})")
            accept_logprob = min(np.log(1.0), log_ratio)
            accept_prob = np.exp(accept_logprob)

            # Accept/reject r_k^*
            if np.random.rand() <= accept_prob:
                # Accept new reward
                print(
                    f"Accepting R_{cluster_idx} change from\n{cluster_reward} to\n{cluster_reward_new}"
                )
                rewards[cluster_idx] = cluster_reward_new

                # Solve for policy under new cluster reward
                cluster_policy_new = BoltzmannExplorationPolicy(
                    cluster_q_star_new, reward_prior_confidence
                )
                boltzmann_policies[cluster_idx] = cluster_policy_new
            else:
                # Reject new reward
                # XXX ajs we never reject any changes
                print(f"Rejecting change of R_{cluster_idx}")
                continue

        # Run a compaction step, removing any empty clusters
        clusters, rewards, boltzmann_policies = cluster_compaction(
            clusters, rewards, boltzmann_policies
        )

        print("Current Rewards:")
        print(rewards)

        print(
            f"Mean log acceptance ratio is {np.mean(log_acceptance_probs)} or {np.exp(np.mean(log_acceptance_probs))} - target is {TARGET_REWARD_ACCEPTANCE_PROB_RATIO}"
        )

    # Return learned reward ensemble
    return rewards


def main():
    """Main function"""

    from mdp_extras import q_vi, OptimalPolicy
    from multimodal_irl.envs.element_world import ElementWorldEnv, element_world_extras

    rollouts_per_mode = 10

    num_elements = 2
    env = ElementWorldEnv(num_elements=num_elements, wind=0.0)
    xtr, phi, rewards_gt = element_world_extras(env)
    demos = []
    for reward in rewards_gt:
        q_star = q_vi(xtr, phi, reward)
        pi_star = OptimalPolicy(q_star)
        demos.extend(pi_star.get_rollouts(env, rollouts_per_mode))

    print("GT Rewards:")
    print(np.array([r.theta for r in rewards_gt]))

    k = num_elements * 2
    max_iterations = 100

    with np.errstate(divide="raise", over="raise", invalid="raise"):
        rewards_learned = ch_dpm_birl(
            xtr,
            phi,
            demos,
            k,
            len(phi),
            max_iterations,
            reward_prior_mean=np.mean(rewards_gt[0].theta),
            reward_prior_covariance=np.var(rewards_gt[0].theta),
            reward_prior_confidence=1.0,
            cluster_prior_concentration=2.0,
            reward_bounds=(-10.0, 0.0),
        )

    print("Learned Rewards:")
    print(rewards_learned)

    print("Done")


if __name__ == "__main__":
    main()
