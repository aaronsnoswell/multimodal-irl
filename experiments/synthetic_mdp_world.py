"""ElementWorld experiment script

"""
import os
import tqdm
import pickle
import random
import argparse
import warnings

import numpy as np

from sacred import Experiment
from sacred.observers import MongoObserver

from pprint import pprint
from datetime import datetime
from concurrent import futures

from unimodal_irl import maxent_ml_path, maxlikelihood_ml_path

from experiments.utils import (
    replicate_config,
    get_num_workers,
    mongo_config,
    geometric_distribution,
)
from multimodal_irl.bv_em import (
    MaxEntEMSolver,
    MaxLikEMSolver,
    SigmaGIRLEMSolver,
    bv_em,
    MeanOnlyEMSolver,
)

from mdp_extras import (
    OptimalPolicy,
    BoltzmannExplorationPolicy,
    UniformRandomPolicy,
    padding_trick,
    vi,
    PaddedMDPWarning,
    trajectory_reward,
)

from multimodal_irl.envs.synthetic_world import (
    SyntheticWorldEnv,
    synthetic_world_extras,
)
from multimodal_irl.metrics import (
    normalized_information_distance,
    adjusted_normalized_information_distance,
    min_cost_flow_error_metric,
)
from unimodal_irl.metrics import ile_evd


def base_config():
    num_states = 10
    num_actions = 2
    num_feature_dimensions = 3
    num_behaviour_modes = 5
    num_demos = 100
    demo_skew = 0.0
    num_clusters = 3
    algorithm = "MaxEnt"
    initialisation = "Random"
    gamma = 0.98
    max_demonstration_length = 20
    num_init_restarts = 5000
    em_nll_tolerance = 0.0
    em_resp_tolerance = 1e-4
    max_iterations = 100
    boltzmann_scale = 5.0
    skip_ml_paths = True
    reward_initialisation = "MLE"
    optimization_method = ""
    replicate = 0


def get_avg_total_disc_reward(rollouts, phi, reward_function, discount_factor):
    rew = reward_function(phi.demo_average(rollouts, gamma=discount_factor))
    print("avg. total discounted reward, num demos:", (rew, len(rollouts)))
    return rew


def synthetic_mdp_world_v1(
    num_states,
    num_actions,
    num_feature_dimensions,
    num_behaviour_modes,
    num_demos,
    demo_skew,
    num_clusters,
    algorithm,
    initialisation,
    gamma,
    max_demonstration_length,
    num_init_restarts,
    em_nll_tolerance,
    em_resp_tolerance,
    max_iterations,
    boltzmann_scale,
    skip_ml_paths,
    reward_initialisation,
    optimization_method,
    seed,
    _log,
    _run,
):
    """ElementWorld Sacred Experiment"""

    # Construct EW
    _log.info(f"{seed}: Preparing environment...")
    env = SyntheticWorldEnv(
        num_states=num_states,
        num_actions=num_actions,
        num_behaviour_modes=num_behaviour_modes,
        num_feature_dimensions=num_feature_dimensions,
        discount_factor=gamma,
        seed=seed,
    )

    xtr, phi, gt_rewards = synthetic_world_extras(env)
    reward_parameter_range = (-1.0, 1.0)

    mode_proportions = geometric_distribution(demo_skew, num_behaviour_modes)
    demos_per_mode = np.floor(mode_proportions * num_demos)

    # Ensure every mode has at least 1 demo
    demos_per_mode = np.maximum(demos_per_mode, 1)

    # Ensure correct number of demos are present
    while np.sum(demos_per_mode) > num_demos:
        demos_per_mode[np.argmax(demos_per_mode)] -= 1
    while np.sum(demos_per_mode) < num_demos:
        demos_per_mode[np.argmin(demos_per_mode)] += 1

    # Convert to int
    demos_per_mode = demos_per_mode.astype(int)

    # Solve, get train dataset
    train_demos = []
    train_gt_resp = []
    q_stars = []

    num_opt = np.zeros(num_states)
    for ri, (reward, num_element_demos) in enumerate(zip(gt_rewards, demos_per_mode)):
        resp_row = np.zeros(num_behaviour_modes)
        resp_row[ri] = 1.0
        for _ in range(num_element_demos):
            train_gt_resp.append(resp_row)
        state_values, q_star = vi(xtr, phi, reward)
        q_stars.append(q_star)
        # pi_star = BoltzmannExplorationPolicy(q_star, scale=boltzmann_scale)
        pi_star = OptimalPolicy(q_star)
        rollout = pi_star.get_rollouts(
            env, num_element_demos, max_path_length=max_demonstration_length
        )
        train_demos.extend(rollout)
    # print("train demos", train_demos)
    train_gt_resp = np.array(train_gt_resp)
    train_gt_mixture_weights = np.sum(train_gt_resp, axis=0) / num_demos

    # Solve, get test dataset
    test_demos = []
    test_gt_resp = []
    i = 0
    for ri, (reward, num_element_demos) in enumerate(zip(gt_rewards, demos_per_mode)):
        resp_row = np.zeros(num_behaviour_modes)
        resp_row[ri] = 1.0
        for _ in range(num_element_demos):
            test_gt_resp.append(resp_row)
        # _, q_star = vi(xtr, phi, reward)
        # pi_star = OptimalPolicy(q_stars[i])
        pi_star = BoltzmannExplorationPolicy(q_stars[i], scale=boltzmann_scale)
        test_demos.extend(
            pi_star.get_rollouts(
                env, num_element_demos, max_path_length=max_demonstration_length
            )
        )

        i += 1

    test_gt_resp = np.array(test_gt_resp)
    test_gt_mixture_weights = np.sum(test_gt_resp, axis=0) / num_demos

    if reward_initialisation == "MLE":
        # We use the current IRL model for Maximum Likelihood initialisation of the
        # reward parameters
        if algorithm == "MaxEnt":
            solver = MaxEntEMSolver(
                method=optimization_method, min_path_length=max_demonstration_length
            )
            if env._needs_padding == True:
                xtr_p, train_demos_p = padding_trick(xtr, train_demos)
                _, test_demos_p = padding_trick(xtr, test_demos)
            else:
                (xtr_p, train_demos_p) = (xtr, train_demos)
                (_, test_demos_p) = (None, test_demos)
        elif algorithm == "MaxLik":
            solver = MaxLikEMSolver()
            xtr_p = xtr
            train_demos_p = train_demos
            test_demos_p = test_demos
        elif algorithm == "SigmaGIRL":
            solver = SigmaGIRLEMSolver()
            xtr_p = xtr
            train_demos_p = train_demos
            test_demos_p = test_demos
        else:
            raise ValueError
    elif reward_initialisation == "MeanOnly":
        # We use a 'mean only' solver to do the reward initialisation
        solver = MeanOnlyEMSolver()
        xtr_p = xtr
        train_demos_p = train_demos
        test_demos_p = test_demos
    else:
        raise ValueError()

    # Initialize Mixture
    t0 = datetime.now()
    if initialisation == "Random":
        # Initialize uniformly at random
        # print("Init random")
        init_mode_weights, init_rewards = solver.init_random(
            xtr_p, phi, train_demos_p, num_clusters, reward_parameter_range
        )

        # Below code randomly re-tries random initialisation until we get a non-degenerate initial reward ensemble
        # while True:
        #
        #     init_mode_weights, init_rewards = solver.init_random(
        #         xtr_p, phi, train_demos_p, num_clusters, reward_parameter_range
        #     )
        #
        #     # Sometimes, due to a bad random initialization, the initial learned rewards
        #     # will be identical. This causes degeneracy in the EM algorithm (the modes have collapsed)
        #     # To fix this, we check if this has occurred, and re-sample a new initial reward ensemble
        #
        #     # Check learned rewards are distinct
        #     all_rewards_are_same = True
        #     reward_vecs = [r.theta for r in init_rewards]
        #     for r1_idx in range(len(reward_vecs)):
        #         for r2_idx in range(r1_idx + 1, len(reward_vecs)):
        #             if np.array_equal(reward_vecs[r1_idx], reward_vecs[r2_idx]):
        #                 # This reward pair were identical!
        #                 pass
        #             else:
        #                 # This reward pair were not the same
        #                 all_rewards_are_same = False
        #
        #     if not all_rewards_are_same:
        #         break
        #
        #     warnings.warn(f"Random initialization lead to degeneracy (all mode rewards are == {reward_vecs[0]}) - re-trying random init")

        # print("Init random done")
    elif initialisation == "KMeans":
        # Initialize with K-Means (hard) clustering
        init_mode_weights, init_rewards = solver.init_kmeans(
            xtr_p,
            phi,
            train_demos_p,
            num_clusters,
            reward_parameter_range,
            num_init_restarts,
        )
    elif initialisation == "GMM":
        # Initialize with GMM (soft) clustering
        init_mode_weights, init_rewards = solver.init_gmm(
            xtr_p,
            phi,
            train_demos_p,
            num_clusters,
            reward_parameter_range,
            num_init_restarts,
        )
    elif initialisation == "Supervised":
        # We always have uniform clusters in supervised experiments
        assert num_clusters == num_behaviour_modes

        if isinstance(solver, MaxEntEMSolver):
            # Apply padding trick
            if env._needs_padding == True:
                xtr_p, train_demos_p = padding_trick(xtr, train_demos)
            else:
                (xtr_p, train_demos_p) = (xtr, train_demos)

            # Learn rewards with ground truth responsibility matrix
            learn_rewards = solver.mstep(
                xtr_p, phi, train_gt_resp, train_demos_p, reward_parameter_range
            )

            # Compute baseline NLL
            mixture_nll = solver.mixture_nll(
                xtr_p, phi, train_gt_mixture_weights, learn_rewards, train_demos_p
            )
        elif isinstance(solver, MaxLikEMSolver):

            # Learn rewards with ground truth responsibility matrix
            learn_rewards = solver.mstep(
                xtr, phi, train_gt_resp, train_demos, reward_parameter_range
            )

            # Compute baseline NLL
            mixture_nll = solver.mixture_nll(
                xtr, phi, train_gt_mixture_weights, learn_rewards, train_demos
            )

        else:
            raise ValueError()

        # No initial solution for supervised experiment
        init_resp = None
        init_mode_weights = None
        init_rewards = None
        init_eval = None
        init_eval_train = None

        # Skip BV training
        train_iterations = np.nan
        resp_history = [train_gt_resp]
        mode_weights_history = [train_gt_mixture_weights]
        rewards_history = [learn_rewards]
        nll_history = [mixture_nll]
        train_reason = "Supervised baseline mixture - no training needed"

    else:
        raise ValueError

    def post_em_iteration(
        solver, iteration, resp, mode_weights, rewards, nll, nll_delta, resp_delta
    ):
        _log.info(f"{seed}: Iteration {iteration} ended")
        _run.log_scalar("training.nll", nll)
        _run.log_scalar("training.nll_delta", nll_delta)
        _run.log_scalar("training.resp_delta", resp_delta)
        for mw_idx, mw in enumerate(mode_weights):
            _run.log_scalar(f"training.mw{mw_idx+1}", mw)
        for reward_idx, reward in enumerate(rewards):
            for theta_idx, theta_val in enumerate(reward.theta):
                _run.log_scalar(f"training.r{reward_idx+1}.t{theta_idx+1}", theta_val)

    _log.info(f"{seed}: Initialisation done - switching to MLE reward model for EM alg")
    if algorithm == "MaxEnt":
        solver = MaxEntEMSolver(post_it=post_em_iteration, method=optimization_method)
        if env._needs_padding:
            xtr_p, train_demos_p = padding_trick(xtr, train_demos)
            _, test_demos_p = padding_trick(xtr, test_demos)
        else:
            (xtr_p, train_demos_p) = (xtr, train_demos)
            (_, test_demos_p) = (xtr, test_demos)
    elif algorithm == "MaxLik":
        solver = MaxLikEMSolver(post_it=post_em_iteration)
        xtr_p = xtr
        train_demos_p = train_demos
        test_demos_p = test_demos
    elif algorithm == "SigmaGIRL":
        solver = SigmaGIRLEMSolver(post_it=post_em_iteration)
        xtr_p = xtr
        train_demos_p = train_demos
        test_demos_p = test_demos
    else:
        raise ValueError

    # Evaluate the initial mixture and run EM loop
    if initialisation != "Supervised":
        # Get initial responsibility matrix
        init_resp = solver.estep(
            xtr_p, phi, init_mode_weights, init_rewards, test_demos_p
        )
        init_resp_train = solver.estep(
            xtr_p, phi, init_mode_weights, init_rewards, train_demos_p
        )

        # Evaluate initial mixture
        _log.info(f"{seed}: Evaluating initial solution (test set)")
        init_eval = element_world_eval(
            xtr,
            phi,
            test_demos,
            test_gt_resp,
            test_gt_mixture_weights,
            gt_rewards,
            init_resp,
            init_mode_weights,
            init_rewards,
            solver,
            skip_ml_paths,
            None,
            env._needs_padding,
        )
        _log.info(f"{seed}: Evaluating initial solution (train set)")
        init_eval_train = element_world_eval(
            xtr,
            phi,
            train_demos,
            train_gt_resp,
            train_gt_mixture_weights,
            gt_rewards,
            init_resp_train,
            init_mode_weights,
            init_rewards,
            solver,
            skip_ml_paths,
            non_data_perf=init_eval,
            needs_padding=env._needs_padding,
        )

        # MI-IRL algorithm
        _log.info(f"{seed}: BV-EM Loop")
        (
            train_iterations,
            resp_history,
            mode_weights_history,
            rewards_history,
            nll_history,
            train_reason,
        ) = bv_em(
            solver,
            xtr_p,
            phi,
            train_demos_p,
            num_clusters,
            reward_parameter_range,
            mode_weights=init_mode_weights,
            rewards=init_rewards,
            nll_tolerance=em_nll_tolerance,
            resp_tolerance=em_resp_tolerance,
            max_iterations=max_iterations,
        )
        _log.info(f"{seed}: BV-EM Loop terminated, reason = {train_reason}")

    # Code to plot reward ensemble evolution
    # import time
    # import matplotlib.pyplot as plt
    #
    # fname = f"mixture_tracking-{str(time.time())}-{seed}.png"
    #
    # foo = np.array([[r[0].theta, r[1].theta] for r in rewards_history])
    #
    # plt.figure()
    # plt.plot(foo[:, 0, 0], "r.-", label=f"r0[0]", alpha=0.3)
    # plt.plot(foo[:, 0, 1], "r.--", label=f"r0[1]", alpha=0.3)
    # plt.plot(foo[:, 1, 0], "b.-", label=f"r1[0]", alpha=0.3)
    # plt.plot(foo[:, 1, 1], "b.--", label=f"r1[1]", alpha=0.3)
    # plt.legend()
    # plt.xlabel("EM Iteration")
    # plt.ylabel("Reward parameter value")
    # plt.savefig(fname)
    # plt.close()

    t1 = datetime.now()
    learn_resp = resp_history[-1]
    learn_mode_weights = mode_weights_history[-1]
    learn_rewards = rewards_history[-1]
    learn_nll = nll_history[-1]
    train_duration = (t1 - t0).total_seconds()

    # Derive Responsibility matrix for test paths
    learn_resp_test = solver.estep(
        xtr_p, phi, learn_mode_weights, learn_rewards, test_demos_p
    )

    # Evaluate final mixture
    _log.info(f"{seed}: Evaluating final mixture (test set)")
    learn_eval = element_world_eval(
        xtr,
        phi,
        test_demos,
        test_gt_resp,
        test_gt_mixture_weights,
        gt_rewards,
        learn_resp_test,
        learn_mode_weights,
        learn_rewards,
        solver,
        skip_ml_paths,
        None,
        env._needs_padding,
    )
    _log.info(f"{seed}: Evaluating final mixture (train set)")
    learn_eval_train = element_world_eval(
        xtr,
        phi,
        train_demos,
        train_gt_resp,
        train_gt_mixture_weights,
        gt_rewards,
        learn_resp,
        learn_mode_weights,
        learn_rewards,
        solver,
        skip_ml_paths,
        non_data_perf=learn_eval,
        needs_padding=env._needs_padding,
    )

    out_str = (
        "{}: Finished after {} iterations ({}) =============================\n"
        "NLL: {:.2f} -> {:.2f} (train), {:.2f} -> {:.2f} (test)\n"
        "ANID: {:.2f} -> {:.2f} (train), {:.2f} -> {:.2f} (test)\n"
        "EVD: {:.2f} -> {:.2f} (train), {:.2f} -> {:.2f} (test)\n"
        "Mode Weights: {} -> {}\n"
        "===================================================\n".format(
            seed,
            train_iterations,
            train_reason,
            np.nan if init_eval_train is None else init_eval_train["nll"],
            learn_eval_train["nll"],
            np.nan if init_eval is None else init_eval["nll"],
            learn_eval["nll"],
            np.nan if init_eval_train is None else init_eval_train["anid"],
            learn_eval_train["anid"],
            np.nan if init_eval is None else init_eval["anid"],
            learn_eval["anid"],
            np.nan if init_eval_train is None else init_eval_train["mcf_evd"],
            learn_eval_train["mcf_evd"],
            np.nan if init_eval is None else init_eval["mcf_evd"],
            learn_eval["mcf_evd"],
            init_mode_weights,
            learn_mode_weights,
        )
    )
    print(out_str, flush=True)

    # Dump experimental results to artifact
    _log.info(f"{seed}: Done...")
    result_fname = f"{seed}.result"
    with open(result_fname, "wb") as file:
        pickle.dump(
            {
                # Initial soln
                "init_resp": [] if init_resp is None else init_resp.tolist(),
                "init_mode_weights": []
                if init_mode_weights is None
                else init_mode_weights.tolist(),
                "init_rewards": []
                if init_rewards is None
                else [np.array(r.theta).tolist() for r in init_rewards],
                "init_eval": {} if init_eval is None else init_eval,
                "init_eval_train": {} if init_eval_train is None else init_eval_train,
                # Final soln
                "learn_resp": learn_resp.tolist(),
                "learn_mode_weights": learn_mode_weights.tolist(),
                "learn_rewards": [np.array(r.theta).tolist() for r in learn_rewards],
                "learn_eval": learn_eval,
                "learn_eval_train": learn_eval_train,
                # Training details
                "train_iterations": train_iterations,
                "train_duration": train_duration,
                "resp_history": np.array(resp_history).tolist(),
                "mode_weights_history": np.array(mode_weights_history).tolist(),
                "rewards_history": np.array(
                    [[r.theta for r in r1r2r3] for r1r2r3 in rewards_history]
                ).tolist(),
                "nll_history": np.array(nll_history).tolist(),
                "train_reason": train_reason,
            },
            file,
        )
    _run.add_artifact(result_fname)
    os.remove(result_fname)

    _log.info(f"{seed}: Done")

    return float(learn_nll)


def element_world_eval(
    xtr,
    phi,
    demos,
    gt_resp,
    gt_mixture_weights,
    gt_rewards,
    resp,
    mixture_weights,
    rewards,
    solver,
    skip_ml_paths,
    non_data_perf=None,
    needs_padding=False,
):
    """Evaluate a ElementWorld mixture model

    TODO ajs 13/Jan/2020 Optimize this function - it's sloooow!

    Args:
        TODO
        non_data_perf (dict): If passed, this dictionary will be used to look up
            existing performance stats for the evd/ile metric - saves re-computing it
            across test/training sets.

    Returns:
        TODO
    """
    gt_num_clusters = len(gt_mixture_weights)
    num_clusters = len(mixture_weights)

    ### NON-DATA BASED EVALUATIONS =====================================================

    if (
        (non_data_perf is not None)
        and ("mcf_evd" in non_data_perf)
        and ("mcf_ile" in non_data_perf)
        and ("mcf_evd_flowdict" in non_data_perf)
        and ("mcf_ile_flowdict" in non_data_perf)
    ):
        mcf_ile = non_data_perf["mcf_ile"]
        mcf_evd = non_data_perf["mcf_evd"]
        mcf_evd_flowdict = non_data_perf["mcf_evd_flowdict"]
        mcf_ile_flowdict = non_data_perf["mcf_ile_flowdict"]
    else:
        # Compute ILE, EVD matrices
        ile_mat = np.zeros((num_clusters, gt_num_clusters))
        evd_mat = np.zeros((num_clusters, gt_num_clusters))
        for gt_mode_idx in range(gt_num_clusters):
            gt_state_value_vector = None
            for learned_mode_idx in range(num_clusters):
                ile, evd, gt_state_value_vector = ile_evd(
                    xtr,
                    phi,
                    gt_rewards[gt_mode_idx],
                    rewards[learned_mode_idx],
                    ret_gt_value=True,
                    gt_policy_value=gt_state_value_vector,
                    vi_kwargs=dict(eps=1e-6),
                )
                ile_mat[learned_mode_idx, gt_mode_idx] = ile
                evd_mat[learned_mode_idx, gt_mode_idx] = evd

        # Measure reward performance
        mcf_ile, mcf_ile_flowdict = min_cost_flow_error_metric(
            mixture_weights, gt_mixture_weights, ile_mat
        )
        mcf_evd, mcf_evd_flowdict = min_cost_flow_error_metric(
            mixture_weights, gt_mixture_weights, evd_mat
        )

    ### DATA BASED EVALUATIONS =========================================================

    # Measure NLL
    if isinstance(solver, MaxEntEMSolver):
        if needs_padding == True:
            xtr_p, demos_p = padding_trick(xtr, demos)
        else:
            (xtr_p, demos_p) = (xtr, demos)
        nll = solver.mixture_nll(xtr_p, phi, mixture_weights, rewards, demos_p)
    elif isinstance(solver, MaxLikEMSolver):
        nll = solver.mixture_nll(xtr, phi, mixture_weights, rewards, demos)
    elif isinstance(solver, SigmaGIRLEMSolver):
        raise NotImplementedError
        # TODO
        nll = solver.mixture_nll(xtr, phi, mixture_weights, rewards, demos)
    else:
        raise ValueError

    # Measure clustering performance
    nid = normalized_information_distance(gt_resp, resp)
    anid = adjusted_normalized_information_distance(gt_resp, resp)

    ml_paths = []
    pdms = []
    fds = []
    if skip_ml_paths:
        pass
    else:
        """if isinstance(solver, MaxEntEMSolver):
            # Get ML paths from MaxEnt mixture
            ml_paths = element_world_mixture_ml_path(
                xtr, phi, demos, maxent_ml_path, mixture_weights, rewards
            )
        elif isinstance(solver, MaxLikEMSolver):
            # Get ML paths from MaxLik mixture
            ml_paths = element_world_mixture_ml_path(
                xtr, phi, demos, maxlikelihood_ml_path, mixture_weights, rewards
            )
        elif isinstance(solver, SigmaGIRLEMSolver):
            raise NotImplementedError
            # TODO
            # Get ML paths from SigmaGIRL mixture
        else:
            raise ValueError

        # Measure % Distance Missed of ML paths
        pdms = np.array(
            [
                percent_distance_missed_metric(ml_path, gt_path)
                for (gt_path, ml_path) in zip(demos, ml_paths)
            ]
        )

        # Measure feature distance of ML paths
        fds = np.array(
            [
                phi.feature_distance(gt_path, ml_path)
                for (gt_path, ml_path) in zip(demos, ml_paths)
            ]
        )"""

    return dict(
        nll=nll,
        nid=nid,
        anid=anid,
        mcf_ile=mcf_ile,
        mcf_ile_flowdict=mcf_ile_flowdict,
        mcf_evd=mcf_evd,
        mcf_evd_flowdict=mcf_evd_flowdict,
        pdms=pdms,
        fds=fds,
    )


def run(config_updates, mongodb_url="localhost:27017", mongodb_name="MY_DB"):
    """Run a single experiment with the given configuration

    Args:
        config_updates (dict): Configuration updates
        mongodb_url (str): MongoDB URL, or None if no Mongo observer should be used for
            this run
    """

    # Dynamically bind experiment config and main function
    ex = Experiment()
    ex.config(base_config)
    ex.main(synthetic_mdp_world_v1)

    # Attach MongoDB observer if necessary
    # if mongodb_url is not None and not ex.observers:
    #     ex.observers.append(MongoObserver(url=mongodb_url, db_name=mongodb_name))

    # Suppress warnings about padded MPDs
    with warnings.catch_warnings():
        warnings.filterwarnings(action="ignore", category=PaddedMDPWarning)

        # Run the experiment
        run = ex.run(
            config_updates=config_updates
        )  # , options={"--loglevel": "ERROR"})

    # Return the result
    return run.result


def main():
    """Main function"""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--num_states",
        required=True,
        type=int,
        help="Number of states",
    )

    parser.add_argument(
        "--num_actions",
        required=True,
        type=int,
        help="Number of actions",
    )

    parser.add_argument(
        "--num_feature_dimensions",
        required=True,
        type=int,
        help="Number of actions",
    )

    parser.add_argument(
        "--num_behaviour_modes",
        required=False,
        default=5,
        type=int,
        help="Number of environment modes to create",
    )

    parser.add_argument(
        "--num_demos",
        required=False,
        default=100,
        type=int,
        help="Total number of demonstrations to give MI-IRL algorithm",
    )

    parser.add_argument(
        "--demo_skew",
        required=False,
        default=0.0,
        type=float,
        help="Geometric distribution skew for allocating demonstrations to ground truth modes - 0.0 results in uniform distribution, 1.0 results in a very skewed distribution",
    )

    parser.add_argument(
        "--num_clusters",
        required=False,
        default=3,
        type=int,
        help="Number of clusters to learn",
    )

    parser.add_argument(
        "--algorithm",
        required=False,
        default="MaxEnt",
        type=str,
        choices=("MaxEnt", "MaxLik", "SigmaGIRL"),
        help="IRL model + algorithm to use in EM inner loop",
    )

    parser.add_argument(
        "--initialisation",
        required=False,
        type=str,
        default="Random",
        choices=("Random", "KMeans", "GMM", "Supervised"),
        help="Cluster initialisation method to use",
    )

    parser.add_argument(
        "--reward_initialisation",
        required=False,
        default="MLE",
        type=str,
        choices=("MLE", "MeanOnly"),
        help="Reward initialisation method to use - defaults to MLE",
    )

    parser.add_argument(
        "--num_replicates",
        required=False,
        type=int,
        default=100,
        help="Number of replicates to perform",
    )

    parser.add_argument(
        "--num_workers",
        required=False,
        default=None,
        type=int,
        help="Number of workers to use - if not provided, will be inferred from system and workload",
    )

    parser.add_argument(
        "--em_nll_tolerance",
        required=False,
        default=0.0,
        type=float,
        help="EM convergence tolerance for mixture NLL change",
    )

    parser.add_argument(
        "--em_resp_tolerance",
        required=False,
        default=1e-4,
        type=float,
        help="EM convergence tolerance for the responsibility matrix entries. N.b. this should be scaled with (number of paths) x (number of learned modes). In our experiments we used (1.25e-6)x(num paths)x(num learned modes)",
    )

    parser.add_argument(
        "--max_iterations",
        required=False,
        default=100,
        type=int,
        help="Maximum number of EM iterations to perform",
    )

    parser.add_argument(
        "--eval_ml_paths",
        action="store_true",
        help="Perform ML path evaluations (speeds up experiment substantially)",
    )

    parser.add_argument(
        "--mongodb_name",
        required=False,
        default="MY_DB",
        type=str,
        help="The name of the Mongo database to store the results in",
    )

    parser.add_argument(
        "--seed", required=False, default=None, type=int, help="The seed"
    )

    parser.add_argument(
        "--optimization_method",
        required=False,
        default="L-BFGS-B",
        type=str,
        help="The optimization method",
    )

    parser.add_argument(
        "--max_demonstration_length",
        required=True,
        type=int,
        help="The maximum length of the demonstration trajectories",
    )

    args = parser.parse_args()
    print("META: Arguments:", args, flush=True)

    config_updates = {
        "num_states": args.num_states,
        "num_actions": args.num_actions,
        "num_feature_dimensions": args.num_feature_dimensions,
        "num_behaviour_modes": args.num_behaviour_modes,
        "num_demos": args.num_demos,
        "demo_skew": args.demo_skew,
        "num_clusters": args.num_clusters,
        "algorithm": args.algorithm,
        "initialisation": args.initialisation,
        "reward_initialisation": args.reward_initialisation,
        "em_nll_tolerance": args.em_nll_tolerance,
        "em_resp_tolerance": args.em_resp_tolerance,
        "max_iterations": args.max_iterations,
        "optimization_method": args.optimization_method,
        "skip_ml_paths": not args.eval_ml_paths,
        "max_demonstration_length": args.max_demonstration_length,
    }

    if args.seed != None:
        config_updates["seed"] = args.seed

    print("META: Configuration: ")
    pprint(config_updates)

    configs = replicate_config(config_updates, args.num_replicates)
    num_workers = get_num_workers(len(configs), args.num_workers)
    print(
        f"META: Distributing {args.num_replicates} replicate(s) over {num_workers} workers"
    )

    mongodb_name = args.mongodb_name

    mongodb_url = None
    # Read MongoDB URL from config file, if it exists
    mongodb_url = mongo_config()
    print(f"META: MongoDB Server URL: {mongodb_url}")
    print(f"META: MongoDB Name: ", mongodb_name)

    # Parallel loop
    with tqdm.tqdm(total=len(configs)) as pbar:
        with futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            tasks = {executor.submit(run, config, mongodb_url) for config in configs}
            for future in futures.as_completed(tasks):
                # arg = tasks[future]; result = future.result()
                pbar.update(1)

    # Debugging loop
    # for config in tqdm.tqdm(configs):
    #     run(config, mongodb_url, mongodb_name)


if __name__ == "__main__":
    main()
