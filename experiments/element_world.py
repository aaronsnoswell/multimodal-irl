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
from multimodal_irl.bv_em import MaxEntEMSolver, MaxLikEMSolver, bv_em, MeanOnlyEMSolver

from mdp_extras import (
    OptimalPolicy,
    BoltzmannExplorationPolicy,
    padding_trick,
    q_vi,
    PaddedMDPWarning,
)

from multimodal_irl.envs.element_world import (
    ElementWorldEnv,
    element_world_extras,
    percent_distance_missed_metric,
    element_world_mixture_ml_path,
)
from multimodal_irl.metrics import (
    normalized_information_distance,
    adjusted_normalized_information_distance,
    min_cost_flow_error_metric,
)
from unimodal_irl.metrics import ile_evd


def base_config():
    num_elements = 3
    num_demos = 100
    demo_skew = 0.0
    num_clusters = 3
    wind = 0.1
    algorithm = "MaxEnt"
    initialisation = "Random"
    width = 6
    gamma = 0.99
    max_demonstration_length = 50
    reward_range = (-10.0, 0.0)
    num_init_restarts = 5000
    em_nll_tolerance = 1e-5
    max_iterations = 100
    boltzmann_scale = 5.0
    skip_ml_paths = False
    replicate = 0


def element_world_v4(
    num_elements,
    num_demos,
    demo_skew,
    num_clusters,
    wind,
    algorithm,
    initialisation,
    width,
    gamma,
    max_demonstration_length,
    reward_range,
    num_init_restarts,
    em_nll_tolerance,
    max_iterations,
    boltzmann_scale,
    skip_ml_paths,
    _log,
    _seed,
    _run,
):
    """ElementWorld Sacred Experiment"""

    # Construct EW
    _log.info(f"{_seed}: Preparing environment...")
    env = ElementWorldEnv(
        width=width, num_elements=num_elements, wind=wind, gamma=gamma
    )
    xtr, phi, gt_rewards = element_world_extras(env)

    _log.info(f"{_seed}: Generating rollouts...")
    mode_proportions = geometric_distribution(demo_skew, num_elements)
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
    for ri, (reward, num_element_demos) in enumerate(zip(gt_rewards, demos_per_mode)):
        resp_row = np.zeros(num_elements)
        resp_row[ri] = 1.0
        for _ in range(num_element_demos):
            train_gt_resp.append(resp_row)
        q_star = q_vi(xtr, phi, reward)
        # pi_star = OptimalPolicy(q_star)
        pi_star = BoltzmannExplorationPolicy(q_star, scale=boltzmann_scale)
        train_demos.extend(
            pi_star.get_rollouts(
                env, num_element_demos, max_path_length=max_demonstration_length
            )
        )
    train_gt_resp = np.array(train_gt_resp)
    train_gt_mixture_weights = np.sum(train_gt_resp, axis=0) / num_demos

    # Solve, get test dataset
    test_demos = []
    test_gt_resp = []
    for ri, (reward, num_element_demos) in enumerate(zip(gt_rewards, demos_per_mode)):
        resp_row = np.zeros(num_elements)
        resp_row[ri] = 1.0
        for _ in range(num_element_demos):
            test_gt_resp.append(resp_row)
        q_star = q_vi(xtr, phi, reward)
        # pi_star = OptimalPolicy(q_star)
        pi_star = BoltzmannExplorationPolicy(q_star, scale=boltzmann_scale)
        test_demos.extend(
            pi_star.get_rollouts(
                env, num_element_demos, max_path_length=max_demonstration_length
            )
        )
    test_gt_resp = np.array(test_gt_resp)
    test_gt_mixture_weights = np.sum(test_gt_resp, axis=0) / num_demos

    def post_em_iteration(solver, iteration, resp, mode_weights, rewards, nll):
        _log.info(f"{_seed}: Iteration {iteration} ended")
        _run.log_scalar("training.nll", nll)
        for mw_idx, mw in enumerate(mode_weights):
            _run.log_scalar(f"training.mw{mw_idx+1}", mw)
        for reward_idx, reward in enumerate(rewards):
            for theta_idx, theta_val in enumerate(reward.theta):
                _run.log_scalar(f"training.r{reward_idx+1}.t{theta_idx+1}", theta_val)

        # TODO ajs evaluate model here?

    # Initialize solver
    if algorithm == "MaxEnt":
        solver = MaxEntEMSolver(post_it=post_em_iteration)
        xtr_p, train_demos_p = padding_trick(xtr, train_demos)
        _, test_demos_p = padding_trick(xtr, test_demos)
    elif algorithm == "MaxLik":
        solver = MaxLikEMSolver(post_it=post_em_iteration)
        xtr_p = xtr
        train_demos_p = train_demos
        test_demos_p = test_demos
    elif algorithm == "MeanOnly":
        solver = MeanOnlyEMSolver(post_it=post_em_iteration)
        xtr_p = xtr
        train_demos_p = train_demos
        test_demos_p = test_demos
    else:
        raise ValueError

    # Initialize Mixture
    t0 = datetime.now()
    if initialisation == "Random":
        # Initialize uniformly at random
        init_mode_weights, init_rewards = solver.init_random(
            phi, num_clusters, reward_range
        )
    elif initialisation == "KMeans":
        # Initialize with K-Means (hard) clustering
        init_mode_weights, init_rewards = solver.init_kmeans(
            xtr_p, phi, train_demos_p, num_clusters, reward_range, num_init_restarts
        )
    elif initialisation == "GMM":
        # Initialize with GMM (soft) clustering
        init_mode_weights, init_rewards = solver.init_gmm(
            xtr_p, phi, train_demos_p, num_clusters, reward_range, num_init_restarts
        )
    elif initialisation == "Supervised":
        # We always have uniform clusters in supervised experiments
        assert num_clusters == num_elements

        # Use ground truth responsibility matrix and cluster weights
        xtr_p, train_demos_p = padding_trick(xtr, train_demos)

        # Learn rewards with ground truth responsibility matrix
        learn_rewards = solver.mstep(
            xtr_p, phi, train_gt_resp, train_demos_p, reward_range
        )

        # Compute baseline NLL
        mixture_nll = solver.mixture_nll(
            xtr_p, phi, train_gt_mixture_weights, learn_rewards, train_demos_p
        )

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

    if initialisation != "Supervised":
        # Get initial responsibility matrix
        init_resp = solver.estep(
            xtr_p, phi, init_mode_weights, init_rewards, test_demos_p
        )
        init_resp_train = solver.estep(
            xtr_p, phi, init_mode_weights, init_rewards, train_demos_p
        )

        # Evaluate initial mixture
        _log.info(f"{_seed}: Evaluating initial solution (test set)")
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
            _log,
            _seed,
        )
        _log.info(f"{_seed}: Evaluating initial solution (train set)")
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
            _log,
            _seed,
        )

        # MI-IRL algorithm
        _log.info(f"{_seed}: BV-EM Loop")
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
            reward_range,
            mode_weights=init_mode_weights,
            rewards=init_rewards,
            tolerance=em_nll_tolerance,
            max_iterations=max_iterations,
        )
        _log.info(f"{_seed}: BV-EM Loop terminated, reason = {train_reason}")

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
    _log.info(f"{_seed}: Evaluating final mixture (test set)")
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
        _log,
        _seed,
    )
    _log.info(f"{_seed}: Evaluating final mixture (train set)")
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
        _log,
        _seed,
    )

    out_str = (
        "{}: Finished after {} iterations ({}) =============================\n"
        "NLL: {:.2f} -> {:.2f} (train), {:.2f} -> {:.2f} (test)\n"
        "ANID: {:.2f} -> {:.2f} (train), {:.2f} -> {:.2f} (test)\n"
        "EVD: {:.2f} -> {:.2f} (train), {:.2f} -> {:.2f} (test)\n"
        "{}\n"
        "{}\n"
        "===================================================\n".format(
            _seed,
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
    _log.info(f"{_seed}: Done...")
    result_fname = f"{_seed}.result"
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

    _log.info(f"{_seed}: Done")

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
    _log,
    _seed,
):
    """Evaluate a ElementWorld mixture model
    
    TODO ajs 13/Jan/2020 Optimize this function - it's sloooow!
    
    Args:
        TODO
    
    Returns:
        TODO
    """
    gt_num_clusters = len(gt_mixture_weights)
    num_clusters = len(mixture_weights)

    xtr_p, demos_p = padding_trick(xtr, demos)

    # Measure NLL
    _log.info(f"{_seed}: Evaluating: Measuring NLL")
    nll = solver.mixture_nll(xtr_p, phi, mixture_weights, rewards, demos_p)

    # Measure clustering performance
    _log.info(f"{_seed}: Evaluating: Clustering Performance (NID/ANID)")
    nid = normalized_information_distance(gt_resp, resp)
    anid = adjusted_normalized_information_distance(gt_resp, resp)

    # Compute ILE, EVD matrices
    _log.info(f"{_seed}: Evaluating: Reward Performance (ILE/EVD Matrices)")
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
                vi_kwargs=dict(eps=1e-4),
            )
            ile_mat[learned_mode_idx, gt_mode_idx] = ile
            evd_mat[learned_mode_idx, gt_mode_idx] = evd

    # Measure reward performance
    _log.info(f"{_seed}: Evaluating: Reward Performance (ILE, EVD)")
    mcf_ile, mcf_ile_flowdict = min_cost_flow_error_metric(
        mixture_weights, gt_mixture_weights, ile_mat
    )
    mcf_evd, mcf_evd_flowdict = min_cost_flow_error_metric(
        mixture_weights, gt_mixture_weights, evd_mat
    )

    ml_paths = []
    pdms = []
    fds = []
    if skip_ml_paths:
        _log.info(f"{_seed}: Evaluating: Skipping ML Path Evaluations")
        pass
    else:
        _log.info(f"{_seed}: Evaluating: ML Paths")
        if isinstance(solver, MaxEntEMSolver):
            # Get ML paths from MaxEnt mixture
            ml_paths = element_world_mixture_ml_path(
                xtr, phi, demos, maxent_ml_path, mixture_weights, rewards
            )
        elif isinstance(solver, MaxLikEMSolver):
            # Get ML paths from MaxLik mixture
            ml_paths = element_world_mixture_ml_path(
                xtr, phi, demos, maxlikelihood_ml_path, mixture_weights, rewards
            )
        else:
            raise ValueError

        _log.info(f"{_seed}: Evaluating: % distance missed")
        # Measure % Distance Missed of ML paths
        pdms = np.array(
            [
                percent_distance_missed_metric(ml_path, gt_path)
                for (gt_path, ml_path) in zip(demos, ml_paths)
            ]
        )

        # Measure feature distance of ML paths
        _log.info(f"{_seed}: Evaluating: feature distance")
        fds = np.array(
            [
                phi.feature_distance(gt_path, ml_path)
                for (gt_path, ml_path) in zip(demos, ml_paths)
            ]
        )

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


def run(config_updates, mongodb_url="localhost:27017"):
    """Run a single experiment with the given configuration
    
    Args:
        config_updates (dict): Configuration updates
        mongodb_url (str): MongoDB URL, or None if no Mongo observer should be used for
            this run
    """

    # Dynamically bind experiment config and main function
    ex = Experiment()
    ex.config(base_config)
    ex.main(element_world_v4)

    # Attach MongoDB observer if necessary
    if mongodb_url is not None and not ex.observers:
        ex.observers.append(MongoObserver(url=mongodb_url))

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
        "--num_elements",
        required=False,
        default=3,
        type=int,
        help="Number of elements (ground truth clusters) to use",
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
        "--wind",
        required=False,
        default=0.1,
        type=float,
        help="Random action probability",
    )

    parser.add_argument(
        "--algorithm",
        required=False,
        default="MaxEnt",
        type=str,
        choices=("MaxEnt", "MaxLik", "MeanOnly"),
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
        default=1e-5,
        type=float,
        help="EM convergence tolerance",
    )

    parser.add_argument(
        "--max_iterations",
        required=False,
        default=100,
        type=int,
        help="Maximum number of EM iterations to perform",
    )

    parser.add_argument(
        "--skip_ml_paths",
        action="store_true",
        help="Skip ML path evaluations (speeds up experiment substantially)",
    )

    args = parser.parse_args()
    print("META: Arguments:", args, flush=True)

    config_updates = {
        "num_elements": args.num_elements,
        "num_demos": args.num_demos,
        "demo_skew": args.demo_skew,
        "num_clusters": args.num_clusters,
        "wind": args.wind,
        "algorithm": args.algorithm,
        "initialisation": args.initialisation,
        "em_nll_tolerance": args.em_nll_tolerance,
        "max_iterations": args.max_iterations,
        "skip_ml_paths": args.skip_ml_paths,
    }

    print("META: Configuration: ")
    pprint(config_updates)

    configs = replicate_config(config_updates, args.num_replicates)
    num_workers = get_num_workers(len(configs), args.num_workers)
    print(
        f"META: Distributing {args.num_replicates} replicate(s) over {num_workers} workers"
    )

    mongodb_url = None
    # Read MongoDB URL from config file, if it exists
    mongodb_url = mongo_config()
    print(f"META: MongoDB Server URL: {mongodb_url}")

    # Parallel loop
    with tqdm.tqdm(total=len(configs)) as pbar:
        with futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            tasks = {executor.submit(run, config, mongodb_url) for config in configs}
            for future in futures.as_completed(tasks):
                # arg = tasks[future]; result = future.result()
                pbar.update(1)

    # # Debugging loop
    # for config in tqdm.tqdm(configs):
    #     run(config, mongodb_url)


if __name__ == "__main__":
    main()
