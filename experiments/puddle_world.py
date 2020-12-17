"""PuddleWorld experiments

"""
import os
import ast
import tqdm
import pymongo
import argparse
import warnings

import numpy as np
import pandas as pd

from sacred import Experiment
from sacred.observers import MongoObserver

from pprint import pprint
from datetime import datetime
from concurrent import futures
from scipy.stats import dirichlet

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from multimodal_irl.bv_em import MaxEntEMSolver, MaxLikEMSolver, bv_em

from multimodal_irl.envs import (
    CanonicalPuddleWorldEnv,
    puddle_world_extras,
)
from mdp_extras import (
    OptimalPolicy,
    padding_trick_mm,
    Linear,
    q_vi,
)

from multimodal_irl.metrics import (
    normalized_information_distance,
    adjusted_normalized_information_distance,
    min_cost_flow_error_metric,
    mean_error_metric,
)
from unimodal_irl.metrics import ile_evd

# Global experiment object
ex = Experiment()

# Attach experiment config
@ex.config
def base_config():

    # Type of transitions
    transition_type = "Stochastic"

    # Environment to use
    environment = "CanonicalPuddleWorld"

    # Ground truth number of clusters
    gt_num_clusters = 2

    # The IRL algorithm to use
    algorithm = "MaxEnt"

    # Initialization for the EM algorithm
    initialisation = "Random"

    # Number of restarts to use for non-random initializations
    num_init_restarts = 5000

    # Number of rollouts to sample per ground truth behaviour mode
    rollouts_per_mode = 5

    # Minimum and maximum reward parameter values
    reward_range = (-10, 0)

    # Tolerance for Negative Log Likelihood convergence
    tolerance = 1e-5

    # Number of learned (training) clusters
    tr_num_clusters = 2

    # Replicate ID for this experiment
    replicate = 0


# Attach experiment main()
@ex.main
def canonical_puddle_world(
    transition_type,
    environment,
    gt_num_clusters,
    rollouts_per_mode,
    algorithm,
    initialisation,
    num_init_restarts,
    tr_num_clusters,
    reward_range,
    tolerance,
    _log,
    _run,
):

    _log.info("Initializing")

    if transition_type == "Stochastic":
        wind = 0.2
    elif transition_type == "Deterministic":
        wind = 0.0
    else:
        raise ValueError

    if environment == "CanonicalPuddleWorld":
        env = CanonicalPuddleWorldEnv(wind=wind)
    else:
        raise ValueError

    xtr, phi, gt_rewards = puddle_world_extras(env)

    if gt_num_clusters == 2:
        # Drop 'any' mode
        gt_rewards = list(gt_rewards.values())[:gt_num_clusters]
    elif gt_num_clusters == 3:
        pass
    else:
        raise ValueError

    # Get rollouts
    q_stars = []
    pi_stars = []
    rollouts_structured = []
    rollouts = []
    for reward in gt_rewards:
        q_star = q_vi(xtr, phi, reward=reward)
        pi_star = OptimalPolicy(q_star)
        _rollouts = pi_star.get_rollouts(env, rollouts_per_mode)

        q_stars.append(q_star)
        pi_stars.append(pi_star)
        rollouts_structured.append(_rollouts)
        rollouts.extend(_rollouts)

    # Apply padding trick
    xtr_p, phi_p, gt_rewards_p, rollouts_p = padding_trick_mm(
        xtr, phi, gt_rewards, rollouts
    )

    # Get solver object
    if algorithm == "MaxEnt":
        solver = MaxEntEMSolver()
    elif algorithm == "MaxLik":
        solver = MaxLikEMSolver()
    else:
        raise ValueError

    _log.info("Solving...")
    t0 = datetime.now()

    # Initialize reward parameters and mode weights randomly
    if initialisation == "Random":

        # Initialize uniformly at random
        st_mode_weights = dirichlet(
            [1.0 / tr_num_clusters for _ in range(tr_num_clusters)]
        ).rvs(1)[0]
        st_rewards = [
            Linear(np.random.uniform(*reward_range, len(phi_p)))
            for _ in range(tr_num_clusters)
        ]

    elif initialisation == "KMeans":

        # Initialize with K-Means (hard) clustering
        feature_mat = np.array(
            [phi_p.expectation([r], xtr_p.gamma) for r in rollouts_p]
        )

        km = KMeans(n_clusters=tr_num_clusters, n_init=num_init_restarts)
        hard_initial_clusters = km.fit_predict(feature_mat)
        soft_initial_clusters = np.zeros((len(rollouts), tr_num_clusters))
        for idx, clstr in enumerate(hard_initial_clusters):
            soft_initial_clusters[idx, clstr] = 1.0

        # Compute initial mode weights from soft clustering
        st_mode_weights = np.sum(soft_initial_clusters, axis=0) / len(rollouts)

        # Compute initial rewards
        st_rewards = solver.mstep(
            xtr_p, phi_p, soft_initial_clusters, rollouts_p, reward_range=reward_range
        )

    elif initialisation == "GMM":

        # Initialize with GMM (soft) clustering
        feature_mat = np.array(
            [phi_p.expectation([r], xtr_p.gamma) for r in rollouts_p]
        )

        gmm = GaussianMixture(n_components=tr_num_clusters, n_init=num_init_restarts)
        gmm.fit(feature_mat)
        soft_initial_clusters = gmm.predict_proba(feature_mat)

        # Compute initial mode weights from soft clustering
        st_mode_weights = np.sum(soft_initial_clusters, axis=0) / len(rollouts)

        # Compute initial rewards
        st_rewards = solver.mstep(
            xtr_p, phi_p, soft_initial_clusters, rollouts_p, reward_range=reward_range
        )

    else:
        raise ValueError
    st_nll = solver.mixture_nll(xtr_p, phi_p, st_mode_weights, st_rewards, rollouts_p)

    resp_history, mode_weights_history, rewards_history, nll_history = bv_em(
        solver,
        xtr_p,
        phi_p,
        rollouts_p,
        tr_num_clusters,
        reward_range,
        mode_weights=st_mode_weights,
        rewards=st_rewards,
        tolerance=tolerance,
    )

    t1 = datetime.now()

    iterations = len(resp_history)

    # Log training progress after experiment - kind of hacky (timestamps will be wrong)
    for it in range(iterations):
        _run.log_scalar("training.mode_weights", mode_weights_history[it].tolist())
        _run.log_scalar(
            "training.rewards", [r.theta.tolist() for r in rewards_history[it]]
        )
        _run.log_scalar("training.nll", float(nll_history[it]))

    learned_resp = resp_history[-1]
    learned_mode_weights = mode_weights_history[-1]
    learned_rewards = rewards_history[-1]
    nll = nll_history[-1]
    duration = (t1 - t0).total_seconds()

    # Prepare ground truth parameters
    gt_responsibility_matrix = np.concatenate(
        [
            np.repeat([np.eye(gt_num_clusters)[row, :]], rollouts_per_mode, 0)
            for row in range(gt_num_clusters)
        ],
        0,
    )
    gt_mode_weights = np.sum(gt_responsibility_matrix, axis=0) / len(
        gt_responsibility_matrix
    )

    _log.info("Evaluating...")

    # Compute cluster metrics
    nid = normalized_information_distance(gt_responsibility_matrix, learned_resp)
    anid = adjusted_normalized_information_distance(
        gt_responsibility_matrix, learned_resp
    )

    # Compute reward recovery metrics
    ile_mat = np.zeros((tr_num_clusters, gt_num_clusters))
    evd_mat = np.zeros((tr_num_clusters, gt_num_clusters))
    for learned_mode_idx in range(tr_num_clusters):
        for gt_mode_idx in range(gt_num_clusters):
            ile, evd = ile_evd(
                xtr, phi, gt_rewards[gt_mode_idx], learned_rewards[learned_mode_idx],
            )
            ile_mat[learned_mode_idx, gt_mode_idx] = ile
            evd_mat[learned_mode_idx, gt_mode_idx] = evd
    mcf_ile, mcf_ile_flowdict = min_cost_flow_error_metric(
        learned_mode_weights, gt_mode_weights, ile_mat
    )
    mcf_evd, mcf_evd_flowdict = min_cost_flow_error_metric(
        learned_mode_weights, gt_mode_weights, evd_mat
    )
    mean_ile = mean_error_metric(learned_resp, gt_responsibility_matrix, ile_mat)
    mean_evd = mean_error_metric(learned_resp, gt_responsibility_matrix, evd_mat)

    _log.info("Done...")

    return {
        "st_mode_weights": st_mode_weights.tolist(),
        "st_rewards": [st_r.theta.tolist() for st_r in st_rewards],
        "st_nll": float(st_nll),
        "iterations": int(iterations),
        "learned_resp": learned_resp.tolist(),
        "learned_mode_weights": learned_mode_weights.tolist(),
        "learned_rewards": [learned_r.theta.tolist() for learned_r in learned_rewards],
        "nll": float(nll),
        "duration": float(duration),
        "normalized_information_distance": float(nid),
        "normalized_information_distance_adjusted": float(anid),
        "min_cost_flow_ile": float(mcf_ile),
        "min_cost_flow_ile_flow": mcf_ile_flowdict,
        "min_cost_flow_evd": float(mcf_evd),
        "min_cost_flow_evd_flow": mcf_evd_flowdict,
        "mean_ile": float(mean_ile),
        "mean_evd": float(mean_evd),
    }


def run(config):
    """Run a single experiment with the given configuration"""

    # Attach MongoDB observer if necessary
    if not ex.observers:
        ex.observers.append(MongoObserver())

    # Run the experiment
    run = ex.run(config_updates=config, options={"--loglevel": "ERROR"})

    # Return the result
    return run.result


def status():
    """Check the MongoDB database for the remaining experiments that need doing"""

    # Open connection
    client = pymongo.MongoClient("mongodb://localhost:27017")

    # Get Sacred DB
    db = client["sacred"]

    # Get experimental runs collection
    runs = db.get_collection("runs")

    # Query for the set of configs in the database
    configs = runs.find({"status": {"$eq": "COMPLETED"}}, {"config": 1, "status": 1})

    # Slice out config objects
    configs = [c["config"] for c in configs]

    config_max_replicates = {}
    for c in configs:
        c2 = c.copy()
        del c2["seed"]
        del c2["replicate"]
        plain_config_str = str(c2)
        replicate_num = c["replicate"]
        cur_replicate_val = config_max_replicates.get(plain_config_str, 0)
        config_max_replicates[plain_config_str] = max(cur_replicate_val, replicate_num)

    vals = []
    for config_str, max_replicates in config_max_replicates.items():
        _config = ast.literal_eval(config_str)
        if max_replicates < 99:
            _config["Status"] = "RUNNING"
        else:
            _config["Status"] = "COMPLETED"
        _config["Max Replicates"] = max_replicates + 1
        vals.append(_config)

    df = pd.DataFrame(vals)

    return df


def main():
    """Main"""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-k",
        "--num_learned_modes",
        required=False,
        default=2,
        type=int,
        help="Number of clusters to learn",
    )

    parser.add_argument(
        "-n",
        "--rollouts_per_mode",
        required=False,
        default=5,
        type=int,
        help="Number of rollouts per mode to give to algorithm",
    )

    parser.add_argument(
        "-i",
        "--init",
        required=False,
        type=str,
        default="Random",
        choices=("Random", "KMeans", "GMM"),
        help="Initialisation method to use",
    )

    parser.add_argument(
        "--stochastic",
        dest="stochastic",
        action="store_true",
        help="Use stochastic PuddleWorld",
    )

    parser.add_argument(
        "--deterministic",
        dest="stochastic",
        action="store_false",
        help="Use deterministic PuddleWorld",
    )

    parser.add_argument(
        "-N",
        "--num_replicates",
        required=False,
        type=int,
        default=100,
        help="Number of replicates to perform",
    )

    args = parser.parse_args()
    print("Arguments:", args, flush=True)

    # Try and determine how many CPUs we are allowed to use
    num_cpus = (
        len(os.sched_getaffinity(0))
        # Ask the (linux) OS how many CPUs wer are scheduled to use
        if "sched_getaffinity" in dir(os)
        # If we can't find our scheduled number of CPUs, just use one less than the
        # system's physical socket count - leave one for GUI, bookkeeping etc.
        else os.cpu_count() - 1
    )

    _base_config = {
        "tr_num_clusters": args.num_learned_modes,
        "rollouts_per_mode": args.rollouts_per_mode,
        "initialisation": args.init,
        "transition_type": "Stochastic" if args.stochastic else "Deterministic",
    }

    print(
        f"META: Distributing {args.num_replicates} replicates over {num_cpus} workers"
    )
    print("META: Base configuration: ")
    pprint(_base_config)

    configs = []
    for replicate in range(args.num_replicates):
        _config = _base_config.copy()
        _config.update({"replicate": replicate})
        configs.append(_config)

    with tqdm.tqdm(total=len(configs)) as pbar:
        with futures.ProcessPoolExecutor(max_workers=num_cpus) as executor:
            tasks = {executor.submit(run, config) for config in configs}
            for future in futures.as_completed(tasks):
                # Use arg or result here if desired
                # arg = tasks[future]
                # result = future.result()
                pbar.update(1)

    print("META: Finished replicate sweep")


if __name__ == "__main__":
    main()