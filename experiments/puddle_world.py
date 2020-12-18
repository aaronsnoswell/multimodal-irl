"""PuddleWorld experiments

"""
import os
import ast
import tqdm
import pymongo
import argparse

import numpy as np
import pandas as pd

from sacred import Experiment
from sacred.observers import MongoObserver

from pprint import pprint
from datetime import datetime
from concurrent import futures

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

    # Number of rollouts to sample per ground truth behaviour mode (training set)
    tr_rollouts_per_mode = 5

    # Number of rollouts to sample per ground truth behaviour mode (test set)
    te_rollouts_per_mode = 100

    # Minimum and maximum reward parameter values
    reward_range = (-10, 0)

    # Tolerance for Negative Log Likelihood convergence
    tolerance = 1e-5

    # Number of learned clusters
    num_clusters = 2

    # Replicate ID for this experiment
    replicate = 0


# Attach experiment main()
@ex.main
def canonical_puddle_world(
    transition_type,
    environment,
    gt_num_clusters,
    tr_rollouts_per_mode,
    te_rollouts_per_mode,
    algorithm,
    initialisation,
    num_init_restarts,
    num_clusters,
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

    if gt_num_clusters == 3:
        pass
    elif gt_num_clusters == 2:
        # Drop 'any' mode
        gt_rewards = list(gt_rewards.values())[:gt_num_clusters]
    else:
        raise ValueError

    # Get rollouts
    q_stars = []
    pi_stars = []
    tr_rollouts_structured = []
    tr_rollouts = []
    te_rollouts_structured = []
    te_rollouts = []
    for reward in gt_rewards:
        # Get Q* function
        q_star = q_vi(xtr, phi, reward=reward)
        q_stars.append(q_star)

        # Get optimal stochastic policy
        pi_star = OptimalPolicy(q_star)
        pi_stars.append(pi_star)

        # Sample training rollouts from optimal policy
        _tr_rollouts = pi_star.get_rollouts(env, tr_rollouts_per_mode)
        tr_rollouts_structured.append(_tr_rollouts)
        tr_rollouts.extend(_tr_rollouts)

        # Sample distinct testing rollouts from optimal policy
        _te_rollouts = pi_star.get_rollouts(env, te_rollouts_per_mode)
        te_rollouts_structured.append(_te_rollouts)
        te_rollouts.extend(_te_rollouts)

    # Apply padding trick
    xtr_p, phi_p, gt_rewards_p, tr_rollouts_p = padding_trick_mm(
        xtr, phi, gt_rewards, tr_rollouts
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

    if initialisation == "Random":
        # Initialize uniformly at random
        st_mode_weights, st_rewards = solver.init_random(
            phi_p, num_clusters, reward_range
        )
    elif initialisation == "KMeans":
        # Initialize with K-Means (hard) clustering
        st_mode_weights, st_rewards = solver.init_kmeans(
            xtr_p, phi_p, tr_rollouts_p, num_clusters, reward_range, num_init_restarts
        )
    elif initialisation == "GMM":
        # Initialize with GMM (soft) clustering
        st_mode_weights, st_rewards = solver.init_gmm(
            xtr_p, phi_p, tr_rollouts_p, num_clusters, reward_range, num_init_restarts
        )
    else:
        raise ValueError

    (
        tr_resp_history,
        mode_weights_history,
        rewards_history,
        st_nll,
        tr_nll_history,
    ) = bv_em(
        solver,
        xtr_p,
        phi_p,
        tr_rollouts_p,
        num_clusters,
        reward_range,
        mode_weights=st_mode_weights,
        rewards=st_rewards,
        tolerance=tolerance,
    )

    t1 = datetime.now()

    iterations = len(tr_resp_history)

    # Log training progress after experiment - timestamps will be wrong
    for it in range(iterations):
        _run.log_scalar("training.mode_weights", mode_weights_history[it].tolist())
        _run.log_scalar(
            "training.rewards", [r.theta.tolist() for r in rewards_history[it]]
        )
        _run.log_scalar("training.nll", float(tr_nll_history[it]))

    tr_learned_resp = tr_resp_history[-1]
    learned_mode_weights = mode_weights_history[-1]
    learned_rewards = rewards_history[-1]
    tr_nll = tr_nll_history[-1]
    duration = (t1 - t0).total_seconds()

    _log.info("Evaluating...")

    def eval_clustering(
        gt_resp, learned_resp,
    ):
        """Evaluate a mixture model's clustering performance"""

        # Compute cluster metrics
        nid = normalized_information_distance(gt_resp, learned_resp)
        anid = adjusted_normalized_information_distance(gt_resp, learned_resp)

        return nid, anid

    def eval_rewards(
        gt_mode_weights, gt_rewards, learned_mode_weights, learned_rewards,
    ):
        """Evaluate a mixture model's reward performance"""
        gt_num_clusters = len(gt_mode_weights)
        num_clusters = len(learned_mode_weights)

        # Compute reward recovery metrics
        ile_mat = np.zeros((num_clusters, gt_num_clusters))
        evd_mat = np.zeros((num_clusters, gt_num_clusters))
        for learned_mode_idx in range(num_clusters):
            for gt_mode_idx in range(gt_num_clusters):
                ile, evd = ile_evd(
                    xtr,
                    phi,
                    gt_rewards[gt_mode_idx],
                    learned_rewards[learned_mode_idx],
                )
                ile_mat[learned_mode_idx, gt_mode_idx] = ile
                evd_mat[learned_mode_idx, gt_mode_idx] = evd
        mcf_ile, mcf_ile_flowdict = min_cost_flow_error_metric(
            learned_mode_weights, gt_mode_weights, ile_mat
        )
        mcf_evd, mcf_evd_flowdict = min_cost_flow_error_metric(
            learned_mode_weights, gt_mode_weights, evd_mat
        )

        return (mcf_ile, mcf_ile_flowdict, mcf_evd, mcf_evd_flowdict)

    # Lambda to get ground truth responsibility matrix
    gt_resp = lambda k, rpm: (
        np.concatenate([np.repeat([np.eye(k)[r, :]], rpm, 0) for r in range(k)], 0,)
    )

    # Evaluate training set clustering
    tr_nid, tr_anid = eval_clustering(
        gt_resp(gt_num_clusters, tr_rollouts_per_mode), tr_learned_resp
    )
    # Evaluate test set clustering
    te_learned_resp = solver.estep(
        xtr_p, phi_p, learned_mode_weights, learned_rewards, te_rollouts
    )
    te_nid, te_anid = eval_clustering(
        gt_resp(gt_num_clusters, te_rollouts_per_mode), te_learned_resp
    )
    # Evaluate test set NLL
    te_nll = solver.mixture_nll(
        xtr_p, phi_p, learned_mode_weights, learned_rewards, te_rollouts
    )

    # Evaluate reward performance
    (mcf_ile, mcf_ile_flowdict, mcf_evd, mcf_evd_flowdict) = eval_rewards(
        np.ones(gt_num_clusters) / gt_num_clusters,
        gt_rewards,
        learned_mode_weights,
        learned_rewards,
    )

    _log.info("Done...")

    return {
        # Mixture Initialization
        "st_mode_weights": st_mode_weights.tolist(),
        "st_rewards": [st_r.theta.tolist() for st_r in st_rewards],
        "st_nll": float(st_nll),
        #
        # Learned model
        "iterations": int(iterations),
        "duration": float(duration),
        "learned_mode_weights": learned_mode_weights.tolist(),
        "learned_rewards": [learned_r.theta.tolist() for learned_r in learned_rewards],
        #
        # Training set performance
        "tr_learned_resp": tr_learned_resp.tolist(),
        "tr_nll": float(tr_nll),
        "tr_normalized_information_distance": float(tr_nid),
        "tr_normalized_information_distance_adjusted": float(tr_anid),
        #
        # Test set performance
        "te_learned_resp": te_learned_resp.tolist(),
        "te_nll": float(te_nll),
        "te_normalized_information_distance": float(te_nid),
        "te_normalized_information_distance_adjusted": float(te_anid),
        #
        # Reward performance
        "min_cost_flow_ile": float(mcf_ile),
        "min_cost_flow_ile_flow": mcf_ile_flowdict,
        "min_cost_flow_evd": float(mcf_evd),
        "min_cost_flow_evd_flow": mcf_evd_flowdict,
    }


def run(config, mongodb_url="localhost:27017"):
    """Run a single experiment with the given configuration"""

    # Attach MongoDB observer if necessary
    if not ex.observers:
        ex.observers.append(MongoObserver(url=mongodb_url))

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
        "-K",
        "--gt_num_modes",
        required=False,
        default=2,
        choices=(2, 3),
        type=int,
        help="Number of ground truth clusters",
    )

    parser.add_argument(
        "-k",
        "--num_modes",
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

    _base_config = {
        "gt_num_clusters": args.gt_num_modes,
        "num_clusters": args.num_modes,
        "tr_rollouts_per_mode": args.rollouts_per_mode,
        "initialisation": args.init,
        "transition_type": "Stochastic" if args.stochastic else "Deterministic",
    }

    print("META: Base configuration: ")
    pprint(_base_config)

    configs = []
    for replicate in range(args.num_replicates):
        _config = _base_config.copy()
        _config.update({"replicate": replicate})
        configs.append(_config)

    # Try and determine how many CPUs we are allowed to use
    num_cpus = (
        len(os.sched_getaffinity(0))
        # Ask the (linux) OS how many CPUs wer are scheduled to use
        if "sched_getaffinity" in dir(os)
        # If we can't find our scheduled number of CPUs, just use one less than the
        # system's physical socket count - leave one for GUI, bookkeeping etc.
        else os.cpu_count() - 1
    )

    print(f"META: {num_cpus} CPUs available")
    num_workers = min(num_cpus, len(configs))

    print(
        f"META: Distributing {args.num_replicates} replicate(s) over {num_workers} workers"
    )

    # Read MongoDB URL from config file, if it exists
    mongodb_config_file = "mongodb-config.txt"
    mongodb_url = "localhost:27017"
    if os.path.exists(mongodb_config_file):
        with open(mongodb_config_file, "r") as file:
            mongodb_url = file.readline()

    # Parallel loop
    with tqdm.tqdm(total=len(configs)) as pbar:
        with futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            tasks = {executor.submit(run, config, mongodb_url) for config in configs}
            for future in futures.as_completed(tasks):
                # Use arg or result here if desired
                # arg = tasks[future]
                # result = future.result()
                pbar.update(1)

    # # Non-parallel loop for debugging
    # for config in tqdm.tqdm(configs):
    #     run(config)

    print("META: Finished replicate sweep")


if __name__ == "__main__":
    main()
