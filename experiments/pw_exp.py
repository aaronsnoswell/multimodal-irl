"""Run an experiment on the PuddleWorld environment"""

import os
import pickle
import argparse
import numpy as np
import pandas as pd

from datetime import datetime
from joblib import Parallel, delayed

from puddle_world.envs import CanonicalPuddleWorldEnv

from multimodal_irl.bv_em import reward_weights_from_feature_clusters, bv_em_maxent


def pw_exp(
    rollouts_per_gt_mode,
    num_learned_modes,
    init="uniform",
    replicate=0,
    stochastic=False,
    three_gt_modes=False,
    verbose=False,
):
    """Perform a single experiment on the PuddleWorld MM-IRL problem
    
    Args:
        rollouts_per_gt_mode (int): Number of rollouts per GT mode
        num_learned_modes (int): Number of modes to learn
        init (str): Initial clustering method, one of 'uniform', 'kmeans', 'gmm'
        replicate (int): Replicate ID of this experiment
        stochastic (bool): Use stochastic environment?
        three_gt_modes (bool): Use three GT modes (if False, use two)
        verbose (bool): Print progress information
    
    Returns:
        (list): Number of EM iterations, Responsibility matrix, Reward weights,
            Elapsed runtime in seconds
    """

    STOCHASTIC_WIND = 0.2

    # Get rollouts and environment template
    if stochastic:
        _env = CanonicalPuddleWorldEnv(wind=STOCHASTIC_WIND)
        rollout_filenames = ["pw-stochastic-wet.pkl", "pw-stochastic-dry.pkl"]
        if three_gt_modes:
            rollout_filenames.append("pw-stochastic-any.pkl")
    else:
        _env = CanonicalPuddleWorldEnv(wind=0.0)
        rollout_filenames = ["pw-deterministic-wet.pkl", "pw-deterministic-dry.pkl"]
        if three_gt_modes:
            rollout_filenames.append("pw-deterministic-any.pkl")
    all_rollouts = []
    for rollout_filename in rollout_filenames:
        with open(rollout_filename, "rb") as file:
            all_rollouts.append(pickle.load(file))

    # Slice out rollouts for this experiment
    start_idx = replicate * rollouts_per_gt_mode
    end_idx = (replicate + 1) * rollouts_per_gt_mode
    rollouts = []
    for mode_rollouts in all_rollouts:
        rollouts.extend(mode_rollouts[start_idx:end_idx])

    # Begin experiment
    t0 = datetime.now()
    initial_mode_weights, initial_reward_weights = init_from_feature_clusters(
        _env, rollouts, num_learned_modes, init=init, verbose=verbose
    )
    num_iterations, responsibility_matrix, _, reward_weights = bv_em_maxent(
        _env,
        rollouts,
        num_learned_modes,
        initial_mode_weights=initial_mode_weights,
        initial_reward_weights=initial_reward_weights,
        verbose=verbose,
    )
    t1 = datetime.now()
    dt = (t1 - t0).total_seconds()

    # Return results
    return [
        num_iterations,
        responsibility_matrix.tolist(),
        reward_weights.tolist(),
        dt,
    ]


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-n",
        "--rollouts_per_gt_mode",
        required=True,
        type=int,
        help="Number of rollouts per mode to give to algorithm.",
    )

    parser.add_argument(
        "-k",
        "--num_learned_modes",
        required=True,
        type=int,
        help="Number of clusters to learn.",
    )

    parser.add_argument(
        "-i",
        "--init",
        required=False,
        type=str,
        default="uniform",
        choices=("uniform", "kmeans", "gmm"),
        help="Initialisation method to use. Default is 'uniform' (no up-front clustering).",
    )

    parser.add_argument(
        "-N",
        "--num_replicates",
        required=False,
        type=int,
        default=1,
        help="Number of replicates to perform.",
    )

    parser.add_argument(
        "-s",
        "--stochastic",
        required=False,
        action="store_true",
        default=False,
        help="Use stochastic PuddleWorld?",
    )

    parser.add_argument(
        "-t",
        "--three_gt_modes",
        required=False,
        action="store_true",
        default=False,
        help="Use three GT modes?",
    )

    args = parser.parse_args()
    print("Arguments:", args, flush=True)

    # Prepare dataframe for experimental results
    df = pd.DataFrame(
        columns=[
            # Independent Variables
            "Environment",
            "Transition Type",
            "Num GT Clusters",
            "Num Learned Clusters",
            "Num Rollouts",
            "Algorithm",
            "Initialisation",
            "Replicate",
            #
            # Dependent Variables
            "Iterations",
            "Responsibility Matrix",
            "Reward Weights",
            "Runtime (s)",
        ],
        index=range(args.num_replicates),
    )
    df["Responsibility Matrix"] = df["Responsibility Matrix"].astype("object")
    df["Reward Weights"] = df["Reward Weights"].astype("object")

    ENVIRONMENT = "CanonicalPuddleWorld"
    TRANSITIONS = "stochastic" if args.stochastic else "deterministic"
    NUM_GT_CLUSTERS = 3 if args.three_gt_modes else 2
    NUM_CLUSTERS = args.num_learned_modes
    NUM_ROLLOUTS_PER_MODE = args.rollouts_per_gt_mode
    NUM_ROLLOUTS = NUM_ROLLOUTS_PER_MODE * NUM_GT_CLUSTERS
    NUM_REPLICATES = args.num_replicates
    ALGORITHM = "BV-MaxEnt"
    INITIALISATION = args.init

    # Unique results filename for this experiment
    RESULTS_FILENAME = f"cmd-{ENVIRONMENT}-{TRANSITIONS}-{NUM_GT_CLUSTERS}-{INITIALISATION}-{NUM_ROLLOUTS_PER_MODE}-{NUM_CLUSTERS}.csv"

    # Try and determine how many CPUs we are allowed to use
    num_cpus = (
        len(os.sched_getaffinity(0))
        # Ask the (linux) OS how many CPUs wer are scheduled to use
        if "sched_getaffinity" in dir(os)
        # If we can't find our scheduled number of CPUs, just use one less than the
        # system's physical socket count - leave one for GUI, bookkeeping etc.
        else os.cpu_count() - 1
    )

    # Determine how many CPUs we actually need
    num_jobs = max(min(num_cpus, args.num_replicates), 1)
    if num_jobs > 1:
        print(f"Parallelizing over {num_jobs} CPU(s)", flush=True)

    # Parallelize over replicates
    results = Parallel(
        n_jobs=num_jobs,
        # prefer="threads"
    )(
        delayed(pw_exp)(
            args.rollouts_per_gt_mode,
            args.num_learned_modes,
            args.init,
            replicate,
            args.stochastic,
            args.three_gt_modes,
            verbose=True,
        )
        for replicate in range(args.num_replicates)
    )

    # Copy results to dataframe
    for ri, r in enumerate(results):

        df.iloc[ri] = [
            ENVIRONMENT,
            TRANSITIONS,
            NUM_GT_CLUSTERS,
            NUM_CLUSTERS,
            NUM_ROLLOUTS,
            ALGORITHM,
            INITIALISATION,
            ri,
            *r,
        ]

    print(f"Saving results to {RESULTS_FILENAME}", flush=True)
    df.to_csv(RESULTS_FILENAME)
