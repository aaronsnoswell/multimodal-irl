"""Run an experiment on the PuddleWorld environment"""

import os
import pickle
import random
import argparse
import numpy as np
import pandas as pd

from datetime import datetime
from joblib import Parallel, delayed

from puddle_world.envs import CanonicalPuddleWorldEnv

from multimodal_irl.bv_em import init_from_feature_clusters, bv_em_maxent


def nchain_exp(
    rollouts_per_gt_mode, num_learned_modes, init="uniform", replicate=0, verbose=False,
):
    """Perform a single experiment on the PuddleWorld MM-IRL problem
    
    Args:
        rollouts_per_gt_mode (int): Number of rollouts per GT mode
        num_learned_modes (int): Number of modes to learn
        init (str): Initial clustering method, one of 'uniform', 'kmeans', 'gmm'
        replicate (int): Replicate ID of this experiment
        verbose (bool): Print progress information
    
    Returns:
        (list): Number of EM iterations, Responsibility matrix, Reward weights,
            Elapsed runtime in seconds
    """

    from explicit_env.envs.explicit_nchain import ExplicitNChainEnv
    from explicit_env.soln import q_value_iteration, OptimalPolicy

    rollouts = []
    MAX_ROLLOUT_LENGTH = 20
    env = ExplicitNChainEnv()
    env._gamma = 0.99

    # Seed with the replicate to get a consistent experiment across the other parameters
    random.seed(replicate)
    np.random.seed(replicate)
    env.seed(replicate)

    pi = OptimalPolicy(q_value_iteration(env))
    rollouts.extend(
        pi.get_rollouts(env, rollouts_per_gt_mode, max_path_length=MAX_ROLLOUT_LENGTH)
    )

    # Swap optimal actions in the middle, final state
    env._state_action_rewards[[2, 4], :] = env._state_action_rewards[[4, 2], :]
    pi = OptimalPolicy(q_value_iteration(env))
    rollouts.extend(
        pi.get_rollouts(env, rollouts_per_gt_mode, max_path_length=MAX_ROLLOUT_LENGTH)
    )

    # Begin experiment
    t0 = datetime.now()
    initial_mode_weights, _, initial_rsa, _ = init_from_feature_clusters(
        env, rollouts, num_learned_modes, init=init, verbose=verbose
    )
    num_iterations, responsibility_matrix, _, rsa, _ = bv_em_maxent(
        env,
        rollouts,
        num_learned_modes,
        rsa=True,
        initial_mode_weights=initial_mode_weights,
        initial_state_action_reward_weights=initial_rsa,
        verbose=verbose,
    )
    t1 = datetime.now()
    dt = (t1 - t0).total_seconds()

    # Return results
    return [
        num_iterations,
        responsibility_matrix.tolist(),
        rsa.tolist(),
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

    ENVIRONMENT = "NChain"
    TRANSITIONS = "stochastic"
    NUM_GT_CLUSTERS = 2
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
        delayed(nchain_exp)(
            args.rollouts_per_gt_mode,
            args.num_learned_modes,
            args.init,
            replicate,
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
