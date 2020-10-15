"""Run an experiment on the PuddleWorld environment"""

import os
import copy
import pickle
import argparse
import random
import pickle

import numpy as np
import pandas as pd

from enum import Enum
from datetime import datetime
from joblib import Parallel, delayed

from puddle_world.envs import CanonicalPuddleWorldEnv
from multimodal_irl.bv_em import (
    init_from_feature_clusters,
    bv_em_maxent,
    responsibilty_matrix_maxent,
)
from explicit_env.soln import (
    value_iteration,
    q_from_v,
    OptimalPolicy,
    policy_evaluation,
)


class TransitionType(Enum):
    """Enum for configuring PuddleWorld experiment transition dynamics"""

    DETERMINISTIC = "deterministic"
    STOCHASTIC = "stochastic"

    @staticmethod
    def get_wind_factor(transition_type):
        if transition_type == TransitionType.DETERMINISTIC:
            return 0.0
        elif transition_type == TransitionType.STOCHASTIC:
            return 0.2
        else:
            raise ValueError


class NumGTModes(Enum):
    """Enum for configuring PuddleWorld experiment number of modes"""

    TWO = 2
    THREE = 3

    @staticmethod
    def get_mode_names(num_gt_modes):
        if num_gt_modes == NumGTModes.TWO:
            return ["wet", "dry"]
        elif num_gt_modes == NumGTModes.THREE:
            return ["wet", "dry", "any"]
        else:
            raise ValueError


class ExperimentConfig:
    """Holder for experimental configuration data"""

    def __init__(self, transition_type, num_gt_modes):
        self.transition_type = transition_type
        self.num_gt_modes = num_gt_modes


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
    
    TODO ajs 15/Oct/2020 use ExperimentConfig here
    
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
    initial_mode_weights, init_rs, _, _ = init_from_feature_clusters(
        _env, rollouts, num_learned_modes, init=init, verbose=verbose
    )
    num_iterations, responsibility_matrix, rs, _, _ = bv_em_maxent(
        _env,
        rollouts,
        num_learned_modes,
        rs=True,
        initial_mode_weights=initial_mode_weights,
        initial_state_reward_weights=init_rs,
        verbose=verbose,
    )
    t1 = datetime.now()
    dt = (t1 - t0).total_seconds()

    # Return results
    return [
        num_iterations,
        responsibility_matrix.tolist(),
        rs.tolist(),
        dt,
    ]


def get_experiment_fixed_inputs(experiment_config):
    """Fetch the fixed input data for a specific experiment
    
    Args:
        experiment_config (ExperimentConfig): Experimental configuration
    
    Returns:
        (dict): Dictionary of experimtal fixed inputs
    """

    # Get mode names
    mode_names = NumGTModes.get_mode_names(experiment_config.num_gt_modes)

    # Load ground truth environments
    environments = [
        CanonicalPuddleWorldEnv(
            mode=mode_name,
            wind=TransitionType.get_wind_factor(experiment_config.transition_type),
        )
        for mode_name in mode_names
    ]

    environment_noreward = copy.deepcopy(environments[0])
    environment_noreward._state_rewards = None

    # Compute RL solution for each mode
    mode_optimal_state_value_functions = [value_iteration(env) for env in environments]
    mode_optimal_state_action_value_functions = [
        q_from_v(v, env)
        for v, env in zip(mode_optimal_state_value_functions, environments)
    ]
    mode_optimal_deterministic_policies = [
        OptimalPolicy(q, stochastic=False)
        for q in mode_optimal_state_action_value_functions
    ]
    mode_optimal_stochastic_policies = [
        OptimalPolicy(q, stochastic=True)
        for q in mode_optimal_state_action_value_functions
    ]
    mode_optimal_policy_state_value_functions = [
        policy_evaluation(env, pi)
        for env, pi in zip(environments, mode_optimal_deterministic_policies)
    ]

    # Load rollout data
    rollout_filename = "pw-{}-{}.pkl"
    mode_rollouts = []
    for mode_name in mode_names:
        with open(
            rollout_filename.format(experiment_config.transition_type.value, mode_name),
            "rb",
        ) as file:
            mode_rollouts.append(pickle.load(file))

    result = {
        "mode_names": mode_names,
        "environments": environments,
        "environment_noreward": environment_noreward,
        "mode_optimal_state_value_functions": mode_optimal_state_value_functions,
        "mode_optimal_state_action_value_functions": mode_optimal_state_action_value_functions,
        "mode_optimal_deterministic_policies": mode_optimal_deterministic_policies,
        "mode_optimal_stochastic_policies": mode_optimal_stochastic_policies,
        "mode_optimal_policy_state_value_functions": mode_optimal_policy_state_value_functions,
        "mode_rollouts": mode_rollouts,
    }

    return result


def get_experiment_train_inputs(fixed_inputs, exp):
    """Get the training inputs given an experimental result slice
    
    Args:
        fixed_inputs (dict): Fixed experimental input dictionary
        exp (pandas Series): One row from the experimental result table
    
    Returns:
        (dict): Dictionary of experimental training inputs
    """

    num_gt_modes = int(exp["Num GT Clusters"])
    num_learned_modes = int(exp["Num Learned Clusters"])
    num_rollouts = int(exp["Num Rollouts"])
    num_rollouts_per_mode = int(num_rollouts // num_gt_modes)
    replicate_id = int(exp["Replicate"])

    # Get the exact set of training rollouts used for this experiment
    rollout_start_idx = int(replicate_id * num_rollouts_per_mode)
    rollout_end_idx = int((replicate_id + 1) * num_rollouts_per_mode)
    rollout_slice = slice(rollout_start_idx, rollout_end_idx)
    rollouts = []
    for mode_rollouts in fixed_inputs["mode_rollouts"]:
        rollouts.extend(mode_rollouts[rollout_slice])

    # Compute train GT responsibility matrix
    responsibility_matrix_gt = np.ones((len(rollouts), num_gt_modes))
    for ri in range(responsibility_matrix_gt.shape[0]):
        rollout = rollouts[ri]
        for mi in range(responsibility_matrix_gt.shape[1]):
            env = fixed_inputs["environments"][mi]
            optimal_policy = fixed_inputs["mode_optimal_stochastic_policies"][mi]
            for s, a in rollout[:-1]:
                responsibility_matrix_gt[
                    ri, mi
                ] *= optimal_policy.prob_for_state_action(s, a)
            responsibility_matrix_gt[ri, mi] *= np.exp(
                env.path_log_probability(rollout)
            )
    responsibility_matrix_gt /= np.sum(responsibility_matrix_gt, axis=1, keepdims=True)

    # The ground truth mode weights are uniform
    mode_weights_gt = np.ones(responsibility_matrix_gt.shape[1])
    mode_weights_gt /= np.sum(mode_weights_gt)

    result = {
        "num_gt_modes": num_gt_modes,
        "num_learned_modes": num_learned_modes,
        "num_rollouts": num_rollouts,
        "num_rollouts_per_mode": num_rollouts_per_mode,
        "rollouts": rollouts,
        "responsibility_matrix_gt": responsibility_matrix_gt,
        "mode_weights_gt": mode_weights_gt,
    }

    return result


def get_experiment_test_inputs(fixed_inputs, exp):
    """Get testing inputs given an experimental result slice
    
    Args:
        fixed_inputs (dict): Fixed experimental input dictionary
        exp (pandas Series): One row from the experimental result table
    
    Returns:
        (dict): Dictionary of experimental testing inputs
    """

    num_gt_modes = int(exp["Num GT Clusters"])
    num_learned_modes = int(exp["Num Learned Clusters"])
    num_rollouts = int(exp["Num Rollouts"])
    num_rollouts_per_mode = int(num_rollouts // num_gt_modes)
    replicate_id = int(exp["Replicate"])

    # Get a testing set of rollouts
    rollouts = []
    for mode_rollouts in fixed_inputs["mode_rollouts"]:
        rollouts.extend(random.sample(mode_rollouts, num_rollouts_per_mode))

    # Compute test GT responsibility matrix
    responsibility_matrix_gt = np.ones((len(rollouts), num_gt_modes))
    for ri in range(responsibility_matrix_gt.shape[0]):
        rollout = rollouts[ri]
        for mi in range(responsibility_matrix_gt.shape[1]):
            env = fixed_inputs["environments"][mi]
            optimal_policy = fixed_inputs["mode_optimal_stochastic_policies"][mi]
            for s, a in rollout[:-1]:
                responsibility_matrix_gt[
                    ri, mi
                ] *= optimal_policy.prob_for_state_action(s, a)
            responsibility_matrix_gt[ri, mi] *= np.exp(
                env.path_log_probability(rollout)
            )
    responsibility_matrix_gt /= np.sum(responsibility_matrix_gt, axis=1, keepdims=True)

    result = {
        "num_gt_modes": num_gt_modes,
        "num_learned_modes": num_learned_modes,
        "num_rollouts": num_rollouts,
        "num_rollouts_per_mode": num_rollouts_per_mode,
        "rollouts": rollouts,
        "responsibility_matrix_gt": responsibility_matrix_gt,
    }

    return result


def get_experiment_outputs(fixed_inputs, exp, test_inputs):
    """Get the outputs of our experiment
    
    Args:
        fixed_inputs (dict): Dictionary of experimtal fixed inputs
        exp (pandas Series): One row from the experimental result table
        (dict): Dictionary of experimental training inputs
        (dict): Dictionary of experimental testing inputs
        
    Returns:
        (dict): Dictionary of experimental outputs
    """

    # Get learned model
    state_reward_parameters_learned = np.array(eval(exp["Reward Weights"]))
    responsibility_matrix_train = np.array(eval(exp["Responsibility Matrix"]))
    mode_weights_learned = (
        np.sum(responsibility_matrix_train, axis=0)
        / responsibility_matrix_train.shape[0]
    )

    # Compute test set clustering
    responsibility_matrix_test = responsibilty_matrix_maxent(
        fixed_inputs["environment_noreward"],
        test_inputs["rollouts"],
        test_inputs["num_learned_modes"],
        state_reward_weights=state_reward_parameters_learned,
        mode_weights=mode_weights_learned,
    )

    result = {
        "state_reward_parameters_learned": state_reward_parameters_learned,
        "mode_weights_learned": mode_weights_learned,
        "responsibility_matrix_train": responsibility_matrix_train,
        "responsibility_matrix_test": responsibility_matrix_test,
    }

    return result


if __name__ == "__main__":

    # TODO ajs 15/Oct/2020 use ExperimentConfig here

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
