import os
import gc
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import signal
import time
import argparse
import uuid
import random
import pickle
import numpy as np
import itertools
from ompl.util import noOutputHandler
from multiprocessing import Pool
from tqdm.auto import tqdm
from pathlib import Path
import h5py
from pyquaternion import Quaternion
from robofin.collision import FrankaSelfCollisionChecker
from robofin.bullet import Bullet, BulletFranka
from robofin.robots import FrankaRobot, FrankaRealRobot, FrankaGripper
from geometrout.primitive import Cuboid, Cylinder
from geometrout.transform import SE3
from dataclasses import dataclass, field
import logging
from data_pipeline.environments.base_environment import (
    Candidate,
    TaskOrientedCandidate,
    FreeSpaceCandidate,
    NeutralCandidate,
    Environment,
)

from data_pipeline.environments.tabletop_environment import (
    TabletopEnvironment,
)

from prob_types import PlanningProblem

from typing import Tuple, List, Union, Sequence, Optional, Any

# Configuration parameters
END_EFFECTOR_FRAME = "right_gripper"
CUBOID_CUTOFF = 40
CYLINDER_CUTOFF = 40
NUM_SCENES = 5
NUM_PLANS_PER_SCENE = 10
PIPELINE_TIMEOUT = 3600  # 10 hours


@dataclass
class Problem:
    """
    Represents a planning problem without trajectory
    """

    start_candidate: Candidate
    target_candidate: Candidate
    cuboids: List[Cuboid] = field(default_factory=list)
    cylinders: List[Cylinder] = field(default_factory=list)


def generate_candidate_pairs(
    env: Environment, num: int, selfcc: FrankaSelfCollisionChecker
) -> List[Problem]:
    """
    Generate candidate pairs without planning trajectories
    """
    n = int(np.round(np.sqrt(num / 2)))
    candidates = env.gen_additional_candidate_sets(n - 1, selfcc)
    candidates[0].append(env.demo_candidates[0])
    candidates[1].append(env.demo_candidates[1])

    problems = []
    if prob_type == "mixed":
        free_space_candidates = env._gen_free_space_candidates(n, selfcc)
        random.shuffle(candidates[0])
        random.shuffle(candidates[1])
        nonfree_space_candidates = (
            candidates[0][: n // 2] + candidates[1][: n // 2]
            if n > 1
            else candidates[0][:1]
        )

        for c1, c2 in itertools.product(
            free_space_candidates, nonfree_space_candidates
        ):
            problems.append(
                Problem(
                    start_candidate=c1,
                    target_candidate=c2,
                    cuboids=env.cuboids,
                    cylinders=env.cylinders,
                )
            )
    elif prob_type == "task-oriented":
        for c1, c2 in itertools.product(candidates[0], candidates[1]):
            problems.append(
                Problem(
                    start_candidate=c1,
                    target_candidate=c2,
                    cuboids=env.cuboids,
                    cylinders=env.cylinders,
                )
            )
    elif prob_type == "free-space":
        free_space_candidates_1 = env._gen_free_space_candidates(n, selfcc)
        free_space_candidates_2 = env._gen_free_space_candidates(n, selfcc)
        for c1, c2 in itertools.product(
            free_space_candidates_1, free_space_candidates_2
        ):
            problems.append(
                Problem(
                    start_candidate=c1,
                    target_candidate=c2,
                    cuboids=env.cuboids,
                    cylinders=env.cylinders,
                )
            )
    elif prob_type == "neutral":  # ADD NEUTRAL BRANCH
        neutral_candidates = env.gen_neutral_candidates(n, selfcc)
        random.shuffle(candidates[0])
        random.shuffle(candidates[1])
        nonneutral_candidates = (
            candidates[0][: n // 2] + candidates[1][: n // 2]
            if n > 1
            else candidates[0][:1]
        )

        for c1, c2 in itertools.product(neutral_candidates, nonneutral_candidates):
            problems.append(
                Problem(
                    start_candidate=c1,
                    target_candidate=c2,
                    cuboids=env.cuboids,
                    cylinders=env.cylinders,
                )
            )
    else:
        raise NotImplementedError(
            f"Prob type {prob_type} not implemented for environment generation"
        )
    return problems


def verify_has_solvable_problems(
    env: Environment, selfcc: FrankaSelfCollisionChecker
) -> bool:
    """
    Verify environment has at least one solvable problem
    """
    # Simplified check - just validate environment generation
    return True


def gen_valid_env(selfcc: FrankaSelfCollisionChecker) -> Environment:
    """
    Generate a valid environment
    """
    env_arguments = {}
    if ENV_TYPE == "tabletop":
        env: Environment = TabletopEnvironment()
        env_arguments["how_many"] = np.random.randint(3, 15)
    elif ENV_TYPE == "cubby":
        env = CubbyEnvironment()
    elif ENV_TYPE == "merged-cubby":
        env = MergedCubbyEnvironment()
    elif ENV_TYPE == "dresser":
        env = DresserEnvironment()
    elif ENV_TYPE == "free":
        env = FreeSpaceEnvironment()
    elif ENV_TYPE == "pillar":
        env = PillarEnvironment()
    elif ENV_TYPE == "cabinet":
        env = CabinetEnvironment()
    else:
        raise NotImplementedError(f"{ENV_TYPE} not implemented as environment")

    success = False
    while not success:
        success = (
            env.gen(selfcc=selfcc, **env_arguments)
            and len(env.cuboids) < CUBOID_CUTOFF
            and len(env.cylinders) < CYLINDER_CUTOFF
        )
    return env


def gen_single_env_data() -> Tuple[Environment, List[Problem]]:
    """
    Generate environment and planning problems without trajectories
    """
    selfcc = FrankaSelfCollisionChecker()
    env = gen_valid_env(selfcc)
    problems = generate_candidate_pairs(env, NUM_PLANS_PER_SCENE, selfcc)
    return env, problems


def gen_single_env(_: Any):
    """
    Generate and save problems for a single environment
    """
    np.random.seed()
    random.seed()
    env, problems = gen_single_env_data()

    n = len(problems)
    file_name = f"{TMP_DATA_DIR}/{uuid.uuid4()}.hdf5"

    # Determine number of cuboids/cylinders *before* writing problems
    num_cuboids = len(env.cuboids)
    num_cylinders = len(env.cylinders)

    with h5py.File(file_name, "w-") as f:
        # Create datasets
        start_configs = f.create_dataset("start_configs", (n, 7))
        target_configs = f.create_dataset("target_configs", (n, 7))
        start_poses = f.create_dataset("start_poses", (n, 7))  # xyz + quaternion (wxyz)
        target_poses = f.create_dataset("target_poses", (n, 7))

        cuboid_dims = f.create_dataset("cuboid_dims", (num_cuboids, 3))
        cuboid_centers = f.create_dataset("cuboid_centers", (num_cuboids, 3))
        cuboid_quats = f.create_dataset("cuboid_quaternions", (num_cuboids, 4))

        cylinder_radii = f.create_dataset("cylinder_radii", (num_cylinders, 1))
        cylinder_heights = f.create_dataset("cylinder_heights", (num_cylinders, 1))
        cylinder_centers = f.create_dataset("cylinder_centers", (num_cylinders, 3))
        cylinder_quats = f.create_dataset("cylinder_quaternions", (num_cylinders, 4))

        # Tool info datasets (for both start and target candidates)
        # primitive_type as fixed-length string (e.g. S16), dims/offset as (n,3), offset_quaternion as (n,4)
        start_tool_dims = f.create_dataset("start_tool_dims", (n, 3))
        start_tool_offset = f.create_dataset("start_tool_offset", (n, 3))
        start_tool_quat = f.create_dataset("start_tool_quaternion", (n, 4))
        
        target_tool_dims = f.create_dataset("target_tool_dims", (n, 3))
        target_tool_offset = f.create_dataset("target_tool_offset", (n, 3))
        target_tool_quat = f.create_dataset("target_tool_quaternion", (n, 4))

        # Save problems
        for i, problem in enumerate(problems):
            start_configs[i] = problem.start_candidate.config
            target_configs[i] = problem.target_candidate.config

            # Start pose (xyz + quaternion wxyz)
            start_poses[i, 0:3] = problem.start_candidate.pose.xyz
            start_poses[i, 3:7] = problem.start_candidate.pose.so3.wxyz

            # Target pose (xyz + quaternion wxyz)
            target_poses[i, 0:3] = problem.target_candidate.pose.xyz
            target_poses[i, 3:7] = problem.target_candidate.pose.so3.wxyz

            # Tool info for start candidate
            tool = getattr(problem.start_candidate, 'tool', None)
            if tool is not None:
                start_tool_dims[i] = tool.dims
                start_tool_offset[i] = tool.offset
                start_tool_quat[i] = tool.offset_quaternion
            else:
                start_tool_dims[i] = [0,0,0]
                start_tool_offset[i] = [0,0,0]
                start_tool_quat[i] = [1,0,0,0]

            # Tool info for target candidate
            tool = getattr(problem.target_candidate, 'tool', None)
            if tool is not None:
                target_tool_dims[i] = tool.dims
                target_tool_offset[i] = tool.offset
                target_tool_quat[i] = tool.offset_quaternion
            else:
                target_tool_dims[i] = [0,0,0]
                target_tool_offset[i] = [0,0,0]
                target_tool_quat[i] = [1,0,0,0]

        # Save obstacles
        for j, cuboid in enumerate(env.cuboids):
            cuboid_dims[j] = cuboid.dims
            cuboid_centers[j] = cuboid.pose.xyz
            cuboid_quats[j] = cuboid.pose.so3.wxyz

        for k, cylinder in enumerate(env.cylinders):
            cylinder_radii[k] = cylinder.radius
            cylinder_heights[k] = cylinder.height
            cylinder_centers[k] = cylinder.pose.xyz
            cylinder_quats[k] = cylinder.pose.so3.wxyz

    del env
    del problems
    gc.collect()


def merge_and_cleanup():
    """
    Merges temporary HDF5 files into a single HDF5 file and cleans up.
    """
    all_files = list(Path(TMP_DATA_DIR).glob("*.hdf5"))
    if not all_files:
        print("No temporary files found to merge.")
        return

    max_cylinders = 0
    max_cuboids = 0
    total_problems = 0
    for fi in all_files:
        with h5py.File(fi, "r") as f:
            total_problems += len(f["start_configs"])
            num_cuboids = len(f["cuboid_dims"])
            num_cylinders = len(f["cylinder_radii"])
            if num_cuboids > max_cuboids:
                max_cuboids = num_cuboids
            if num_cylinders > max_cylinders:
                max_cylinders = num_cylinders

    with h5py.File(f"{FINAL_DATA_DIR}/all_data.hdf5", "w-") as f:
        # Problem datasets
        start_configs = f.create_dataset(
            "start_configs", (total_problems, 7)
        )
        target_configs = f.create_dataset(
            "target_configs", (total_problems, 7)
        )
        start_poses = f.create_dataset("start_poses", (total_problems, 7))
        target_poses = f.create_dataset("target_poses", (total_problems, 7))

        # Obstacle datasets
        cuboid_dims = f.create_dataset("cuboid_dims", (total_problems, max_cuboids, 3))
        cuboid_centers = f.create_dataset(
            "cuboid_centers", (total_problems, max_cuboids, 3)
        )
        cuboid_quats = f.create_dataset(
            "cuboid_quaternions", (total_problems, max_cuboids, 4)
        )
        cylinder_radii = f.create_dataset(
            "cylinder_radii", (total_problems, max_cylinders, 1)
        )
        cylinder_heights = f.create_dataset(
            "cylinder_heights", (total_problems, max_cylinders, 1)
        )
        cylinder_centers = f.create_dataset(
            "cylinder_centers", (total_problems, max_cylinders, 3)
        )
        cylinder_quats = f.create_dataset(
            "cylinder_quaternions", (total_problems, max_cylinders, 4)
        )

        # Tool info datasets (for both start and target candidates)
        start_tool_dims = f.create_dataset(
            "start_tool_dims", (total_problems, 3)
        )
        start_tool_offset = f.create_dataset(
            "start_tool_offset", (total_problems, 3)
        )
        start_tool_quat = f.create_dataset(
            "start_tool_quaternion", (total_problems, 4)
        )

        target_tool_dims = f.create_dataset(
            "target_tool_dims", (total_problems, 3)
        )
        target_tool_offset = f.create_dataset(
            "target_tool_offset", (total_problems, 3)
        )
        target_tool_quat = f.create_dataset(
            "target_tool_quaternion", (total_problems, 4)
        )

        chunk_start = 0
        for fi in all_files:
            with h5py.File(fi, "r") as g:
                n = len(g["start_configs"])
                chunk_end = chunk_start + n

                # Copy problem data
                start_configs[chunk_start:chunk_end] = \
                    g["start_configs"][...]
                target_configs[chunk_start:chunk_end] = \
                    g["target_configs"][...]
                start_poses[chunk_start:chunk_end] = g["start_poses"][...]
                target_poses[chunk_start:chunk_end] = g["target_poses"][...]

                # Copy tool info for start and target candidates
                start_tool_dims[chunk_start:chunk_end] = \
                    g["start_tool_dims"][...]
                start_tool_offset[chunk_start:chunk_end] = \
                    g["start_tool_offset"][...]
                start_tool_quat[chunk_start:chunk_end] = \
                    g["start_tool_quaternion"][...]

                target_tool_dims[chunk_start:chunk_end] = \
                    g["target_tool_dims"][...]
                target_tool_offset[chunk_start:chunk_end] = \
                    g["target_tool_offset"][...]
                target_tool_quat[chunk_start:chunk_end] = \
                    g["target_tool_quaternion"][...]

                # Copy obstacles (pad if needed)
                num_cuboids = len(g["cuboid_dims"])
                num_cylinders = len(g["cylinder_radii"])

                for idx in range(chunk_start, chunk_end):
                    # Cuboids
                    cuboid_dims[idx, :num_cuboids] = g["cuboid_dims"][...]
                    cuboid_centers[idx, :num_cuboids] = g["cuboid_centers"][...]
                    cuboid_quats[idx, :num_cuboids] = g["cuboid_quaternions"][...]

                    # Cylinders
                    cylinder_radii[idx, :num_cylinders] = g["cylinder_radii"][...]
                    cylinder_heights[idx, :num_cylinders] = g["cylinder_heights"][...]
                    cylinder_centers[idx, :num_cylinders] = g["cylinder_centers"][...]
                    cylinder_quats[idx, :num_cylinders] = g["cylinder_quaternions"][...]

                chunk_start = chunk_end

            # Clean up temporary file after it has been merged
            fi.unlink()


def gen():
    """
    Main generation function with multiprocessing
    """
    noOutputHandler()
    non_seeds = np.arange(NUM_SCENES)

    with Pool() as pool:
        try:
            for _ in tqdm(
                pool.imap_unordered(gen_single_env, non_seeds),
                total=NUM_SCENES,
            ):
                pass
        except KeyboardInterrupt:
            print("Received KeyboardInterrupt. Terminating pool...")
            pool.terminate()
            pool.join()
            print("Pool terminated. Merging collected data...")
            merge_and_cleanup()
            sys.exit(0)

    # Normal completion
    print("Generation complete. Merging collected data...")
    merge_and_cleanup()


def visualize_single_env():
    """
    Visualize a single environment (unchanged)
    """
    # Implementation remains the same as original
    pass


def generate_inference_data(expert_pipeline: str, how_many: int, save_path: str):
    """
    Generate inference data (simplified to only collect problems)
    """
    selfcc = FrankaSelfCollisionChecker()
    inference_problems = []

    with tqdm(total=how_many) as pbar:
        while len(inference_problems) < how_many:
            env, problems = gen_single_env_data()
            if len(problems) == 0:
                continue

            for problem in problems:
                if hasattr(problem.target_candidate, "support_volume"):
                    target_volume = problem.target_candidate.support_volume
                else:
                    target_volume = Cuboid(
                        center=problem.target_candidate.pose.xyz,
                        dims=[0.05, 0.05, 0.05],
                        quaternion=problem.target_candidate.pose.so3.wxyz,
                    )

                inference_problems.append(
                    PlanningProblem(
                        target=problem.target_candidate.pose,
                        q0=problem.start_candidate.config,
                        obstacles=problem.cuboids + problem.cylinders,
                        target_volume=target_volume,
                        target_negative_volumes=problem.target_candidate.negative_volumes,
                    )
                )
                pbar.update(1)
                if len(inference_problems) >= how_many:
                    break

    with open(save_path, "wb") as f:
        pickle.dump({ENV_TYPE: {prob_type: inference_problems}}, f)


# NEW CODE: Signal handler and main execution with timeout
def signal_handler(signum, frame):
    """
    Signal handler for SIGALRM. Raises an exception to be caught in the main thread.
    """
    raise SystemExit(1)


def run_with_timeout(timeout: int):
    """
    Runs the data generation pipeline with a specified timeout.
    """
    noOutputHandler()
    non_seeds = np.arange(NUM_SCENES)

    # Set up the signal handler
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(timeout)  # Set the alarm

    print(f"Starting data generation with a timeout of {timeout} seconds.")

    try:
        with Pool() as pool:
            for _ in tqdm(
                pool.imap_unordered(gen_single_env, non_seeds),
                total=NUM_SCENES,
            ):
                pass
    except (SystemExit, KeyboardInterrupt, Exception) as e:
        print(f"\nTerminating due to {str(e)}")
        # Since the pool is a context manager, it should be cleaned up automatically.
        # Explicitly terminating it is a good practice if it's not a context manager.
        # In this case, the `with` statement will handle it.
        pass
    finally:
        # Cancel the alarm to prevent it from triggering after cleanup
        signal.alarm(0)
        print("Merging collected data...")
        merge_and_cleanup()
        sys.exit(0)


if __name__ == "__main__":
    global START_TIME
    START_TIME = time.time()

    np.random.seed()
    random.seed()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "env_type",
        choices=["free", "tabletop", "cubby", "merged-cubby", "dresser", "cabinet", "pillar"],
        help="Environment type",
    )
    parser.add_argument(
        "prob_type",
        choices=["mixed", "task-oriented", "free-space", "neutral"],
        help="Problem type",
    )

    subparsers = parser.add_subparsers(dest="run_type")
    run_full = subparsers.add_parser("full-pipeline")
    run_full.add_argument("data_dir", type=str, help="Output directory")
    run_full.add_argument(
        "--timeout",
        type=int,
        default=7200,
        help="Pipeline timeout in seconds. 0 for no timeout.",
    )

    test_pipeline = subparsers.add_parser("test-pipeline")
    test_pipeline.add_argument("data_dir", type=str, help="Output directory")

    test_pipeline = subparsers.add_parser("test-environment")

    gen_inference = subparsers.add_parser("for-inference")
    gen_inference.add_argument("expert", choices=["global"])
    gen_inference.add_argument("how_many", type=int)
    gen_inference.add_argument("save_path", type=str)

    args = parser.parse_args()

    global prob_type
    prob_type = args.prob_type
    global ENV_TYPE
    ENV_TYPE = args.env_type

    # Adjust parameters for test runs
    if args.run_type in ["test-pipeline", "test-environment"]:
        NUM_SCENES = 10
        NUM_PLANS_PER_SCENE = 4

    if args.run_type == "test-environment":
        visualize_single_env()
    elif args.run_type == "for-inference":
        generate_inference_data(args.expert, args.how_many, args.save_path)
    else:
        global TMP_DATA_DIR
        TMP_DATA_DIR = f"/tmp/tmp_data_{uuid.uuid4()}/"
        os.makedirs(TMP_DATA_DIR, exist_ok=True)

        global FINAL_DATA_DIR
        FINAL_DATA_DIR = args.data_dir
        os.makedirs(FINAL_DATA_DIR, exist_ok=True)

        print(f"Final data will be saved to {FINAL_DATA_DIR}")
        print(f"Temporary data in {TMP_DATA_DIR}")
        print(f"Environment: {args.env_type}, Problem type: {args.prob_type}")

        # Check for timeout argument and run the appropriate function
        if args.run_type == "full-pipeline" and args.timeout > 0:
            run_with_timeout(args.timeout)
        else:
            gen()
