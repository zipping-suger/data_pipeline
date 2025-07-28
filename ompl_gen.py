import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import argparse
import uuid
import random
import pickle
import numpy as np
from ompl.util import noOutputHandler
from multiprocessing import Pool
from tqdm.auto import tqdm
from pathlib import Path
import h5py
from pyquaternion import Quaternion
from robofin.collision import FrankaSelfCollisionChecker
from robofin.bullet import Bullet, BulletFranka
from robofin.robots import FrankaRobot, FrankaRealRobot, FrankaGripper
from atob.planner import (
    FrankaAITStarPlanner,
    FrankaRRTConnectPlanner,
    FrankaAITStarHandPlanner,
)
import itertools
from geometrout.primitive import Cuboid, Cylinder
from geometrout.transform import SE3
from termcolor import colored
from dataclasses import dataclass, field
import logging
from atob.trajectory import Trajectory
from data_pipeline.environments.base_environment import (
    Candidate,
    TaskOrientedCandidate,
    FreeSpaceCandidate,
    NeutralCandidate,
    Environment,
)
from data_pipeline.environments.cubby_environment import (
    CubbyEnvironment,
    MergedCubbyEnvironment,
)
from data_pipeline.environments.dresser_environment import (
    DresserEnvironment,
)
from data_pipeline.environments.tabletop_environment import (
    TabletopEnvironment,
)

from data_pipeline.environments.free_environment import (
    FreeSpaceEnvironment
)

from prob_types import PlanningProblem

from typing import Tuple, List, Union, Sequence, Optional, Any

# These are the current config parameters used throughout the script.
END_EFFECTOR_FRAME = "right_gripper"  # Used everywhere and is the default in robofin
MAX_JERK = 0.15  # Used for validating the expert trajectories
SEQUENCE_LENGTH = 50  # The final sequence length
NUM_SCENES = 2  # The maximum number of scenes to generate in a single job
NUM_PLANS_PER_SCENE = (
    8  # The number of total candidate start or goals to use to plan experts
)
PIPELINE_TIMEOUT = 36000  # 10 hours in seconds--after which all new scenes will immediately return nothing

# This parameter dictates the maximum number of cuboids to be used in an environment
# Some environments have random generation methods and may generate outliers that are extremely complicated
CUBOID_CUTOFF = 40
CYLINDER_CUTOFF = 40


@dataclass
class Result:
    """
    Describes an individual result from a single planning problem
    """

    start_candidate: Candidate
    target_candidate: Candidate
    error_codes: List[str] = field(default_factory=list)
    cuboids: List[Cuboid] = field(default_factory=list)
    cylinders: List[Cylinder] = field(default_factory=list)
    global_solution: np.ndarray = field(default_factory=lambda: np.array([]))


def solve_global_plan(
    start_candidate: Candidate,
    target_candidate: Candidate,
    obstacles: List[Union[Cuboid, Cylinder]],
    selfcc: FrankaSelfCollisionChecker,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Runs AIT* and smoothing to solve global plan and checks to make sure it doesn't have collisions
    or other weird errors

    :param start_candidate Candidate: The candidate for the start configuration
    :param target_candidate Candidate: The candidate for the target configuration
    :param obstacles List[Union[Cuboid, Cylinder]]: The obstacles in the scene
    :param selfcc FrankaSelfCollisionChecker: Checks for self collisions using spheres that
                                              mimic the internal Franka collision checker.
    :rtype Tuple[np.ndarray, np.ndarray]: The path going forward and backward (these paths
                                          will not be exactly the same because smoothing
                                          is run separately on each
    """
    planner = FrankaAITStarPlanner()
    sim = Bullet(gui=False)
    sim.load_primitives(obstacles)
    robot = sim.load_robot(FrankaRobot)
    planner.load_simulation(sim, robot)
    planner.load_self_collision_checker(selfcc)
    path = planner.plan(
        start=start_candidate.config,
        goal=target_candidate.config,
        max_runtime=20,
        min_solution_time=15,
        exact=True,
        shortcut=True,
        spline=True,
        verbose=False,
    )
    if path is None:
        return np.array([]), np.array([])
    forward_smoothed = np.asarray(planner.smooth(path, SEQUENCE_LENGTH))
    for q in forward_smoothed:
        robot.marionette(q)
        if sim.in_collision(robot, check_self=True) or selfcc.has_self_collision(q):
            return np.array([]), np.array([])
    backward_smoothed = np.asarray(planner.smooth(path[::-1], SEQUENCE_LENGTH))
    for q in backward_smoothed:
        robot.marionette(q)
        if sim.in_collision(robot, check_self=True) or selfcc.has_self_collision(q):
            return np.array([]), np.array([])
    return forward_smoothed, backward_smoothed


def downsample(trajectory: Sequence[np.ndarray]) -> Optional[np.ndarray]:
    """
    Retimes the trajectory to have constant-ish velocity

    :param trajectory Sequence[np.ndarray]: The trajectory
    :rtype Optional[List[np.ndarray]]: The output trajectory, returns None if there is an error
    """
    with np.errstate(over="raise", divide="raise", under="raise", invalid="raise"):
        try:
            sampled = Trajectory.from_path(
                trajectory, length=SEQUENCE_LENGTH
            ).milestones
        except Exception as _:
            return None
    return np.asarray(sampled)


def has_high_jerk(trajectory: np.ndarray) -> bool:
    """
    Checks whether the trajectory has high jerk as a proxy for smoothness

    :param trajectory np.ndarray: The trajectory
    :rtype bool: Whether the jerk is too high
    """
    velocities = []
    accelerations = []
    for qi, qj in zip(trajectory[:-1], trajectory[1:]):
        velocities.append(qj - qi)
    for vi, vj in zip(velocities[:-1], velocities[1:]):
        accelerations.append(vj - vi)
    for ai, aj in zip(accelerations[:-1], accelerations[1:]):
        max_jerk = np.max(np.abs(aj - ai))
        if max_jerk > MAX_JERK:
            return True
    return False


def has_self_collision(
    trajectory: np.ndarray, selfcc: FrankaSelfCollisionChecker
) -> bool:
    """
    Checks whether there are any self collisions according to Franka's internal controller

    :param trajectory np.ndarray: The trajectory
    :param selfcc FrankaSelfCollisionChecker: Checks for self collisions using spheres that
                                              mimic the internal Franka collision checker.
    :rtype bool: Whether the robot has a self collision (or whether the robot would report
                 a self collision)
    """
    for q in trajectory:
        if selfcc.has_self_collision(q):
            return True
    return False


def in_collision(trajectory: np.ndarray, sim: Bullet, robot: BulletFranka) -> bool:
    """
    Checks whether the trajectory colllides with the environment

    :param trajectory np.ndarray: The trajectory
    :param sim Bullet: A bullet simulator object loaded up with the scene and robot
    :param robot BulletFranka: The robot in the simulator
    :rtype bool: Whether the simulattor reports a collision
    """
    for i, q in enumerate(trajectory):
        robot.marionette(q)
        if sim.in_collision(robot, check_self=True):
            return True
    return False


def violates_joint_limits(trajectory: np.ndarray) -> bool:
    """
    Checks whether the solution lies within the empirically determined Franka joint limits.
    These joint limits represent what we were actually able to get the Franka to perform and
    are a subset of the published limits.

    :param trajectory np.ndarray: The trajectory
    :rtype bool: Whether any of the configurations violate joint limits
    """
    for i, q in enumerate(trajectory):
        if not FrankaRealRobot.within_limits(q):
            return True
    return False


def verify_trajectory(
    sim: Bullet,
    robot: BulletFranka,
    trajectory: np.ndarray,
    final_pose: SE3,
    goal_pose: SE3,
    selfcc: FrankaSelfCollisionChecker,
) -> List[str]:
    """
    Runs a set of checks on the trajectory to determine wheter to keep it or
    throw it out

    :param sim Bullet: A bullet simulator object loaded up with the scene
    :param robot BulletFranka: The robot object in the simulator scene
    :param trajectory np.ndarray: The trajectory
    :param final_pose SE3: The final pose in the trajectory
    :param goal_pose SE3: The goal pose
    :param selfcc FrankaSelfCollisionChecker: Checks for self collisions using spheres that
                                              mimic the internal Franka collision checker.
    :rtype List[str]: A list of strings representing any errors in this trajectory (can be
                      used to filter out bad trajectories and/or to collect metrics on failures
    """

    error_codes = []
    if np.linalg.norm(final_pose._xyz - goal_pose._xyz) > 0.05:
        error_codes.append("miss")
    if has_high_jerk(trajectory):
        error_codes.append("high jerk")
    if has_self_collision(trajectory, selfcc):
        error_codes.append("self collision")
    if in_collision(trajectory, sim, robot):
        error_codes.append("collision")
    if violates_joint_limits(trajectory):
        error_codes.append("joint limit")
    return error_codes


def forward_backward(
    candidate1: Candidate,
    candidate2: Candidate,
    cuboids: List[Cuboid],
    cylinders: List[Cylinder],
    selfcc: FrankaSelfCollisionChecker,
) -> List[Result]:
    """
    Run the OMPL expert pipeline going forward and backward between the two candidates

    :param candidate1 Candidate: The first candidate
    :param candidate2 Candidate: The second candidate
    :param cuboids List[Cuboid]: The cuboids in the scene
    :param cylinders List[Cylinder]: The cylinders in the scene
    :param selfcc FrankaSelfCollisionChecker: Checks for self collisions using spheres that
                                              mimic the internal Franka collision checker.
    :rtype List[Result]: The two results, one for going from `candidate1` to `candidate2`
                         and one for going from `candidate2` to `candidate1`
    """
    sim = Bullet(gui=False)
    arm = sim.load_robot(FrankaRobot)
    sim.load_primitives(cuboids + cylinders)

    global_forward, global_backward = solve_global_plan(
        candidate1, candidate2, cuboids + cylinders, selfcc
    )
    if len(global_forward) != len(global_backward):
        logging.warning(
            "Length of global forward and backward solutions are different--something might be buggy"
        )
    if len(global_forward) == 0 or len(global_backward) == 0:
        return []
    forward_result = Result(
        global_solution=global_forward,
        cuboids=cuboids,
        cylinders=cylinders,
        start_candidate=candidate1,
        target_candidate=candidate2,
    )
    backward_result = Result(
        global_solution=global_backward,
        cuboids=cuboids,
        cylinders=cylinders,
        start_candidate=candidate2,
        target_candidate=candidate1,
    )
    results = [forward_result, backward_result]
    return results


def exhaust_environment(
    env: Environment, num: int, selfcc: FrankaSelfCollisionChecker
) -> List[Result]:
    """
    Given a valid environment, i.e. one with at least one solvable problem,
    generate a bunch of candidates and plan between them.

    Generates roughly `num` problems in this environment and tries to solve them

    :param env Environment: The environment in which to find plans
    :param num int: The approximate number of plans to generate for this environment
    :param selfcc FrankaSelfCollisionChecker: Checks for self collisions using spheres that
                                              mimic the internal Franka collision checker.
    :rtype List[Result]: The results for this environment
    """
    n = int(np.round(np.sqrt(num / 2)))
    candidates = env.gen_additional_candidate_sets(n - 1, selfcc)
    candidates[0].append(env.demo_candidates[0])
    candidates[1].append(env.demo_candidates[1])

    results = []
    if prob_type == "mixed":
        free_space_candidates = env._gen_free_space_candidates(n, selfcc)
        random.shuffle(candidates[0])
        random.shuffle(candidates[1])
        if n <= 1:
            # This code path exists for testing purposes to make sure the
            # pipeline is working. In a typical usecase, you should be generating more
            # data than this
            nonfree_space_candidates = candidates[0][:1]
        else:
            nonfree_space_candidates = candidates[0][: n // 2] + candidates[1][: n // 2]
        for c1, c2 in itertools.product(free_space_candidates, nonfree_space_candidates):
            results.extend(forward_backward(c1, c2, env.cuboids, env.cylinders, selfcc))
    elif prob_type == "neutral":
        neutral_candidates = env.gen_neutral_candidates(n, selfcc)
        random.shuffle(candidates[0])
        random.shuffle(candidates[1])
        if n <= 1:
            # This code path exists for testing purposes to make sure the
            # pipeline is working. In a typical usecase, you should be generating more
            # data than this
            nonneutral_candidates = candidates[0][:1]
        else:
            nonneutral_candidates = candidates[0][: n // 2] + candidates[1][: n // 2]
        for c1, c2 in itertools.product(neutral_candidates, nonneutral_candidates):
            results.extend(forward_backward(c1, c2, env.cuboids, env.cylinders, selfcc))
    elif prob_type == "task-oriented":
        for c1, c2 in itertools.product(candidates[0], candidates[1]):
            results.extend(forward_backward(c1, c2, env.cuboids, env.cylinders, selfcc))
    elif prob_type == "free-space":
         free_space_candidates_1 = env._gen_free_space_candidates(n, selfcc)
         free_space_candidates_2 = env._gen_free_space_candidates(n, selfcc)
         for c1, c2 in itertools.product(free_space_candidates_1, free_space_candidates_2):
            results.extend(forward_backward(c1, c2, env.cuboids, env.cylinders, selfcc))
    else:
        raise NotImplementedError(
            f"Prob type {prob_type} not implemented for environment generation"
        )
    return results


def verify_has_solvable_problems(
    env: Environment, selfcc: FrankaSelfCollisionChecker
) -> bool:
    """
    Every environment class has a pair of "demo" candidates that represent a possible
    problem. This function verifies that there is in fact a valid path being these
    demo candidates. Note that this is not an exhaustive search on whether any solvable
    plan exists in the environment and instead if meant to weed out any that don't
    immediately have a solution that BiRRT can find within 10 seconds.

    :param env Environment: The environment
    :param selfcc FrankaSelfCollisionChecker: Checks for self collisions using spheres that
                                              mimic the internal Franka collision checker.
    :rtype bool: Whether a path exists between the demo candidates
    """
    planner = FrankaRRTConnectPlanner()
    sim = Bullet(gui=False)
    sim.load_primitives(env.obstacles)
    robot = sim.load_robot(FrankaRobot)
    planner.load_simulation(sim, robot)
    planner.load_self_collision_checker(selfcc)
    path = planner.plan(
        start=env.demo_candidates[0].config,
        goal=env.demo_candidates[1].config,
        max_runtime=10,  # Change me if the environment is super hard
        exact=True,
        verbose=False,
    )
    if path is None:
        return False
    return True


def gen_valid_env(selfcc: FrankaSelfCollisionChecker) -> Environment:
    """
    Generates the environment itself, based on what subtype was specified to the program
    (and is set in the global variable).

    :param selfcc FrankaSelfCollisionChecker: Checks for self collisions using spheres that
                                              mimic the internal Franka collision checker.
    :rtype Environment: A successfully generated environment
    """
    # TODO Replace the subtype check with the correct subtype or remove
    # if environment class doesn't have subtypes
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
    else:
        raise NotImplementedError(f"{ENV_TYPE} not implemented as environment")
    success = False
    # Continually regenerate environment until there is at least one valid solution
    while not success:
        # You can pass any other parameters into gen that you want here
        success = (
            env.gen(selfcc=selfcc, **env_arguments)
            and len(env.cuboids) < CUBOID_CUTOFF
            and len(env.cylinders) < CYLINDER_CUTOFF
        )
        if success:
            success = verify_has_solvable_problems(env, selfcc)
    return env


def gen_single_env_data() -> Tuple[Environment, List[Result]]:
    """
    Generates an environment and a bunch of trajectories in it.

    :rtype Tuple[Environment, List[Result]]: The environment and the trajectories in it
    """
    # The physical Franka's internal collision checker is more conservative than Bullet's
    # This will allow for more realistic collision checks
    selfcc = FrankaSelfCollisionChecker()

    env = gen_valid_env(selfcc)
    results = exhaust_environment(env, NUM_PLANS_PER_SCENE, selfcc)
    return env, results


def gen_single_env(_: Any):
    """
    Calls `gen_single_env_data` to generates a bunch of trajectories for a
    single environment and then saves them to a temporary file.

    :param _ Any: This is a throwaway needed to run the multiprocessing
    """
    # If we're already past the timeout, do nothing
    if time.time() - START_TIME > PIPELINE_TIMEOUT:
        return
    # Set the random seeds for this process--if you don't do this, all processes
    # will generate the same data
    np.random.seed()
    random.seed()
    env, results = gen_single_env_data()

    n = len(results)
    cuboids = env.cuboids
    cylinders = env.cylinders
    file_name = f"{TMP_DATA_DIR}/{uuid.uuid4()}.hdf5"
    with h5py.File(file_name, "w-") as f:
        global_solutions = f.create_dataset("global_solutions", (n, SEQUENCE_LENGTH, 7))
        cuboid_dims = f.create_dataset("cuboid_dims", (len(cuboids), 3))
        cuboid_centers = f.create_dataset("cuboid_centers", (len(cuboids), 3))
        cuboid_quats = f.create_dataset("cuboid_quaternions", (len(cuboids), 4))

        cylinder_radii = f.create_dataset("cylinder_radii", (len(cylinders), 1))
        cylinder_heights = f.create_dataset("cylinder_heights", (len(cylinders), 1))
        cylinder_centers = f.create_dataset("cylinder_centers", (len(cylinders), 3))
        cylinder_quats = f.create_dataset("cylinder_quaternions", (len(cylinders), 4))

        for ii in range(n):
            global_solutions[ii, :, :] = results[ii].global_solution
        for jj in range(len(cuboids)):
            cuboid_dims[jj, :] = cuboids[jj].dims
            cuboid_centers[jj, :] = cuboids[jj].pose.xyz
            cuboid_quats[jj, :] = cuboids[jj].pose.so3.wxyz
        for kk in range(len(cylinders)):
            cylinder_radii[kk, :] = cylinders[kk].radius
            cylinder_heights[kk, :] = cylinders[kk].height
            cylinder_centers[kk, :] = cylinders[kk].pose.xyz
            cylinder_quats[kk, :] = cylinders[kk].pose.so3.wxyz


def gen():
    """
    This is the multiprocess workhorse. It launches a ton of parallel subprocesses
    that each generate a bunch of trajectories across a single environment.
    Then, it merges everything into a single file.
    """
    noOutputHandler()
    non_seeds = np.arange(NUM_SCENES)  # Needed for imap_unordered

    with Pool() as pool:
        for _ in tqdm(
            pool.imap_unordered(gen_single_env, non_seeds),
            total=NUM_SCENES,
        ):
            pass

    all_files = list(Path(TMP_DATA_DIR).glob("*.hdf5"))
    # Merge all the files generated by the subprocesses into a large hdf5 file
    max_cylinders = 0
    max_cuboids = 0
    total_trajectories = 0
    for fi in all_files:
        with h5py.File(fi) as f:
            total_trajectories += len(f["global_solutions"])
            num_cuboids = len(f["cuboid_dims"])
            num_cylinders = len(f["cylinder_radii"])
            if num_cuboids > max_cuboids:
                max_cuboids = num_cuboids
            if num_cylinders > max_cylinders:
                max_cylinders = num_cylinders

    with h5py.File(f"{FINAL_DATA_DIR}/all_data.hdf5", "w-") as f:
        global_solutions = f.create_dataset(
            "global_solutions", (total_trajectories, SEQUENCE_LENGTH, 7)
        )
        cuboid_dims = f.create_dataset(
            "cuboid_dims", (total_trajectories, max_cuboids, 3)
        )
        cuboid_centers = f.create_dataset(
            "cuboid_centers", (total_trajectories, max_cuboids, 3)
        )
        cuboid_quats = f.create_dataset(
            "cuboid_quaternions", (total_trajectories, max_cuboids, 4)
        )

        cylinder_radii = f.create_dataset(
            "cylinder_radii", (total_trajectories, max_cylinders, 1)
        )
        cylinder_heights = f.create_dataset(
            "cylinder_heights", (total_trajectories, max_cylinders, 1)
        )
        cylinder_centers = f.create_dataset(
            "cylinder_centers", (total_trajectories, max_cylinders, 3)
        )
        cylinder_quats = f.create_dataset(
            "cylinder_quaternions", (total_trajectories, max_cylinders, 4)
        )

        chunk_start = 0
        chunk_end = 0
        for fi in all_files:
            with h5py.File(fi, "r") as g:
                chunk_end += len(g["global_solutions"])
                global_solutions[chunk_start:chunk_end, ...] = g["global_solutions"][
                    ...
                ]

                num_cuboids = len(g["cuboid_dims"])
                num_cylinders = len(g["cylinder_radii"])
                for idx in range(chunk_start, chunk_end):
                    cuboid_dims[idx, :num_cuboids, ...] = g["cuboid_dims"][...]
                    cuboid_centers[idx, :num_cuboids, ...] = g["cuboid_centers"][...]
                    cuboid_quats[idx, :num_cuboids, ...] = g["cuboid_quaternions"][...]

                    cylinder_radii[idx, :num_cylinders, ...] = g["cylinder_radii"][...]
                    cylinder_heights[idx, :num_cylinders, ...] = g["cylinder_heights"][
                        ...
                    ]
                    cylinder_centers[idx, :num_cylinders, ...] = g["cylinder_centers"][
                        ...
                    ]
                    cylinder_quats[idx, :num_cylinders, ...] = g[
                        "cylinder_quaternions"
                    ][...]
            chunk_start = chunk_end
    for fi in all_files:
        fi.unlink()


def visualize_single_env():
    env, results = gen_single_env_data()
    if len(results) == 0:
        print("Found no results")
        return
    sim = Bullet(gui=True)
    robot = sim.load_robot(FrankaRobot)
    sim.load_primitives(env.obstacles)
    for r in results:
        print("Visualizing global solution")
        for q in r.global_solution:
            robot.marionette(q)
            time.sleep(0.1)
        time.sleep(0.2)


def generate_free_space_inference_data(
    expert_pipeline: str, how_many: int, save_path: str
):
    selfcc = FrankaSelfCollisionChecker()
    inference_problems = []
    with tqdm(total=how_many) as pbar:
        while len(inference_problems) < how_many:
            env, all_results = gen_single_env_data()
            if len(all_results) == 0:
                continue
            if expert_pipeline == "global":
                results = all_results
            else:
                raise NotImplementedError(
                    "Task oriented inference data generation only supports global"
                )
            for result in results:
                if expert_pipeline == "both":
                    plan, _ = solve_global_plan(
                        result.start_candidate,
                        result.target_candidate,
                        result.cuboids + result.cylinders,
                        selfcc,
                    )
                    if len(plan) == 0:
                        continue

                if hasattr(result.target_candidate, "support_volume"):
                    target_volume = result.target_candidate.support_volume
                else:
                    target_volume = Cuboid(
                        center=result.target_candidate.pose.xyz,
                        dims=[0.05, 0.05, 0.05],
                        quaternion=result.target_candidate.pose.so3.wxyz,
                    )
                inference_problems.append(
                    PlanningProblem(
                        target=result.target_candidate.pose,
                        q0=result.start_candidate.config,
                        obstacles=result.cuboids + result.cylinders,
                        target_volume=target_volume,
                        target_negative_volumes=result.target_candidate.negative_volumes,
                    )
                )
                pbar.update(1)

    with open(save_path, "wb") as f:
        pickle.dump({ENV_TYPE: {"free_space": inference_problems}}, f)
    
def generate_task_oriented_inference_data(
    expert_pipeline: str, how_many: int, save_path: str
):
    selfcc = FrankaSelfCollisionChecker()
    inference_problems = []
    with tqdm(total=how_many) as pbar:
        while len(inference_problems) < how_many:
            env, all_results = gen_single_env_data()
            if len(all_results) == 0:
                continue
            if expert_pipeline == "global":
                results = all_results
            else:
                raise NotImplementedError(
                    "Task oriented inference data generation only supports global"
                )
            for result in results:
                if expert_pipeline == "both":
                    plan, _ = solve_global_plan(
                        result.start_candidate,
                        result.target_candidate,
                        result.cuboids + result.cylinders,
                        selfcc,
                    )
                    if len(plan) == 0:
                        continue

                if hasattr(result.target_candidate, "support_volume"):
                    target_volume = result.target_candidate.support_volume
                else:
                    target_volume = Cuboid(
                        center=result.target_candidate.pose.xyz,
                        dims=[0.05, 0.05, 0.05],
                        quaternion=result.target_candidate.pose.so3.wxyz,
                    )
                inference_problems.append(
                    PlanningProblem(
                        target=result.target_candidate.pose,
                        q0=result.start_candidate.config,
                        obstacles=result.cuboids + result.cylinders,
                        target_volume=target_volume,
                        target_negative_volumes=result.target_candidate.negative_volumes,
                    )
                )
                pbar.update(1)

    with open(save_path, "wb") as f:
        pickle.dump({ENV_TYPE: {"task_oriented": inference_problems}}, f)
        
        
def generate_neutral_inference_data(
    expert_pipeline: str, how_many: int, save_path: str
):
    selfcc = FrankaSelfCollisionChecker()
    assert (
        how_many % 2 == 0
    ), "When generating neutral inference problems, the number of problems must be even"
    to_neutral_problems = []
    from_neutral_problems = []
    with tqdm(total=how_many) as pbar:
        while len(to_neutral_problems) + len(from_neutral_problems) < how_many:
            env, all_results = gen_single_env_data()
            if len(all_results) == 0:
                continue
            if expert_pipeline == "global":
                results = all_results
            else:
                raise NotImplementedError(
                    "Task oriented inference data generation only supports global"
                )
            for result in results:
                if (
                    len(to_neutral_problems) < how_many // 2
                    and isinstance(result.start_candidate, TaskOrientedCandidate)
                    and isinstance(result.target_candidate, NeutralCandidate)
                ):
                    problem_list = to_neutral_problems
                elif (
                    len(from_neutral_problems) < how_many // 2
                    and isinstance(result.start_candidate, NeutralCandidate)
                    and isinstance(result.target_candidate, TaskOrientedCandidate)
                ):
                    problem_list = from_neutral_problems
                else:
                    continue

                if expert_pipeline == "both":
                    plan, _ = solve_global_plan(
                        result.start_candidate,
                        result.target_candidate,
                        result.cuboids + result.cylinders,
                        selfcc,
                    )
                    if len(plan) == 0:
                        continue

                if hasattr(result.target_candidate, "support_volume"):
                    target_volume = result.target_candidate.support_volume
                else:
                    target_volume = Cuboid(
                        center=result.target_candidate.pose.xyz,
                        dims=[0.05, 0.05, 0.05],
                        quaternion=result.target_candidate.pose.so3.wxyz,
                    )
                problem_list.append(
                    PlanningProblem(
                        target=result.target_candidate.pose,
                        q0=result.start_candidate.config,
                        obstacles=result.cuboids + result.cylinders,
                        target_volume=target_volume,
                        target_negative_volumes=result.target_candidate.negative_volumes,
                    )
                )
                pbar.update(1)
                break
    with open(save_path, "wb") as f:
        pickle.dump(
            {
                ENV_TYPE: {
                    "neutral_start": from_neutral_problems,
                    "neutral_goal": to_neutral_problems,
                }
            },
            f,
        )


def generate_mixed_inference_data(
    expert_pipeline: str, how_many: int, save_path: str
):
    selfcc = FrankaSelfCollisionChecker()
    assert (
        how_many % 2 == 0
    ), "When generating mixed inference problems, the number of problems must be even"
    to_free_problems = []
    from_free_problems = []
    with tqdm(total=how_many) as pbar:
        while len(to_free_problems) + len(from_free_problems) < how_many:
            env, all_results = gen_single_env_data()
            if len(all_results) == 0:
                continue
            if expert_pipeline == "global":
                results = all_results
            else:
                raise NotImplementedError(
                    "Task oriented inference data generation only supports global"
                )
            for result in results:
                if (
                    len(to_free_problems) < how_many // 2
                    and isinstance(result.start_candidate, TaskOrientedCandidate)
                    and isinstance(result.target_candidate, FreeSpaceCandidate)
                ):
                    problem_list = to_free_problems
                elif (
                    len(from_free_problems) < how_many // 2
                    and isinstance(result.start_candidate, FreeSpaceCandidate)
                    and isinstance(result.target_candidate, TaskOrientedCandidate)
                ):
                    problem_list = from_free_problems
                else:
                    continue

                if expert_pipeline == "both":
                    plan, _ = solve_global_plan(
                        result.start_candidate,
                        result.target_candidate,
                        result.cuboids + result.cylinders,
                        selfcc,
                    )
                    if len(plan) == 0:
                        continue

                if hasattr(result.target_candidate, "support_volume"):
                    target_volume = result.target_candidate.support_volume
                else:
                    target_volume = Cuboid(
                        center=result.target_candidate.pose.xyz,
                        dims=[0.05, 0.05, 0.05],
                        quaternion=result.target_candidate.pose.so3.wxyz,
                    )
                problem_list.append(
                    PlanningProblem(
                        target=result.target_candidate.pose,
                        q0=result.start_candidate.config,
                        obstacles=result.cuboids + result.cylinders,
                        target_volume=target_volume,
                        target_negative_volumes=result.target_candidate.negative_volumes,
                    )
                )
                pbar.update(1)
                break
    with open(save_path, "wb") as f:
        pickle.dump(
            {
                ENV_TYPE: {
                    "free_start": from_free_problems,
                    "free_goal": to_free_problems,
                }
            },
            f,
        )


def generate_inference_data(expert_pipeline: str, how_many: int, save_path: str):
    """
    Generates a file with inference data for a given scene type. If the problems are mixed
    (as specified by the global variable), then this will generate an even number of problems.

    Note that the sub-methods here are not the most efficient. For example, they generate two paths
    for every environment instead of 1. This is for code-simplicity as performance does not matter
    nearly as much for these examples. To generate a lot of examples, it would be best to run this
    in a multiprocessing setting for performance (just as we do for the full data generation)

    :param expert_pipeline str: The pipeline for which this problem is solvable (can be expert, global, or both)
    :param how_many int: How many examples to generate
    :param save_path str: The path to which the problem set should be saved
    """
    assert (
        not Path(save_path).resolve().exists()
    ), "Cannot save inference data to a file that already exists"

    if prob_type == "mixed":
        generate_mixed_inference_data(expert_pipeline, how_many, save_path)
    elif prob_type == "task-oriented":
        generate_task_oriented_inference_data(expert_pipeline, how_many, save_path)
    elif prob_type == "free-space":
        generate_free_space_inference_data(expert_pipeline, how_many, save_path)
    elif prob_type == "neutral":
        generate_neutral_inference_data(expert_pipeline, how_many, save_path)
    else:
        raise NotImplementedError(
            f"Prob type {prob_type} not implemented for inference data generation"
        )
    


if __name__ == "__main__":
    """
    This program makes heavy use of global variables. This is **not** best practice,
    but helps immensely with constant variables that need to be set for Python multiprocessing
    """
    # This start time is used globally to tell the program to shut down after a
    # configured timeout
    global START_TIME
    START_TIME = time.time()

    np.random.seed()
    random.seed()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "env_type",
        choices=[
            "free",
            "tabletop",
            "cubby",
            "merged-cubby",
            "dresser",
        ],
        help="Include this argument if there are subtypes",
    )

    parser.add_argument(
        "prob_type",
        choices=[
            "mixed",
            "task-oriented",
            "free-space",
            "neutral",
        ],
        help="Include this argument if there are subtypes",
    )
        
    subparsers = parser.add_subparsers(
        help="Whether to run the full pipeline, the test pipeline, or an environment test",
        dest="run_type",
    )
    run_full = subparsers.add_parser(
        "full-pipeline",
        help=(
            "Run full pipeline with multiprocessing. Specific configuration (job size,"
            " timeouts, etc) are hardcoded at the top of the file."
        ),
    )
    run_full.add_argument(
        "data_dir",
        type=str,
        help="An existing _empty_ directory where the output data will be saved",
    )

    test_pipeline = subparsers.add_parser(
        "test-pipeline",
        help=(
            "Runs a miniature version of the full pipeline. Specific configuration (job size,"
            " timeouts, etc) are hardcoded at the top of the file."
        ),
    )
    test_pipeline.add_argument(
        "data_dir",
        type=str,
        help="An existing _empty_ directory where the output data will be saved",
    )

    test_pipeline = subparsers.add_parser(
        "test-environment",
        help="Generates a few trajectories for a single environment and visualizes them with Pybullet",
    )

    gen_inference = subparsers.add_parser(
        "for-inference",
        help="Generates data that be be used to run inference on the model",
    )

    gen_inference.add_argument(
        "expert",
        choices=["global"],
        help="Which expert pipeline to use when generating the data",
    )

    gen_inference.add_argument(
        "how_many",
        type=int,
        help="How many problems to generate (1 problem, 1 environment). If mixed flag is specified, this number must be even.",
    )
    gen_inference.add_argument(
        "save_path",
        type=str,
        help="The output file to which the inference problems should be saved. Should be a .pkl file",
    )

    args = parser.parse_args()

    # Used to tell all the various subprocesses whether to use free space poses
    global prob_type
    prob_type = args.prob_type

    # Sets the env type
    global ENV_TYPE
    ENV_TYPE = args.env_type

    if args.run_type in ["test-pipeline", "test-environment"]:
        NUM_SCENES = 10  # The maximum number of scenes to generate in a single job
        NUM_PLANS_PER_SCENE = (
            4  # The number of total candidate start or goals to use to plan experts
        )
    elif args.run_type == "for-inference":
        NUM_PLANS_PER_SCENE = 2

    if args.run_type == "test-environment":
        visualize_single_env()
    elif args.run_type == "for-inference":
        generate_inference_data(args.expert, args.how_many, args.save_path)
    else:
        # A temporary directory where the per-scene data will be saved
        global TMP_DATA_DIR
        TMP_DATA_DIR = f"/tmp/tmp_data_{uuid.uuid4()}/"
        assert not Path(TMP_DATA_DIR).exists()
        os.mkdir(TMP_DATA_DIR)
        assert (
            os.path.isdir(TMP_DATA_DIR)
            and len(os.listdir(TMP_DATA_DIR)) == 0
            and os.access(TMP_DATA_DIR, os.W_OK)
        )

        # The directory where the final data will be saved--checks whether it's writeable and empty
        global FINAL_DATA_DIR
        FINAL_DATA_DIR = args.data_dir
        assert (
            os.path.isdir(FINAL_DATA_DIR)
            and len(os.listdir(FINAL_DATA_DIR)) == 0
            and os.access(FINAL_DATA_DIR, os.W_OK)
        )

        print(f"Final data with save to {FINAL_DATA_DIR}")
        print(f"Temporary data will save to {TMP_DATA_DIR}")
        print("Using args:")
        print(f"    (env_type: {args.env_type})")
        print(f"    (prob_type: {args.prob_type})")
        gen()
