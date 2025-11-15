# cubby_environment.py
from dataclasses import dataclass
import random
from typing import List, Optional, Tuple

from geometrout.primitive import Cuboid, Cylinder
from geometrout.transform import SE3
from robofin.bullet import Bullet, BulletFranka, BulletFrankaGripper
from robofin.robots import FrankaGripper, FrankaRobot, FrankaRealRobot
from robofin.collision import FrankaSelfCollisionChecker
import numpy as np
from copy import deepcopy

from pyquaternion import Quaternion

from data_pipeline.environments.base_environment import (
    TaskOrientedCandidate,
    NeutralCandidate,
    FreeSpaceCandidate,
    Environment,
    radius_sample,
    Tool,
    SO3
)


@dataclass
class CubbyCandidate(TaskOrientedCandidate):
    """
    Represents a configuration, its end-effector pose (in right_gripper frame), and
    some metadata about the cubby (i.e. which cubby pocket it belongs to and the free space
    inside that cubby)
    """

    pocket_idx: int = 0
    support_volume: Cuboid = None


class Cubby:
    """
    The actual cubby construction itself, without any robot info.
    """

    def __init__(self):
        self.cubby_left = radius_sample(0.7, 0.1)
        self.cubby_right = radius_sample(-0.7, 0.1)
        self.cubby_bottom = radius_sample(0.1, 0.1)
        self.cubby_front = radius_sample(0.55, 0.1)
        self.cubby_back = self.cubby_front + radius_sample(0.4, 0.2)
        self.cubby_top = radius_sample(0.8, 0.1)
        self.cubby_mid_h_z = radius_sample(0.45, 0.2)
        self.cubby_mid_v_y = radius_sample(0.0, 0.1)
        self.thickness = radius_sample(0.02, 0.01)
        self.middle_shelf_thickness = self.thickness
        self.center_wall_thickness = self.thickness
        self.in_cabinet_rotation = radius_sample(0, np.pi / 18)
          
    @property
    def rotation_matrix(self) -> np.ndarray:
        """
        Cubbies are essentially represented as unrotated boxes that are then rotated around
        their central yaw axis by `self.in_cabinet_rotation`. This function produces the
        rotation matrix corresponding to that value and axis.

        :rtype np.ndarray: The rotation matrix
        """
        cabinet_T_world = np.array(
            [
                [1, 0, 0, -(self.cubby_front + self.cubby_back) / 2],
                [0, 1, 0, -(self.cubby_left + self.cubby_right) / 2],
                [0, 0, 1, -(self.cubby_top + self.cubby_bottom) / 2],
                [0, 0, 0, 1],
            ]
        )
        in_cabinet_rotation = np.array(
            [
                [
                    np.cos(self.in_cabinet_rotation),
                    -np.sin(self.in_cabinet_rotation),
                    0,
                    0,
                ],
                [
                    np.sin(self.in_cabinet_rotation),
                    np.cos(self.in_cabinet_rotation),
                    0,
                    0,
                ],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        world_T_cabinet = np.array(
            [
                [1, 0, 0, (self.cubby_front + self.cubby_back) / 2],
                [0, 1, 0, (self.cubby_left + self.cubby_right) / 2],
                [0, 0, 1, (self.cubby_top + self.cubby_bottom) / 2],
                [0, 0, 0, 1],
            ]
        )
        pivot = np.matmul(
            world_T_cabinet, np.matmul(in_cabinet_rotation, cabinet_T_world)
        )
        return pivot

    def _unrotated_cuboids(self) -> List[Cuboid]:
        """
        Returns the unrotated cuboids that must then be rotated to produce the final cubby.

        :rtype List[Cuboid]: All the cuboids in the cubby
        """
        cuboids = [
            # Back Wall
            Cuboid(
                center=[
                    self.cubby_back,
                    (self.cubby_left + self.cubby_right) / 2,
                    self.cubby_top / 2,
                ],
                dims=[
                    self.thickness,
                    (self.cubby_left - self.cubby_right),
                    self.cubby_top,
                ],
                quaternion=[1, 0, 0, 0],
            ),
            # Bottom Shelf
            Cuboid(
                center=[
                    (self.cubby_front + self.cubby_back) / 2,
                    (self.cubby_left + self.cubby_right) / 2,
                    self.cubby_bottom,
                ],
                dims=[
                    self.cubby_back - self.cubby_front,
                    self.cubby_left - self.cubby_right,
                    self.thickness,
                ],
                quaternion=[1, 0, 0, 0],
            ),
            # Top Shelf
            Cuboid(
                center=[
                    (self.cubby_front + self.cubby_back) / 2,
                    (self.cubby_left + self.cubby_right) / 2,
                    self.cubby_top,
                ],
                dims=[
                    self.cubby_back - self.cubby_front,
                    self.cubby_left - self.cubby_right,
                    self.thickness,
                ],
                quaternion=[1, 0, 0, 0],
            ),
            # Right Wall
            Cuboid(
                center=[
                    (self.cubby_front + self.cubby_back) / 2,
                    self.cubby_right,
                    (self.cubby_top + self.cubby_bottom) / 2,
                ],
                dims=[
                    self.cubby_back - self.cubby_front,
                    self.thickness,
                    (self.cubby_top - self.cubby_bottom) + self.thickness,
                ],
                quaternion=[1, 0, 0, 0],
            ),
            # Left Wall
            Cuboid(
                center=[
                    (self.cubby_front + self.cubby_back) / 2,
                    self.cubby_left,
                    (self.cubby_top + self.cubby_bottom) / 2,
                ],
                dims=[
                    self.cubby_back - self.cubby_front,
                    self.thickness,
                    (self.cubby_top - self.cubby_bottom) + self.thickness,
                ],
                quaternion=[1, 0, 0, 0],
            ),
        ]
        if not np.isclose(self.cubby_mid_v_y, 0.0):
            # Center Wall (vertical)
            cuboids.append(
                Cuboid(
                    center=[
                        (self.cubby_front + self.cubby_back) / 2,
                        self.cubby_mid_v_y,
                        (self.cubby_top + self.cubby_bottom) / 2,
                    ],
                    dims=[
                        self.cubby_back - self.cubby_front,
                        self.center_wall_thickness,
                        self.cubby_top - self.cubby_bottom + self.thickness,
                    ],
                    quaternion=[1, 0, 0, 0],
                )
            )
        if not np.isclose(self.cubby_mid_h_z, 0.0):
            # Middle Shelf
            cuboids.append(
                Cuboid(
                    center=[
                        (self.cubby_front + self.cubby_back) / 2,
                        (self.cubby_left + self.cubby_right) / 2,
                        self.cubby_mid_h_z,
                    ],
                    dims=[
                        self.cubby_back - self.cubby_front,
                        self.cubby_left - self.cubby_right,
                        self.middle_shelf_thickness,
                    ],
                    quaternion=[1, 0, 0, 0],
                )
            )
        return cuboids

    @property
    def cuboids(self) -> List[Cuboid]:
        """
        Returns the cuboids that make up the cubby

        :rtype List[Cuboid]: The cuboids that make up each section of the cubby
        """
        cuboids: List[Cuboid] = []
        for cuboid in self._unrotated_cuboids():
            center = cuboid.center
            new_matrix = np.matmul(
                self.rotation_matrix,
                np.array(
                    [
                        [1, 0, 0, center[0]],
                        [0, 1, 0, center[1]],
                        [0, 0, 1, center[2]],
                        [0, 0, 0, 1],
                    ]
                ),
            )
            center = new_matrix[:3, 3]
            quat = Quaternion(matrix=new_matrix)
            cuboids.append(
                Cuboid(center, cuboid.dims, quaternion=[quat.w, quat.x, quat.y, quat.z])
            )
        return cuboids

    @property
    def support_volumes(self) -> List[Cuboid]:
        """
        Returns the support volumes inside each of the cubby pockets.
        These support volumes could be tighter to make for more efficient environment
        queries, but right now they include half of the surrounding shelves.

        :rtype List[Cuboid]: The list of support volumes
        """
        if np.isclose(self.center_wall_thickness, 0) and np.isclose(
            self.middle_shelf_thickness, 0
        ):
            centers = [
                np.array(
                    [
                        self.cubby_front + self.cubby_back,
                        self.cubby_left + self.cubby_right,
                        self.cubby_top + self.cubby_bottom,
                    ]
                )
                / 2
            ]
            dims = [
                np.array(
                    [
                        self.cubby_back - self.cubby_front,
                        self.cubby_left - self.cubby_right,
                        self.cubby_top - self.cubby_bottom,
                    ]
                )
            ]
        elif np.isclose(self.center_wall_thickness, 0):
            centers = [
                np.array(
                    [
                        (self.cubby_front + self.cubby_back) / 2,
                        (self.cubby_left + self.cubby_right) / 2,
                        (self.cubby_top + self.cubby_mid_h_z) / 2,
                    ]
                ),
                np.array(
                    [
                        (self.cubby_front + self.cubby_back) / 2,
                        (self.cubby_left + self.cubby_right) / 2,
                        (self.cubby_bottom + self.cubby_mid_h_z) / 2,
                    ]
                ),
            ]
            dims = [
                np.array(
                    [
                        (self.cubby_back - self.cubby_front),
                        (self.cubby_left - self.cubby_right),
                        (self.cubby_top - self.cubby_mid_h_z),
                    ]
                ),
                np.array(
                    [
                        (self.cubby_back - self.cubby_front),
                        (self.cubby_left - self.cubby_right),
                        (self.cubby_mid_h_z - self.cubby_bottom),
                    ]
                ),
            ]
        elif np.isclose(self.middle_shelf_thickness, 0):
            centers = [
                np.array(
                    [
                        (self.cubby_front + self.cubby_back) / 2,
                        (self.cubby_left + self.cubby_mid_v_y) / 2,
                        (self.cubby_top + self.cubby_bottom) / 2,
                    ]
                ),
                np.array(
                    [
                        (self.cubby_front + self.cubby_back) / 2,
                        (self.cubby_mid_v_y + self.cubby_right) / 2,
                        (self.cubby_top + self.cubby_bottom) / 2,
                    ]
                ),
            ]
            dims = [
                np.array(
                    [
                        (self.cubby_back - self.cubby_front),
                        (self.cubby_left - self.cubby_mid_v_y),
                        (self.cubby_top - self.cubby_bottom),
                    ]
                ),
                np.array(
                    [
                        (self.cubby_back - self.cubby_front),
                        (self.cubby_mid_v_y - self.cubby_right),
                        (self.cubby_top - self.cubby_bottom),
                    ]
                ),
            ]
        else:
            centers = [
                np.array(
                    [
                        (self.cubby_front + self.cubby_back) / 2,
                        (self.cubby_left + self.cubby_mid_v_y) / 2,
                        (self.cubby_top + self.cubby_mid_h_z) / 2,
                    ]
                ),
                np.array(
                    [
                        (self.cubby_front + self.cubby_back) / 2,
                        (self.cubby_mid_v_y + self.cubby_right) / 2,
                        (self.cubby_top + self.cubby_mid_h_z) / 2,
                    ]
                ),
                np.array(
                    [
                        (self.cubby_front + self.cubby_back) / 2,
                        (self.cubby_left + self.cubby_mid_v_y) / 2,
                        (self.cubby_bottom + self.cubby_mid_h_z) / 2,
                    ]
                ),
                np.array(
                    [
                        (self.cubby_front + self.cubby_back) / 2,
                        (self.cubby_mid_v_y + self.cubby_right) / 2,
                        (self.cubby_bottom + self.cubby_mid_h_z) / 2,
                    ]
                ),
            ]
            dims = [
                np.array(
                    [
                        (self.cubby_back - self.cubby_front),
                        (self.cubby_left - self.cubby_mid_v_y),
                        (self.cubby_top - self.cubby_mid_h_z),
                    ]
                ),
                np.array(
                    [
                        (self.cubby_back - self.cubby_front),
                        (self.cubby_mid_v_y - self.cubby_right),
                        (self.cubby_top - self.cubby_mid_h_z),
                    ]
                ),
                np.array(
                    [
                        (self.cubby_back - self.cubby_front),
                        (self.cubby_left - self.cubby_mid_v_y),
                        (self.cubby_mid_h_z - self.cubby_bottom),
                    ]
                ),
                np.array(
                    [
                        (self.cubby_back - self.cubby_front),
                        (self.cubby_mid_v_y - self.cubby_right),
                        (self.cubby_mid_h_z - self.cubby_bottom),
                    ]
                ),
            ]

        volumes = []
        for c, d in zip(centers, dims):
            unrotated_pose = np.eye(4)
            unrotated_pose[:3, 3] = c
            pose = SE3(np.matmul(self.rotation_matrix, unrotated_pose))
            volumes.append(Cuboid(center=pose.xyz, dims=d, quaternion=pose.so3.wxyz))
        return volumes


class CubbyEnvironment(Environment):
    def __init__(self):
        super().__init__()
        self.demo_candidates = []
        self._tools = None

    def _gen(self, selfcc: FrankaSelfCollisionChecker) -> bool:
        """
        Generates an environment and a pair of start/end candidates

        :param selfcc FrankaSelfCollisionChecker: Checks for self collisions using spheres that
                                                  mimic the internal Franka collision checker.
        :rtype bool: Whether the environment was successfully generated
        """
        self.cubby = Cubby()
        self.generate_tool()
        support_idxs = np.arange(len(self.cubby.support_volumes))
        random.shuffle(support_idxs)
        supports = self.cubby.support_volumes

        sim = Bullet(gui=False)
        sim.load_primitives(self.cubby.cuboids)
        gripper = sim.load_robot(FrankaGripper)
        arm = sim.load_robot(FrankaRobot)

        (
            start_pose,
            start_q,
            start_support_volume,
            target_pose,
            target_q,
            target_support_volume,
        ) = (None, None, None, None, None, None)

        for ii, idx in enumerate(support_idxs):
            start_support_volume = supports[idx]
            start_pose, start_q = self.random_pose_and_config(
                sim, gripper, arm, selfcc, start_support_volume
            )
            if start_pose is None or start_q is None:
                continue
            for jdx in support_idxs[ii + 1 :]:
                target_support_volume = supports[jdx]
                target_pose, target_q = self.random_pose_and_config(
                    sim, gripper, arm, selfcc, target_support_volume
                )
                if target_pose is not None and target_q is not None:
                    break
            if target_pose is not None and target_q is not None:
                break

        if start_q is None or target_q is None:
            self.demo_candidates = None
            return False
        self.demo_candidates = (
            CubbyCandidate(
                pose=start_pose,
                config=start_q,
                pocket_idx=idx,
                support_volume=start_support_volume,
                negative_volumes=[s for ii, s in enumerate(supports) if ii != idx],
                tool=self._tools,
            ),
            CubbyCandidate(
                pose=target_pose,
                config=target_q,
                pocket_idx=jdx,
                support_volume=target_support_volume,
                negative_volumes=[s for ii, s in enumerate(supports) if ii != jdx],
                tool=self._tools,
            ),
        )
        return True

    def random_pose_and_config(
        self,
        sim: Bullet,
        gripper: BulletFrankaGripper,
        arm: BulletFranka,
        selfcc: FrankaSelfCollisionChecker,
        support_volume: Cuboid,
    ) -> Tuple[Optional[SE3], Optional[np.ndarray]]:
        """
        Creates a random end effector pose in the desired support volume and solves for
        collision free IK

        :param sim Bullet: A simulator already loaded with the obstacles
        :param gripper BulletFrankaGripper: The simulated gripper (necessary for collision
                                            checking for the pose)
        :param arm BulletFranka: The simulated arm (necessary for collision checking during
                                 collision free IK)
        :param selfcc FrankaSelfCollisionChecker: Checks for self collisions using spheres that
                                                  mimic the internal Franka collision checker.
                                                  Necessary for collision free IK
        :param support_volume Cuboid: The desired support volume to search inside
        :rtype Tuple[Optional[SE3], Optional[np.ndarray]]: Returns a valid pose and IK solution if
                                                          one is found. Otherwise, returns None, None
        """
        samples = support_volume.sample_volume(100)
        pose, q = None, None
        for sample in samples:
            theta = radius_sample(0, np.pi / 4)
            z = np.array([np.cos(theta), np.sin(theta), 0])
            x = np.array([0, 0, -1])
            y = np.cross(z, x)
            pose = SE3.from_unit_axes(
                origin=sample,
                x=x,
                y=y,
                z=z,
            )
            gripper.marionette(pose)
            
            # Check tool collision
            if self._check_tool_collision(pose, self.obstacles, self._tools):
                pose = None
                continue
                
            if sim.in_collision(gripper):
                pose = None
                continue
            q = FrankaRealRobot.collision_free_ik(sim, arm, selfcc, pose, retries=1000)
            if q is not None:
                break
        return pose, q

    def _gen_free_space_candidates(
        self, how_many: int, selfcc: FrankaSelfCollisionChecker
    ) -> List[FreeSpaceCandidate]:
        """
        Generates a set of free space candidates by first sampling poses in a specified range
        and then solving for collision-free IK solutions.

        :param how_many int: How many to generate ideally--can be less if there are a lot of failures
        :param selfcc FrankaSelfCollisionChecker: Checks for self collisions using spheres that
                                                mimic the internal Franka collision checker.
        :rtype List[FreeSpaceCandidate]: A list of free space candidates
        """
        sim = Bullet(gui=False)
        gripper = sim.load_robot(FrankaGripper)
        arm = sim.load_robot(FrankaRobot)
        sim.load_primitives(self.obstacles)
        candidates: List[FreeSpaceCandidate] = []

        # Define the sampling ranges for pose generation (similar to free_environment)
        position_ranges = {
            "x": (0.2, 1),  # meters from robot base
            "y": (-0.8, 0.8),  # meters from robot base
            "z": (-0.1, 1),  # meters from floor
        }
        orientation_ranges = {
            "roll": (-np.pi, np.pi),
            "pitch": (-np.pi / 2, np.pi / 2),
            "yaw": (-np.pi, np.pi),
        }

        while len(candidates) < how_many:
            # Generate random pose within specified ranges
            x = np.random.uniform(*position_ranges["x"])
            y = np.random.uniform(*position_ranges["y"])
            z = np.random.uniform(*position_ranges["z"])
            position = np.array([x, y, z])

            roll = np.random.uniform(*orientation_ranges["roll"])
            pitch = np.random.uniform(*orientation_ranges["pitch"])
            yaw = np.random.uniform(*orientation_ranges["yaw"])

            pose = SE3(xyz=position, rpy=[roll, pitch, yaw])

            # Check gripper collision
            gripper.marionette(pose)
            if sim.in_collision(gripper):
                continue

            # Check tool collision
            if self._check_tool_collision(pose, self.obstacles, self._tools):
                continue

            # Solve IK
            q = FrankaRealRobot.collision_free_ik(sim, arm, selfcc, pose, retries=5)
            if q is not None:
                arm.marionette(q)
                if not (
                    sim.in_collision(arm, check_self=True) or selfcc.has_self_collision(q)
                ):
                    candidates.append(
                        FreeSpaceCandidate(
                            config=q,
                            pose=pose,
                            negative_volumes=self.cubby.support_volumes,
                            tool=self._tools,
                        )
                    )
        return candidates

    def _gen_neutral_candidates(
        self, how_many: int, selfcc: FrankaSelfCollisionChecker
    ) -> List[NeutralCandidate]:
        """
        Generates a set of neutral candidates (all collision free)

        :param how_many int: How many to generate ideally--can be less if there are a lot of failures
        :param selfcc FrankaSelfCollisionChecker: Checks for self collisions using spheres that
                                                  mimic the internal Franka collision checker.
        :rtype List[NeutralCandidate]: A list of neutral candidates
        """
        sim = Bullet(gui=False)
        gripper = sim.load_robot(FrankaGripper)
        arm = sim.load_robot(FrankaRobot)
        sim.load_primitives(self.obstacles)
        candidates: List[NeutralCandidate] = []
        for _ in range(how_many * 50):
            if len(candidates) >= how_many:
                break
            sample = FrankaRealRobot.random_neutral(method="uniform")
            arm.marionette(sample)
            if not (
                sim.in_collision(arm, check_self=True)
                or selfcc.has_self_collision(sample)
            ):
                pose = FrankaRealRobot.fk(sample, eff_frame="right_gripper")
                gripper.marionette(pose)
                if not sim.in_collision(gripper):
                    # Check tool collision
                    if not self._check_tool_collision(pose, self.obstacles, self._tools):
                        candidates.append(
                            NeutralCandidate(
                                config=sample,
                                pose=pose,
                                negative_volumes=self.cubby.support_volumes,
                                tool=self._tools,
                            )
                        )
        return candidates

    def _gen_additional_candidate_sets(
        self, how_many: int, selfcc: FrankaSelfCollisionChecker
    ) -> List[List[TaskOrientedCandidate]]:
        """
        Creates additional candidates, where the candidates correspond to the support volumes
        of the environment's generated candidates (created by the `gen` function)

        :param how_many int: How many candidates to generate in each support volume (the result is guaranteed
                             to match this number or the function will run forever)
        :param selfcc FrankaSelfCollisionChecker: Checks for self collisions using spheres that
                                                  mimic the internal Franka collision checker.
        :rtype List[List[TaskOrientedCandidate]]: A pair of candidate sets, where each has `how_many`
                                      candidates that matches the corresponding support volume
                                      for the respective element in `self.demo_candidates`
        """
        candidate_sets = []

        sim = Bullet(gui=False)
        gripper = sim.load_robot(FrankaGripper)
        arm = sim.load_robot(FrankaRobot)
        sim.load_primitives(self.obstacles)

        for idx, candidate in enumerate(self.demo_candidates):
            candidate_set: List[TaskOrientedCandidate] = []
            ii = 0
            while ii < how_many:
                pose, q = self.random_pose_and_config(
                    sim, gripper, arm, selfcc, candidate.support_volume
                )
                if pose is not None and q is not None:
                    candidate_set.append(
                        CubbyCandidate(
                            pose=pose,
                            config=q,
                            pocket_idx=candidate.pocket_idx,
                            support_volume=candidate.support_volume,
                            negative_volumes=candidate.negative_volumes,
                            tool=self._tools,
                        )
                    )
                    ii += 1
            candidate_sets.append(candidate_set)
        return candidate_sets

    def generate_tool(self):
        """
        Generates a complex tool shape with randomized dimensions, offsets, and orientations
        for its constituent primitives to create more variety.
        """
        tool_type = np.random.choice(["T_shape", "L_shape", "U_shape", "bar", "driller", "box"])
        primitives = []

        # Helper to create small random rotations for the entire tool
        def random_tool_rotation_quat():
            rpy = np.random.uniform(-np.pi / 18, np.pi / 18, 3)  # +/- 10 degrees
            return SO3.from_rpy(rpy[0], rpy[1], rpy[2]).wxyz

        # Generate a single random rotation for the entire tool (for composite tools)
        tool_rotation_quat = random_tool_rotation_quat()

        if tool_type == "T_shape":
            stem_dims = [
                np.random.uniform(0.025, 0.035),
                np.random.uniform(0.025, 0.035),
                np.random.uniform(0.08, 0.12),
            ]
            bar_dims = [
                np.random.uniform(0.12, 0.18),
                np.random.uniform(0.025, 0.035),
                np.random.uniform(0.025, 0.035),
            ]
            primitives = [
                { # Vertical stem
                    "dims": stem_dims,
                    "offset": [0, 0, stem_dims[2] / 2],
                    "offset_quaternion": tool_rotation_quat,
                },
                { # Horizontal bar
                    "dims": bar_dims,
                    "offset": [
                        np.random.uniform(-0.01, 0.01),
                        np.random.uniform(-0.01, 0.01),
                        stem_dims[2] + bar_dims[2] / 2 - 0.01,
                    ],
                    "offset_quaternion": tool_rotation_quat,
                },
            ]

        elif tool_type == "L_shape":
            vert_dims = [
                np.random.uniform(0.025, 0.035),
                np.random.uniform(0.025, 0.035),
                np.random.uniform(0.08, 0.12),
            ]
            horiz_dims = [
                np.random.uniform(0.07, 0.1),
                np.random.uniform(0.025, 0.035),
                np.random.uniform(0.025, 0.035),
            ]
            primitives = [
                { # Vertical part
                    "dims": vert_dims,
                    "offset": [0, 0, vert_dims[2] / 2],
                    "offset_quaternion": tool_rotation_quat,
                },
                { # Horizontal extension
                    "dims": horiz_dims,
                    "offset": [
                        horiz_dims[0] / 2 - 0.01,
                        0,
                        vert_dims[2] - horiz_dims[2] / 2,
                    ],
                    "offset_quaternion": tool_rotation_quat,
                },
            ]

        elif tool_type == "U_shape":
            base_dims = [
                np.random.uniform(0.1, 0.2),
                np.random.uniform(0.025, 0.035),
                np.random.uniform(0.025, 0.035),
            ]
            arm_dims = [
                np.random.uniform(0.025, 0.035),
                np.random.uniform(0.025, 0.035),
                np.random.uniform(0.05, 0.1),
            ]
            base_width = base_dims[0]
            primitives = [
                { # Base
                    "dims": base_dims,
                    "offset": [0, 0, 0],
                    "offset_quaternion": tool_rotation_quat,
                },
                { # Left arm
                    "dims": arm_dims,
                    "offset": [
                        -base_width / 2 + arm_dims[0] / 2,
                        0,
                        arm_dims[2] / 2,
                    ],
                    "offset_quaternion": tool_rotation_quat,
                },
                { # Right arm
                    "dims": arm_dims,
                    "offset": [
                        base_width / 2 - arm_dims[0] / 2,
                        0,
                        arm_dims[2] / 2,
                    ],
                    "offset_quaternion": tool_rotation_quat,
                },
            ]
        
        elif tool_type == "bar":
            bar_dims = [
                np.random.uniform(0.15, 0.20),
                np.random.uniform(0.03, 0.04),
                np.random.uniform(0.03, 0.04),
            ]
            primitives = [{
                "dims": bar_dims,
                "offset": [0, 0, bar_dims[1] / 2],
                "offset_quaternion": tool_rotation_quat,
            }]

        elif tool_type == "driller":
            drill_dims = [
                np.random.uniform(0.02, 0.025),
                np.random.uniform(0.02, 0.025),
                np.random.uniform(0.18, 0.25),
            ]
            primitives = [{
                "dims": drill_dims,
                "offset": [0, 0, drill_dims[2] / 2],
                "offset_quaternion": tool_rotation_quat,
            }]
            
        elif tool_type == "box":
            box_dims = [
                np.random.uniform(0.05, 0.2),
                np.random.uniform(0.01, 0.04),
                np.random.uniform(0.05, 0.2),
            ]
            primitives = [{
                "dims": box_dims,
                "offset": [0, 0, box_dims[2] / 2],
                "offset_quaternion": tool_rotation_quat,
            }]

        self._tools = Tool(primitive_type="composite", primitives=primitives)

    @property
    def obstacles(self) -> List[Cuboid]:
        """
        Returns all cuboids in the scene (and the scene is entirely made of cuboids)

        :rtype List[Cuboid]: The list of cuboids in this scene
        """
        return self.cubby.cuboids

    @property
    def cuboids(self) -> List[Cuboid]:
        """
        Returns all cuboids in the scene (and the scene is entirely made of cuboids)

        :rtype List[Cuboid]: The list of cuboids in this scene
        """
        return self.cubby.cuboids

    @property
    def cylinders(self) -> List[Cylinder]:
        """
        Returns an empty list because there are no cylinders in this scene, but left in
        to conform to the standard

        :rtype List[Cuboid]: The list of cuboids in this scene
        """
        return []


class MergedCubbyEnvironment(CubbyEnvironment):
    """
    This class first creates a standard 2x2 cubby and then merges the cubby holes necessary
    to create an unobstructed path between the start and goal.
    """

    def gen(self, selfcc: FrankaSelfCollisionChecker) -> bool:
        """
        Generates an environment and a pair of start/end candidates

        :param selfcc FrankaSelfCollisionChecker: Checks for self collisions using spheres that
                                                  mimic the internal Franka collision checker.
        :rtype bool: Whether the environment was successfully generated
        """
        success = super().gen(selfcc)
        if not success:
            return False
        assert len(self.cubby.support_volumes) == 4
        start, target = self.demo_candidates[0], self.demo_candidates[1]
        if (start.pocket_idx in [0, 1] and target.pocket_idx in [2, 3]) or (
            target.pocket_idx in [0, 1] and start.pocket_idx in [2, 3]
        ):
            self.cubby.middle_shelf_thickness = 0.0
        if (start.pocket_idx in [0, 2] and target.pocket_idx in [1, 3]) or (
            target.pocket_idx in [0, 2] and start.pocket_idx in [1, 3]
        ):
            self.cubby.center_wall_thickness = 0.0
        new_supports = self.cubby.support_volumes
        start_support_idx, target_support_idx = -1, -1
        for ii, s in enumerate(new_supports):
            if s.sdf(start.pose.xyz) < 0:
                start_support_idx = ii
            if s.sdf(target.pose.xyz) < 0:
                target_support_idx = ii
        assert start_support_idx == target_support_idx
        start.support_volume, start.pocket_idx = (
            new_supports[start_support_idx],
            start_support_idx,
        )
        target.support_volume, target.pocket_idx = (
            new_supports[target_support_idx],
            target_support_idx,
        )
        self.demo_candidates = (start, target)
        return success