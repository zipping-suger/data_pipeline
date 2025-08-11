from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import numpy as np
import random

from geometrout.primitive import Cuboid, Cylinder
from geometrout.transform import SE3
from robofin.bullet import Bullet, BulletFranka, BulletFrankaGripper
from robofin.robots import FrankaGripper, FrankaRobot, FrankaRealRobot
from robofin.collision import FrankaSelfCollisionChecker
from pyquaternion import Quaternion

from data_pipeline.environments.base_environment import (
    TaskOrientedCandidate,
    NeutralCandidate,
    FreeSpaceCandidate,
    Environment,
    radius_sample,
)


@dataclass
class CabinetCandidate(TaskOrientedCandidate):
    pocket_idx: int
    support_volume: Cuboid
    negative_volumes: List[Cuboid]


class Cabinet:
    """
    The actual cabinet construction itself, without any robot info.
    """

    def __init__(self):
        self.cabinet_left = radius_sample(0.4, 0.1)
        self.cabinet_right = radius_sample(-0.4, 0.1)
        self.cabinet_bottom = radius_sample(0.2, 0.1)
        self.cabinet_front = radius_sample(0.5, 0.2)
        self.cabinet_back = self.cabinet_front + radius_sample(0.4, 0.2)
        self.cabinet_top = radius_sample(0.8, 0.2)
        self.thickness = radius_sample(0.02, 0.01)
        self.in_cabinet_rotation = radius_sample(0, np.pi / 18)
        self.close_open_angle = radius_sample(np.pi*3/4, np.pi/4)
        self.wide_open_angle = radius_sample(np.pi*1/4, np.pi/4)

        # Randomly assign one door to be wide open and the other close open
        if random.random() < 0.5:
            self.left_open_angle = self.wide_open_angle
            self.right_open_angle = self.close_open_angle
        else:
            self.left_open_angle = self.close_open_angle
            self.right_open_angle = self.wide_open_angle

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
                [1, 0, 0, -(self.cabinet_front + self.cabinet_back) / 2],
                [0, 1, 0, -(self.cabinet_left + self.cabinet_right) / 2],
                [0, 0, 1, -(self.cabinet_top + self.cabinet_bottom) / 2],
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
                [1, 0, 0, (self.cabinet_front + self.cabinet_back) / 2],
                [0, 1, 0, (self.cabinet_left + self.cabinet_right) / 2],
                [0, 0, 1, (self.cabinet_top + self.cabinet_bottom) / 2],
                [0, 0, 0, 1],
            ]
        )
        pivot = np.matmul(
            world_T_cabinet, np.matmul(in_cabinet_rotation, cabinet_T_world)
        )
        return pivot

    def _get_door_cuboid(
        self, is_left_door: bool, angle: float, hinge_x: float, hinge_y: float
    ) -> Cuboid:
        """Helper function to generate a single door cuboid."""
        door_width = (self.cabinet_left - self.cabinet_right) / 2.0
        door_height = self.cabinet_top - self.cabinet_bottom

        # Calculate the center of the door *before* rotation.
        # The center is in the middle of the door's width and height.
        if is_left_door:
            center_x = hinge_x - np.sin(angle) * door_width / 2
            center_y = hinge_y + np.cos(angle) * door_width / 2
            theta = angle  # Left door rotates in the opposite direction
        else:  # Right door
            center_x = hinge_x - np.sin(angle) * door_width / 2
            center_y = hinge_y - np.cos(angle) * door_width / 2
            theta = -angle
        rotated_center = [center_x, center_y, door_height / 2 + self.cabinet_bottom]
        quat = Quaternion(axis=[0, 0, 1], angle=theta)
        return Cuboid(
            center=rotated_center,
            dims=[
                self.thickness,
                door_width,
                door_height,
            ],
            quaternion=[quat.w, quat.x, quat.y, quat.z],
        )

    def _unrotated_cuboids(self) -> List[Cuboid]:
        """
        Returns the unrotated cuboids that must then be rotated to produce the final cabinet.

        :rtype List[Cuboid]: All the cuboids in the cabinet
        """
        cuboids = [
            # Back Wall
            Cuboid(
                center=[
                    self.cabinet_back,
                    (self.cabinet_left + self.cabinet_right) / 2,
                    (self.cabinet_top + self.cabinet_bottom) / 2,
                ],
                dims=[
                    self.thickness,
                    (self.cabinet_left - self.cabinet_right),
                    (self.cabinet_top - self.cabinet_bottom) + self.thickness
                ],
                quaternion=[1, 0, 0, 0],
            ),
            # Right Wall
            Cuboid(
                center=[
                    (self.cabinet_front + self.cabinet_back) / 2,
                    self.cabinet_right,
                    (self.cabinet_top + self.cabinet_bottom) / 2,
                ],
                dims=[
                    self.cabinet_back - self.cabinet_front,
                    self.thickness,
                    (self.cabinet_top - self.cabinet_bottom) + self.thickness,
                ],
                quaternion=[1, 0, 0, 0],
            ),
            # Left Wall
            Cuboid(
                center=[
                    (self.cabinet_front + self.cabinet_back) / 2,
                    self.cabinet_left,
                    (self.cabinet_top + self.cabinet_bottom) / 2,
                ],
                dims=[
                    self.cabinet_back - self.cabinet_front,
                    self.thickness,
                    (self.cabinet_top - self.cabinet_bottom) + self.thickness,
                ],
                quaternion=[1, 0, 0, 0],
            ),
            # Bottom Shelf
            Cuboid(
                center=[
                    (self.cabinet_front + self.cabinet_back) / 2,
                    (self.cabinet_left + self.cabinet_right) / 2,
                    self.cabinet_bottom,
                ],
                dims=[
                    self.cabinet_back - self.cabinet_front,
                    self.cabinet_left - self.cabinet_right,
                    self.thickness,
                ],
                quaternion=[1, 0, 0, 0],
            ),
            # Top Shelf
            Cuboid(
                center=[
                    (self.cabinet_front + self.cabinet_back) / 2,
                    (self.cabinet_left + self.cabinet_right) / 2,
                    self.cabinet_top,
                ],
                dims=[
                    self.cabinet_back - self.cabinet_front,
                    self.cabinet_left - self.cabinet_right,
                    self.thickness,
                ],
                quaternion=[1, 0, 0, 0],
            ),
            # Left Door
            self._get_door_cuboid(
                is_left_door=True,
                angle=self.left_open_angle,
                hinge_x=self.cabinet_front,
                hinge_y=self.cabinet_left,
            ),
            # Right Door
            self._get_door_cuboid(
                is_left_door=False,
                angle=self.right_open_angle,
                hinge_x=self.cabinet_front,
                hinge_y=self.cabinet_right,
            ),
        ]
        return cuboids


    @property
    def cuboids(self) -> List[Cuboid]:
        """
        Returns the cuboids that make up the cabinet

        :rtype List[Cuboid]: The cuboids that make up each section of the cabinet
        """
        cuboids: List[Cuboid] = []
        for cuboid in self._unrotated_cuboids():
            # Build the transformation matrix for the unrotated cuboid
            quat = Quaternion(cuboid.pose.so3.wxyz)
            cuboid_matrix = quat.transformation_matrix
            cuboid_matrix[:3, 3] = cuboid.center

            # Apply the cabinet's rotation
            new_matrix = np.matmul(self.rotation_matrix, cuboid_matrix)

            # Extract new center and quaternion
            center = new_matrix[:3, 3]
            quat_rot = Quaternion(matrix=new_matrix)
            cuboids.append(
                Cuboid(
                    center,
                    cuboid.dims,
                    quaternion=[quat_rot.w, quat_rot.x, quat_rot.y, quat_rot.z],
                )
            )
        return cuboids

    @property
    def support_volumes(self) -> List[Cuboid]:
        # Only the inside volume of the cabinet
        center = np.array(
            [
                (self.cabinet_front + self.cabinet_back) / 2,
                (self.cabinet_left + self.cabinet_right) / 2,
                (self.cabinet_top + self.cabinet_bottom) / 2,
            ]
        )
        dims = np.array(
            [
                self.cabinet_back - self.cabinet_front,
                self.cabinet_left - self.cabinet_right,
                self.cabinet_top - self.cabinet_bottom,
            ]
        )
        # Transform center
        unrotated_pose = np.eye(4)
        unrotated_pose[:3, 3] = center
        pose = SE3(np.matmul(self.rotation_matrix, unrotated_pose))
        support = Cuboid(
            center=pose.xyz,
            dims=dims,
            quaternion=pose.so3.wxyz,
        )
        return [support]


class CabinetEnvironment(Environment):
    def __init__(self):
        super().__init__()
        self.demo_candidates = []
        self.cabinet = None

    def _gen(self, selfcc: FrankaSelfCollisionChecker) -> bool:
        self.cabinet = Cabinet()
        supports = self.cabinet.support_volumes

        sim = Bullet(gui=False)
        sim.load_primitives(self.cabinet.cuboids)
        gripper = sim.load_robot(FrankaGripper)
        arm = sim.load_robot(FrankaRobot)

        start_pose, start_q = self.random_pose_and_config(
            sim, gripper, arm, selfcc, supports[0]
        )
        target_pose, target_q = self.random_pose_and_config(
            sim, gripper, arm, selfcc, supports[0]
        )
        if start_q is None or target_q is None:
            self.demo_candidates = None
            return False
        self.demo_candidates = (
            CabinetCandidate(
                pose=start_pose,
                config=start_q,
                pocket_idx=0,
                support_volume=supports[0],
                negative_volumes=[],
            ),
            CabinetCandidate(
                pose=target_pose,
                config=target_q,
                pocket_idx=0,
                support_volume=supports[0],
                negative_volumes=[],
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
        samples = support_volume.sample_volume(100)
        pose, q = None, None
        for sample in samples:
            theta = radius_sample(0, np.pi / 4) if np.random.rand() < 0.5 else (np.random.rand() * np.pi / 2 - np.pi / 4)
            x = np.array([np.cos(theta), np.sin(theta), 0])
            z = np.array([0, 0, -1])
            y = np.cross(z, x)
            pose = SE3.from_unit_axes(
                origin=sample,
                x=x,
                y=y,
                z=z,
            )
            gripper.marionette(pose)
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
        sim = Bullet(gui=False)
        gripper = sim.load_robot(FrankaGripper)
        arm = sim.load_robot(FrankaRobot)
        sim.load_primitives(self.obstacles)
        candidates: List[FreeSpaceCandidate] = []

        position_ranges = {
            "x": (0.2, 1),
            "y": (-0.8, 0.8),
            "z": (-0.1, 1),
        }
        orientation_ranges = {
            "roll": (-np.pi, np.pi),
            "pitch": (-np.pi / 2, np.pi / 2),
            "yaw": (-np.pi, np.pi),
        }

        while len(candidates) < how_many:
            x = np.random.uniform(*position_ranges["x"])
            y = np.random.uniform(*position_ranges["y"])
            z = np.random.uniform(*position_ranges["z"])
            position = np.array([x, y, z])

            roll = np.random.uniform(*orientation_ranges["roll"])
            pitch = np.random.uniform(*orientation_ranges["pitch"])
            yaw = np.random.uniform(*orientation_ranges["yaw"])

            pose = SE3(xyz=position, rpy=[roll, pitch, yaw])

            gripper.marionette(pose)
            if sim.in_collision(gripper):
                continue

            q = FrankaRealRobot.collision_free_ik(sim, arm, selfcc, pose, retries=5)
            if q is not None:
                arm.marionette(q)
                if not (
                    sim.in_collision(arm, check_self=True)
                    or selfcc.has_self_collision(q)
                ):
                    candidates.append(
                        FreeSpaceCandidate(
                            config=q,
                            pose=pose,
                            negative_volumes=self.cabinet.support_volumes,
                        )
                    )
        return candidates

    def _gen_neutral_candidates(
        self, how_many: int, selfcc: FrankaSelfCollisionChecker
    ) -> List[NeutralCandidate]:
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
                    candidates.append(
                        NeutralCandidate(
                            config=sample,
                            pose=pose,
                            negative_volumes=self.cabinet.support_volumes,
                        )
                    )
        return candidates

    def _gen_additional_candidate_sets(
        self, how_many: int, selfcc: FrankaSelfCollisionChecker
    ) -> List[List[TaskOrientedCandidate]]:
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
                        CabinetCandidate(
                            pose=pose,
                            config=q,
                            pocket_idx=candidate.pocket_idx,
                            support_volume=candidate.support_volume,
                            negative_volumes=candidate.negative_volumes,
                        )
                    )
                    ii += 1
            candidate_sets.append(candidate_set)
        return candidate_sets

    @property
    def obstacles(self) -> List[Union[Cuboid, Cylinder]]:
        return self.cabinet.cuboids

    @property
    def cuboids(self) -> List[Cuboid]:
        return self.cabinet.cuboids

    @property
    def cylinders(self) -> List[Cylinder]:
        return []
