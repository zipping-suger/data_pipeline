# MIT License
#
# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES, University of Washington. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from typing import List, Optional, Tuple
import numpy as np
from geometrout.primitive import Cuboid, Cylinder
from geometrout.transform import SE3, SO3

from robofin.robots import FrankaRobot, FrankaRealRobot, FrankaGripper
from robofin.bullet import Bullet, BulletFranka, BulletFrankaGripper
from robofin.collision import FrankaSelfCollisionChecker

from data_pipeline.environments.base_environment import (
    TaskOrientedCandidate,
    NeutralCandidate,
    Environment,
)


class FreeSpaceEnvironment(Environment):
    """
    A free space environment with just a dummy floor under the robot.
    This is useful for testing motion planning in obstacle-free space.
    """

    def __init__(self):
        super().__init__()
        self._floor = Cuboid(
            center=[0.0, 0.0, -0.01],  # Just below the robot base
            dims=[2.0, 2.0, 0.02],  # Large enough floor area
            quaternion=[1, 0, 0, 0],  # No rotation
        )
        # Define the sampling ranges for pose generation
        self.position_ranges = {
            'x': (-0.8, 0.8),   # meters from robot base
            'y': (-0.8, 0.8),   # meters from robot base
            'z': (0.0, 1.0)     # meters from floor
        }
        self.orientation_ranges = {
            'roll': (-np.pi, np.pi),   # radians
            'pitch': (-np.pi/2, np.pi/2),  # radians
            'yaw': (-np.pi, np.pi)     # radians
        }

    def generate_random_pose(self) -> SE3:
        """
        Generates a random end-effector pose within the specified ranges.
        
        :return: SE3 transform representing the end-effector pose
        """
        # Sample position
        x = np.random.uniform(*self.position_ranges['x'])
        y = np.random.uniform(*self.position_ranges['y'])
        z = np.random.uniform(*self.position_ranges['z'])
        position = np.array([x, y, z])
        
        # Sample orientation (roll, pitch, yaw)
        roll = np.random.uniform(*self.orientation_ranges['roll'])
        pitch = np.random.uniform(*self.orientation_ranges['pitch'])
        yaw = np.random.uniform(*self.orientation_ranges['yaw'])
        
        # Create SE3 from position and RPY angles
        return SE3(xyz=position, rpy=[roll, pitch, yaw])

    def random_pose_and_config(
        self,
        sim: Bullet,
        gripper: BulletFrankaGripper,
        arm: BulletFranka,
        selfcc: FrankaSelfCollisionChecker,
        max_attempts: int = 100
    ) -> Tuple[Optional[SE3], Optional[np.ndarray]]:
        """
        Creates a random end effector pose within specified ranges and solves for
        collision free IK.

        :param sim Bullet: A simulator already loaded with the obstacles
        :param gripper BulletFrankaGripper: The simulated gripper for collision checking
        :param arm BulletFranka: The simulated arm for collision checking
        :param selfcc FrankaSelfCollisionChecker: Checks for self collisions
        :param max_attempts int: Maximum number of attempts to find a valid pose/config
        :return: (pose, config) tuple if successful, (None, None) otherwise
        """
        for _ in range(max_attempts):
            # Generate random pose within specified ranges
            pose = self.generate_random_pose()
            
            # Check gripper collision
            gripper.marionette(pose)
            if sim.in_collision(gripper):
                continue
                
            # Solve IK
            q = FrankaRealRobot.collision_free_ik(
                sim, arm, selfcc, pose, retries=5
            )
            if q is not None:
                return pose, q
                
        return None, None

    def _gen(self, selfcc: FrankaSelfCollisionChecker) -> bool:
        """
        Generates start and goal candidates in free space.
        """
        sim = Bullet(gui=False)
        sim.load_primitives(self.obstacles)
        gripper = sim.load_robot(FrankaGripper)
        arm = sim.load_robot(FrankaRobot)

        # Get start pose and config
        start_pose, start_q = self.random_pose_and_config(
            sim, gripper, arm, selfcc
        )
        if start_pose is None or start_q is None:
            return False
            
        # Get target pose and config
        target_pose, target_q = self.random_pose_and_config(
            sim, gripper, arm, selfcc
        )
        if target_pose is None or target_q is None:
            return False

        self.demo_candidates = (
            TaskOrientedCandidate(pose=start_pose, config=start_q, negative_volumes=[]),
            TaskOrientedCandidate(pose=target_pose, config=target_q, negative_volumes=[]),
        )
        return True

    def _gen_neutral_candidates(
        self, how_many: int, selfcc: FrankaSelfCollisionChecker
    ) -> List[NeutralCandidate]:
        """
        Generate neutral configurations in free space.
        """
        sim = Bullet(gui=False)
        sim.load_primitives(self.obstacles)
        gripper = sim.load_robot(FrankaGripper)
        arm = sim.load_robot(FrankaRobot)
        
        candidates: List[NeutralCandidate] = []
        for _ in range(how_many * 10):  # Allow for some failures
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
                            negative_volumes=[],
                        )
                    )
        return candidates

    def _gen_additional_candidate_sets(
        self, how_many: int, selfcc: FrankaSelfCollisionChecker
    ) -> List[List[TaskOrientedCandidate]]:
        """
        Generate additional candidate sets in free space.
        """
        sim = Bullet(gui=False)
        sim.load_primitives(self.obstacles)
        gripper = sim.load_robot(FrankaGripper)
        arm = sim.load_robot(FrankaRobot)
        
        candidate_sets = []
        for _ in range(2):  # For start and goal sets
            candidates = []
            while len(candidates) < how_many:
                pose, q = self.random_pose_and_config(
                    sim, gripper, arm, selfcc
                )
                if pose is not None and q is not None:
                    candidates.append(
                        TaskOrientedCandidate(
                            pose=pose, config=q, negative_volumes=[]
                        )
                    )
            candidate_sets.append(candidates)
        return candidate_sets

    @property
    def obstacles(self) -> List[Cuboid]:
        """
        Returns just the floor cuboid.
        :rtype List[Cuboid]: List containing the floor
        """
        return [self._floor]

    @property
    def cuboids(self) -> List[Cuboid]:
        """
        Returns just the floor cuboid.
        :rtype List[Cuboid]: List containing the floor
        """
        return [self._floor]

    @property
    def cylinders(self) -> List[Cylinder]:
        """
        Returns empty list since there are no cylinders.
        :rtype List[Cylinder]: Empty list
        """
        return []