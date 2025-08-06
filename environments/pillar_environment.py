from dataclasses import dataclass
import random
from typing import List, Optional, Tuple, Union

import numpy as np
from geometrout.primitive import Cuboid, Cylinder
from geometrout.transform import SE3
from robofin.bullet import Bullet, BulletFranka, BulletFrankaGripper
from robofin.robots import FrankaGripper, FrankaRobot, FrankaRealRobot
from robofin.collision import FrankaSelfCollisionChecker

from data_pipeline.environments.base_environment import (
    TaskOrientedCandidate,
    NeutralCandidate,
    FreeSpaceCandidate,
    Environment,
    radius_sample,
)


@dataclass
class PillarCandidate(TaskOrientedCandidate):
    """
    Represents a configuration and the corresponding end-effector pose
    (in the right_gripper frame) for a task in the pillar environment.
    """
    pass


class PillarEnvironment(Environment):
    def __init__(self):
        super().__init__()
        self.base_table = None
        self.pillars = []
        self.blocks = []
        self.demo_candidates = []

    def _gen(self, selfcc: FrankaSelfCollisionChecker) -> bool:
        """
        Generates a pillar environment with a base table, cylindrical pillars,
        and cuboid blocks. Also generates start and goal candidates.
        """
        # Clear previous environment
        self.base_table = None
        self.pillars = []
        self.blocks = []
        self.demo_candidates = []

        # Create base table
        self._create_base_table()
        
        # Create obstacles
        self._create_pillars()
        self._create_blocks()
        
        # Generate start and goal candidates
        sim = Bullet(gui=False)
        sim.load_primitives(self.obstacles)
        gripper = sim.load_robot(FrankaGripper)
        arm = sim.load_robot(FrankaRobot)
        
        start_pose, start_q = self._generate_valid_pose_and_config(
            sim, gripper, arm, selfcc
        )
        target_pose, target_q = self._generate_valid_pose_and_config(
            sim, gripper, arm, selfcc
        )
        
        if start_pose is None or start_q is None or target_pose is None or target_q is None:
            return False
            
        self.demo_candidates = [
            PillarCandidate(pose=start_pose, config=start_q, negative_volumes=[]),
            PillarCandidate(pose=target_pose, config=target_q, negative_volumes=[])
        ]
        return True

    def _create_base_table(self):
        """Create a base table for the robot"""
        table_height = 0.0
        table_thickness = 0.02
        table_dims = [1.0, 1.6, table_thickness]  # x, y, z dimensions
        table_center = [0.5, 0.0, table_height - table_thickness/2]
        
        self.base_table = Cuboid(
            center=table_center,
            dims=table_dims,
            quaternion=[1, 0, 0, 0]
        )

    def _create_pillars(self):
        """Create cylindrical pillars in the workspace"""
        num_pillars = random.randint(3, 6)
        for _ in range(num_pillars):
            # Random position within workspace
            x = random.uniform(0.1, 0.9)
            y = random.uniform(-0.7, 0.7)
            
            # Random dimensions
            radius = random.uniform(0.03, 0.08)
            height = random.uniform(0.3, 0.8)
            z = height/2  # Center at half height
            
            pillar = Cylinder(
                center=[x, y, z],
                radius=radius,
                height=height,
                quaternion=[1, 0, 0, 0]
            )
            self.pillars.append(pillar)

    def _create_blocks(self):
        """Create cuboid blocks in the workspace"""
        num_blocks = random.randint(2, 4)
        for _ in range(num_blocks):
            # Random position within workspace
            x = random.uniform(0.1, 0.9)
            y = random.uniform(-0.7, 0.7)
            z = 0.0  # Will be adjusted based on height
            
            # Random dimensions
            dim_x = random.uniform(0.05, 0.15)
            dim_y = random.uniform(0.05, 0.15)
            dim_z = random.uniform(0.05, 0.6)
            z = dim_z/2  # Center at half height
            
            block = Cuboid(
                center=[x, y, z],
                dims=[dim_x, dim_y, dim_z],
                quaternion=[1, 0, 0, 0]
            )
            self.blocks.append(block)

    def _generate_valid_pose_and_config(
        self,
        sim: Bullet,
        gripper: BulletFrankaGripper,
        arm: BulletFranka,
        selfcc: FrankaSelfCollisionChecker,
        max_attempts: int = 100
    ) -> Tuple[Optional[SE3], Optional[np.ndarray]]:
        """
        Generate a valid end effector pose and configuration
        within the workspace, avoiding collisions
        """
        for _ in range(max_attempts):
            # Generate random pose within workspace
            x = random.uniform(0.0, 1.0)
            y = random.uniform(-0.8, 0.8)
            z = random.uniform(0.1, 0.7)
            
            # Random orientation
            roll = random.uniform(-np.pi, np.pi)
            pitch = random.uniform(-np.pi/2, np.pi/2)
            yaw = random.uniform(-np.pi, np.pi)
            
            pose = SE3(xyz=[x, y, z], rpy=[roll, pitch, yaw])
            
            # Check gripper collision
            gripper.marionette(pose)
            if sim.in_collision(gripper):
                continue
                
            # Solve IK
            q = FrankaRealRobot.collision_free_ik(sim, arm, selfcc, pose, retries=5)
            if q is not None:
                arm.marionette(q)
                if not (sim.in_collision(arm) or selfcc.has_self_collision(q)):
                    return pose, q
        return None, None

    def _gen_free_space_candidates(
        self, how_many: int, selfcc: FrankaSelfCollisionChecker
    ) -> List[FreeSpaceCandidate]:
        """
        Generate free space candidates in the pillar environment
        """
        sim = Bullet(gui=False)
        gripper = sim.load_robot(FrankaGripper)
        arm = sim.load_robot(FrankaRobot)
        sim.load_primitives(self.obstacles)
        candidates = []
        
        while len(candidates) < how_many:
            pose, q = self._generate_valid_pose_and_config(
                sim, gripper, arm, selfcc
            )
            if pose is not None and q is not None:
                candidates.append(
                    FreeSpaceCandidate(
                        config=q,
                        pose=pose,
                        negative_volumes=[]
                    )
                )
        return candidates

    def _gen_neutral_candidates(
        self, how_many: int, selfcc: FrankaSelfCollisionChecker
    ) -> List[NeutralCandidate]:
        """
        Generate neutral candidates in the pillar environment
        """
        sim = Bullet(gui=False)
        gripper = sim.load_robot(FrankaGripper)
        arm = sim.load_robot(FrankaRobot)
        sim.load_primitives(self.obstacles)
        candidates = []
        
        for _ in range(how_many * 10):
            if len(candidates) >= how_many:
                break
                
            sample = FrankaRealRobot.random_neutral(method="uniform")
            arm.marionette(sample)
            if not (sim.in_collision(arm) or selfcc.has_self_collision(sample)):
                pose = FrankaRealRobot.fk(sample, eff_frame="right_gripper")
                gripper.marionette(pose)
                if not sim.in_collision(gripper):
                    candidates.append(
                        NeutralCandidate(config=sample, pose=pose, negative_volumes=[])
                    )
        return candidates

    def _gen_additional_candidate_sets(
        self, how_many: int, selfcc: FrankaSelfCollisionChecker
    ) -> List[List[TaskOrientedCandidate]]:
        """
        Generate additional candidate sets for the pillar environment
        """
        sim = Bullet(gui=False)
        gripper = sim.load_robot(FrankaGripper)
        arm = sim.load_robot(FrankaRobot)
        sim.load_primitives(self.obstacles)
        candidate_sets = []
        
        for _ in range(2):  # Start and goal sets
            candidate_set = []
            while len(candidate_set) < how_many:
                pose, q = self._generate_valid_pose_and_config(
                    sim, gripper, arm, selfcc
                )
                if pose is not None and q is not None:
                    candidate_set.append(
                        PillarCandidate(
                            pose=pose,
                            config=q,
                            negative_volumes=[]
                        )
                    )
            candidate_sets.append(candidate_set)
        return candidate_sets

    @property
    def obstacles(self) -> List[Union[Cuboid, Cylinder]]:
        """All obstacles in the scene"""
        return [self.base_table] + self.pillars + self.blocks

    @property
    def cuboids(self) -> List[Cuboid]:
        """Cuboid obstacles in the scene"""
        return [self.base_table] + self.blocks

    @property
    def cylinders(self) -> List[Cylinder]:
        """Cylindrical obstacles in the scene"""
        return self.pillars
