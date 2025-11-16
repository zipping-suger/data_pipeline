from copy import deepcopy
from typing import List, Union, Optional
import numpy as np
from geometrout.primitive import Cuboid, Cylinder
from geometrout.transform import SO3, SE3

from robofin.robots import FrankaRobot, FrankaRealRobot, FrankaGripper
from robofin.bullet import Bullet
from robofin.collision import FrankaSelfCollisionChecker

from data_pipeline.environments.base_environment import (
    TaskOrientedCandidate,
    NeutralCandidate,
    FreeSpaceCandidate,
    Environment,
    Tool,
)
from data_pipeline.environments.tabletop_environment import random_linear_decrease


class BinEnvironment(Environment):
    """
    A bin environment (open box) without objects inside.
    The bin has walls and a bottom, creating a contained workspace.
    """

    def __init__(self):
        super().__init__()
        self.reset()

    def reset(self):
        """
        Resets the state of the environment
        """
        self.objects = []
        self.bin_walls = []  # The walls of the bin
        self.bin_bottom = None  # The bottom of the bin
        self.clear_tables = []  # The table the bin sits on
        self._tools = None
        self.bin_rotation_matrix = np.eye(4)  # Identity matrix for rotation

    def _gen(self, selfcc: FrankaSelfCollisionChecker, how_many: int = 5) -> bool:
        """
        Generates the bin environment and a pair of valid candidates.

        :param selfcc FrankaSelfCollisionChecker: Checks for self collisions
        :param how_many int: How many objects to put in the bin (ignored - no objects placed)
        :rtype bool: Whether the environment was successfully generated
        """
        self.reset()
        self.setup_bin()
        # Removed object placement
        self.generate_tool()

        cand1 = self.gen_candidate(selfcc, self.tools)
        if cand1 is None:
            return False
            
        cand2 = self.gen_candidate(selfcc, self.tools)
        if cand2 is None:
            return False
            
        self.demo_candidates = [cand1, cand2]
        return True

    def setup_bin(self):
        """
        Generate the bin (open box) with specified dimensions and place it on a table.
        Uses proper rotation similar to cubby environment.
        """
        # Bin dimensions based on your specifications
        width = np.random.uniform(0.3, 0.7)  # x-direction
        depth = np.random.uniform(0.2, 0.5)  # y-direction  
        height = np.random.uniform(0.1, 0.4)  # z-direction
        thickness = np.random.uniform(0.02, 0.06)  # wall thickness
        front_scale = np.random.uniform(0.6, 1.0)  # front wall height scale
        
        # Position the bin on the table
        x_pos = np.random.uniform(0.0, 0.8)
        y_pos = np.random.uniform(-0.6, 0.6)
        bin_rotation = np.random.uniform(-1.57, 0.0)  # -90 to 0 degrees
        
        # Create rotation matrix similar to cubby environment
        bin_center = np.array([x_pos, y_pos, 0])
        
        # Build rotation matrix using the same approach as cubby
        cabinet_T_world = np.array([
            [1, 0, 0, -bin_center[0]],
            [0, 1, 0, -bin_center[1]], 
            [0, 0, 1, -bin_center[2]],
            [0, 0, 0, 1]
        ])
        
        in_cabinet_rotation = np.array([
            [np.cos(bin_rotation), -np.sin(bin_rotation), 0, 0],
            [np.sin(bin_rotation), np.cos(bin_rotation), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        world_T_cabinet = np.array([
            [1, 0, 0, bin_center[0]],
            [0, 1, 0, bin_center[1]],
            [0, 0, 1, bin_center[2]],
            [0, 0, 0, 1]
        ])
        
        self.bin_rotation_matrix = world_T_cabinet @ in_cabinet_rotation @ cabinet_T_world
        
        # Create support table similar to tabletop environment (no rotation)
        table_height = np.random.uniform(0.02, 0.05)
        table_dims = [
            width + 0.3,  # Make table wider than bin
            depth + 0.3,  # Make table deeper than bin
            table_height
        ]
        
        # Table is not rotated - positioned directly under the bin
        table_center = [x_pos, y_pos, table_height/2]
        table = Cuboid(
            center=table_center,
            dims=table_dims,
            quaternion=[1, 0, 0, 0]  # No rotation
        )
        self.clear_tables = [table]
        
        # Bin bottom (sits on table)
        bottom_z = table_height  # Bottom starts at table top
        bottom_center = [x_pos, y_pos, bottom_z + thickness/2]
        bottom_dims = [width, depth, thickness]
        
        # Apply rotation to bin bottom
        bottom_pose = self._apply_rotation_to_pose(bottom_center, [1, 0, 0, 0])
        self.bin_bottom = Cuboid(
            center=bottom_pose.xyz,
            dims=bottom_dims,
            quaternion=bottom_pose.so3.wxyz
        )
        
        # Bin walls - create unrotated positions first, then apply rotation
        # Back wall (full height)
        back_wall_center = [
            x_pos, 
            y_pos + (depth - thickness)/2, 
            bottom_z + thickness + (height * 1.0)/2
        ]
        back_wall_dims = [width, thickness, height * 1.0]
        back_wall_pose = self._apply_rotation_to_pose(back_wall_center, [1, 0, 0, 0])
        back_wall = Cuboid(
            center=back_wall_pose.xyz,
            dims=back_wall_dims,
            quaternion=back_wall_pose.so3.wxyz
        )
        
        # Left wall (full height)
        left_wall_center = [
            x_pos - (width - thickness)/2,
            y_pos,
            bottom_z + thickness + (height * 1.0)/2
        ]
        left_wall_dims = [thickness, depth, height * 1.0]
        left_wall_pose = self._apply_rotation_to_pose(left_wall_center, [1, 0, 0, 0])
        left_wall = Cuboid(
            center=left_wall_pose.xyz,
            dims=left_wall_dims,
            quaternion=left_wall_pose.so3.wxyz
        )
        
        # Right wall (full height)
        right_wall_center = [
            x_pos + (width - thickness)/2,
            y_pos,
            bottom_z + thickness + (height * 1.0)/2
        ]
        right_wall_dims = [thickness, depth, height * 1.0]
        right_wall_pose = self._apply_rotation_to_pose(right_wall_center, [1, 0, 0, 0])
        right_wall = Cuboid(
            center=right_wall_pose.xyz,
            dims=right_wall_dims,
            quaternion=right_wall_pose.so3.wxyz
        )
        
        # Front wall (scaled height)
        front_wall_height = height * front_scale
        front_wall_center = [
            x_pos,
            y_pos - (depth - thickness)/2,
            bottom_z + thickness + front_wall_height/2
        ]
        front_wall_dims = [width, thickness, front_wall_height]
        front_wall_pose = self._apply_rotation_to_pose(front_wall_center, [1, 0, 0, 0])
        front_wall = Cuboid(
            center=front_wall_pose.xyz,
            dims=front_wall_dims,
            quaternion=front_wall_pose.so3.wxyz
        )
        
        self.bin_walls = [back_wall, left_wall, right_wall, front_wall]
        
        # Store bin parameters for candidate generation
        self.bin_params = {
            'center': [x_pos, y_pos, bottom_z + thickness],
            'dims': [width - 2*thickness, depth - 2*thickness, height - thickness],  # Interior dimensions
            'rotation': bin_rotation,
            'bottom_height': bottom_z + thickness,
            'rotation_matrix': self.bin_rotation_matrix
        }

    def _apply_rotation_to_pose(self, center: List[float], quaternion: List[float]) -> SE3:
        """
        Apply the bin rotation matrix to a pose.
        
        :param center: The center position [x, y, z]
        :param quaternion: The quaternion [w, x, y, z]
        :return: Transformed SE3 pose
        """
        # Create the original pose
        original_pose = SE3(xyz=center, so3=SO3(quaternion=quaternion))
        
        # Apply rotation matrix
        new_matrix = self.bin_rotation_matrix @ original_pose.matrix
        return SE3(new_matrix)

    def place_objects(self, how_many: int):
        """
        Removed - no objects placed in the bin.
        """
        pass  # No objects placed in the bin

    def random_points_in_bin(self, how_many: int) -> np.ndarray:
        """
        Generate random points inside the bin for candidate generation.

        :param how_many int: How many points to generate
        :rtype np.ndarray: A set of points, has dim [how_many, 3]
        """
        params = self.bin_params
        center = params['center']
        dims = params['dims']
        rotation_matrix = params['rotation_matrix']
        bottom_height = params['bottom_height']
        
        # Generate points in local bin coordinates
        local_points = np.random.uniform(
            [-dims[0]/2, -dims[1]/2, 0],
            [dims[0]/2, dims[1]/2, dims[2]],
            size=(how_many, 3)
        )
        
        # Transform to world coordinates using rotation matrix
        world_points = []
        for point in local_points:
            # Translate to bin center first
            translated_point = point + np.array(center) - np.array([0, 0, bottom_height])
            
            # Apply rotation
            point_homogeneous = np.array([translated_point[0], translated_point[1], translated_point[2], 1.0])
            transformed_point = rotation_matrix @ point_homogeneous
            world_points.append(transformed_point[:3])
        
        return np.array(world_points)

    @property
    def obstacles(self) -> List[Union[Cuboid, Cylinder]]:
        """
        Returns all obstacles in the scene.
        """
        return self.clear_tables + self.bin_walls + [self.bin_bottom] + self.objects
    
    @property
    def tools(self) -> Optional[Tool]:
        """
        Returns all attached tools in the scene.
        """
        return self._tools

    @property
    def cuboids(self) -> List[Cuboid]:
        """
        Returns just the cuboids in the scene
        """
        return [o for o in self.obstacles if isinstance(o, Cuboid)]

    @property
    def cylinders(self) -> List[Cylinder]:
        """
        Returns just the cylinders in the scene
        """
        return [o for o in self.obstacles if isinstance(o, Cylinder)]

    def _gen_additional_candidate_sets(
        self, how_many: int, selfcc: FrankaSelfCollisionChecker
    ) -> List[List[TaskOrientedCandidate]]:
        """
        Generate additional candidate sets for the bin environment.
        """
        cand_set1 = []
        while len(cand_set1) < how_many:
            cand = self.gen_candidate(selfcc, self.tools)
            if cand is not None:
                cand_set1.append(cand)

        cand_set2 = []
        while len(cand_set2) < how_many:
            cand = self.gen_candidate(selfcc, self.tools)
            if cand is not None:
                cand_set2.append(cand)
        
        return [cand_set1, cand_set2]

    def _gen_neutral_candidates(
        self, how_many: int, selfcc: FrankaSelfCollisionChecker
    ) -> List[NeutralCandidate]:
        """
        Generate neutral candidates for the bin environment.
        """
        sim = Bullet(gui=False)
        gripper = sim.load_robot(FrankaGripper)
        arm = sim.load_robot(FrankaRobot)
        sim.load_primitives(self.obstacles)
        candidates = []
                
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
                    if not self._check_tool_collision(pose, self.obstacles, self.tools):
                        candidates.append(
                            NeutralCandidate(config=sample, pose=pose, negative_volumes=[], tool=self.tools)
                        )
        return candidates

    def _gen_free_space_candidates(
        self, how_many: int, selfcc: FrankaSelfCollisionChecker
    ) -> List[FreeSpaceCandidate]:
        """
        Generate free space candidates for the bin environment.
        """
        sim = Bullet(gui=False)
        gripper = sim.load_robot(FrankaGripper)
        arm = sim.load_robot(FrankaRobot)
        sim.load_primitives(self.obstacles)
        candidates: List[FreeSpaceCandidate] = []

        position_ranges = {
            "x": (0, 1),
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

            if self._check_tool_collision(pose, self.obstacles, self.tools):
                continue

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
                            negative_volumes=[],
                            tool=self.tools
                        )
                    )
        return candidates

    def gen_candidate(self, selfcc: FrankaSelfCollisionChecker, tool: Tool) -> Optional[TaskOrientedCandidate]:
        """
        Generate a candidate pose inside the bin.
        """
        points = self.random_points_in_bin(100)
        sim = Bullet(gui=False)
        sim.load_primitives(self.obstacles)
        gripper = sim.load_robot(FrankaGripper)
        arm = sim.load_robot(FrankaRobot)
            
        q = None
        pose = None
        for p in points:
            # Height sampling above bin bottom but below bin top
            bin_bottom = self.bin_params['bottom_height']
            bin_top = bin_bottom + self.bin_params['dims'][2]
            
            min_safe_height = np.random.uniform(0.03, 0.1)
            max_safe_height = min(0.2, bin_top - bin_bottom - 0.05)  # Don't go too close to bin top
            
            if max_safe_height <= min_safe_height:
                continue
                
            p[2] = bin_bottom + random_linear_decrease() * (max_safe_height - min_safe_height) + min_safe_height
            
            # Orientation sampling - avoid pointing into bin walls
            roll = np.random.uniform(4 * np.pi / 5, 6 * np.pi / 5)  # 144°-216°
            pitch = np.random.uniform(-np.pi / 12, np.pi / 12)  # ±15°
            yaw = np.random.uniform(-np.pi / 2, np.pi / 2)
            
            pose = SE3(xyz=p, so3=SO3.from_rpy(roll, pitch, yaw))
            
            gripper.marionette(pose)

            if self._check_tool_collision(pose, self.obstacles, tool):
                pose = None
                continue
            
            if sim.in_collision(gripper):
                pose = None
                continue
                
            q = FrankaRealRobot.collision_free_ik(sim, arm, selfcc, pose, retries=1000)
            if q is not None:
                break
                
        if pose is None or q is None:
            return None
            
        return TaskOrientedCandidate(
            pose=pose,
            config=q,
            negative_volumes=[],
            tool=tool
        )
        
    def generate_tool(self):
        """
        Generate a tool for the bin environment (same as tabletop).
        """
        # Reuse the same tool generation as TabletopEnvironment
        from data_pipeline.environments.tabletop_environment import TabletopEnvironment
        temp_env = TabletopEnvironment()
        temp_env.generate_tool()
        self._tools = temp_env.tools