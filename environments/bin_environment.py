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
    A bin environment (open box) with random objects placed inside it.
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

    def _gen(self, selfcc: FrankaSelfCollisionChecker, how_many: int = 5) -> bool:
        """
        Generates the bin environment and a pair of valid candidates.

        :param selfcc FrankaSelfCollisionChecker: Checks for self collisions
        :param how_many int: How many objects to put in the bin
        :rtype bool: Whether the environment was successfully generated
        """
        self.reset()
        self.setup_bin()
        self.place_objects(how_many)
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
        z_rotation = np.random.uniform(-1.57, 0.0)  # -90 to 0 degrees
        
        # The table that the bin sits on
        table_height = 0.02  # Thin table
        table = Cuboid(
            center=[x_pos, y_pos, -table_height/2],
            dims=[width + 0.1, depth + 0.1, table_height],  # Table slightly larger than bin
            quaternion=[1, 0, 0, 0]
        )
        self.clear_tables = [table]
        
        # Bin bottom (sits on table)
        bottom_z = table_height  # Bottom starts at table top
        self.bin_bottom = Cuboid(
            center=[x_pos, y_pos, bottom_z + thickness/2],
            dims=[width, depth, thickness],
            quaternion=SO3.from_rpy(0, 0, z_rotation).wxyz
        )
        
        # Bin walls
        # Back wall (full height)
        back_wall = Cuboid(
            center=[
                x_pos, 
                y_pos + (depth - thickness)/2, 
                bottom_z + thickness + (height * 1.0)/2
            ],
            dims=[width, thickness, height * 1.0],
            quaternion=SO3.from_rpy(0, 0, z_rotation).wxyz
        )
        
        # Left wall (full height)
        left_wall = Cuboid(
            center=[
                x_pos - (width - thickness)/2,
                y_pos,
                bottom_z + thickness + (height * 1.0)/2
            ],
            dims=[thickness, depth, height * 1.0],
            quaternion=SO3.from_rpy(0, 0, z_rotation).wxyz
        )
        
        # Right wall (full height)
        right_wall = Cuboid(
            center=[
                x_pos + (width - thickness)/2,
                y_pos,
                bottom_z + thickness + (height * 1.0)/2
            ],
            dims=[thickness, depth, height * 1.0],
            quaternion=SO3.from_rpy(0, 0, z_rotation).wxyz
        )
        
        # Front wall (scaled height)
        front_wall_height = height * front_scale
        front_wall = Cuboid(
            center=[
                x_pos,
                y_pos - (depth - thickness)/2,
                bottom_z + thickness + front_wall_height/2
            ],
            dims=[width, thickness, front_wall_height],
            quaternion=SO3.from_rpy(0, 0, z_rotation).wxyz
        )
        
        self.bin_walls = [back_wall, left_wall, right_wall, front_wall]
        
        # Store bin parameters for object placement
        self.bin_params = {
            'center': [x_pos, y_pos, bottom_z + thickness],
            'dims': [width - 2*thickness, depth - 2*thickness, height - thickness],  # Interior dimensions
            'rotation': z_rotation,
            'bottom_height': bottom_z + thickness
        }

    def place_objects(self, how_many: int):
        """
        Places random objects inside the bin.

        :param how_many int: How many objects
        """
        center_candidates = self.random_points_in_bin(10 * how_many)
        objects = []
        point_idx = 0
        for candidate in center_candidates:
            if len(objects) >= how_many:
                break
            candidate_is_good = True
            min_sdf = 1000
            for o in objects:
                sdf_value = o.sdf(candidate)
                if min_sdf is None or sdf_value < min_sdf:
                    min_sdf = sdf_value
                if o.sdf(candidate) <= 0.05:
                    candidate_is_good = False
            if candidate_is_good:
                x, y, z = candidate
                objects.append(self.random_object(x, y, z, 0.05, min_sdf))
        self.objects.extend(objects)

    def random_points_in_bin(self, how_many: int) -> np.ndarray:
        """
        Generate random points inside the bin for object placement.

        :param how_many int: How many points to generate
        :rtype np.ndarray: A set of points, has dim [how_many, 3]
        """
        params = self.bin_params
        center = params['center']
        dims = params['dims']
        rotation = params['rotation']
        bottom_height = params['bottom_height']
        
        # Generate points in local bin coordinates
        local_points = np.random.uniform(
            [-dims[0]/2, -dims[1]/2, 0],
            [dims[0]/2, dims[1]/2, dims[2]],
            size=(how_many, 3)
        )
        
        # Transform to world coordinates
        cos_rot = np.cos(rotation)
        sin_rot = np.sin(rotation)
        
        world_points = []
        for point in local_points:
            # Rotate
            x_rot = point[0] * cos_rot - point[1] * sin_rot
            y_rot = point[0] * sin_rot + point[1] * cos_rot
            z_rot = point[2]
            
            # Translate
            x_world = center[0] + x_rot
            y_world = center[1] + y_rot
            z_world = bottom_height + z_rot
            
            world_points.append([x_world, y_world, z_world])
        
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
            # Determine base height based on object surface
            base_height = p[2]
            for o in self.objects:
                if o.sdf(p) <= 0.01:
                    if isinstance(o, Cuboid):
                        base_height = o.center[2] + o.half_extents[2]
                    elif isinstance(o, Cylinder):
                        base_height = o.center[2] + o.height / 2
            
            # Height sampling above objects but below bin top
            bin_top = self.bin_params['bottom_height'] + self.bin_params['dims'][2]
            min_safe_height = np.random.uniform(0.03, 0.1)
            max_safe_height = min(0.2, bin_top - base_height - 0.05)  # Don't go too close to bin top
            
            if max_safe_height <= min_safe_height:
                continue
                
            p[2] = base_height + random_linear_decrease() * (max_safe_height - min_safe_height) + min_safe_height
            
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
        
    def random_object(
        self, x: float, y: float, base_height: float, dim_min: float, dim_max: float
    ) -> Union[Cuboid, Cylinder]:
        """
        Generate a random object inside the bin.
        """
        xy_dim_max = min(dim_max, 0.15)
        if np.random.rand() < 0.3:
            c = Cylinder.random(
                center_range=None,
                radius_range=[dim_min, xy_dim_max],
                height_range=[0.05, 0.2],  # Smaller objects for bin
                quaternion=False,
            )
            c.center = [x, y, c.height / 2 + base_height]
        else:
            c = Cuboid.random(
                center_range=None,
                dimension_range=[
                    [dim_min, dim_min, 0.05],
                    [xy_dim_max, xy_dim_max, 0.2],  # Smaller objects for bin
                ],
                quaternion=False,
            )
            c.center = [x, y, c.half_extents[2] + base_height]
            c.pose._so3 = SO3.from_rpy(0, 0, np.random.uniform(0, np.pi / 2))
        return c