from copy import deepcopy
from typing import List, Union, Optional
import time
from dataclasses import dataclass

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


def random_linear_decrease():
    """
    Generates a random number according a distribution between 0 and 1 where the PDF looks
    like a linear line with slope -1. Useful for generating numbers within a range where
    the lower numbers are more preferred than the larger ones.
    """
    return 1 - np.sqrt(np.random.uniform())


class TabletopEnvironment(Environment):
    """
    A randomly constructed tabletop environment with random objects placed on it. The tabletop
    Table setup can be L-shaped or l-shaped, but will always include an obstacle free base
    table under the robot and some obstacle free space on the tables.
    """

    def __init__(self):
        super().__init__()
        self.reset()

    def reset(self):
        """
        Resets the state of the environment
        """
        self.objects = []
        self.tables = []  # The tables with objects on it
        self.clear_tables = []  # The tables without objects
        self._tools = None  # Use _tools instead of tools

    def _gen(self, selfcc: FrankaSelfCollisionChecker, how_many: int) -> bool:
        """
        Generates the environment and a pair of valid candidates. The environment has a
        object-free table under the robot's base, as well as possibly some object-free
        sections on the table that are outside of the workspace.

        :param selfcc FrankaSelfCollisionChecker: Checks for self collisions using spheres that
                                                  mimic the internal Franka collision checker.
        :param how_many int: How many objects to put on the tables
        :rtype bool: Whether the environment was successfully generated
        """
        self.reset()
        self.setup_tables()
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

    def _gen_free_space_candidates(
        self, how_many: int, selfcc: FrankaSelfCollisionChecker
    ) -> List[FreeSpaceCandidate]:
        sim = Bullet(gui=False)
        gripper = sim.load_robot(FrankaGripper)
        arm = sim.load_robot(FrankaRobot)
        sim.load_primitives(self.obstacles)
        candidates: List[FreeSpaceCandidate] = []

        # Define the sampling ranges for pose generation (similar to free_environment)
        position_ranges = {
            "x": (0, 1),  # meters from robot base
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

            # NEW: Check tool collision
            if self._check_tool_collision(pose, self.obstacles, self.tools):
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
                            negative_volumes=[],
                            tool=self.tools
                        )
                    )
        return candidates

    def _gen_neutral_candidates(
        self, how_many: int, selfcc: FrankaSelfCollisionChecker
    ) -> List[NeutralCandidate]:
        """
        Generate a set of collision free neutral poses (represented as NeutralCandidate object)

        :param how_many int: How many neutral poses to generate
        :param selfcc FrankaSelfCollisionChecker: Checks for self collisions using spheres that
                                                  mimic the internal Franka collision checker.
        :rtype List[NeutralCandidate]: A list of neutral poses
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
                    # NEW: Check tool collision
                    if not self._check_tool_collision(pose, self.obstacles, self.tools):
                        candidates.append(
                            NeutralCandidate(config=sample, pose=pose, negative_volumes=[], tool=self.tools)
                        )
        return candidates

    def place_objects(self, how_many: int):
        """
        Places random objects on the table's surface, which _should_ not be overlapping.
        Overlap is calculated with a heuristic, so it's possible for them to overlap sometimes.

        :param how_many int: How many objects
        """
        center_candidates = self.random_points_on_table(10 * how_many)
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

    @property
    def obstacles(self) -> List[Union[Cuboid, Cylinder]]:
        """
        Returns all obstacles in the scene.
        :rtype List[Union[Cuboid, Cylinder]]: The list of obstacles in the scene
        """
        return self.tables + self.clear_tables + self.objects
    
    @property
    def tools(self) -> Optional[Tool]:
        """
        Returns all attatched tools in the scene. Can be None.
        """
        return self._tools

    @property
    def cuboids(self) -> List[Cuboid]:
        """
        Returns just the cuboids in the scene
        :rtype List[Cuboid]: The list of obstacles in the scene
        """
        return [o for o in self.obstacles if isinstance(o, Cuboid)]

    @property
    def cylinders(self) -> List[Cylinder]:
        """
        Returns just the cylinders in the scene
        :rtype List[Cylinder]: The list of obstacles in the scene
        """
        return [o for o in self.obstacles if isinstance(o, Cylinder)]

    def random_points_on_table(self, how_many: int) -> np.ndarray:
        """
        Generated random points on the table surface (to be used for obstacle placement).
        Points should be distributed _roughly_ evenly across all tables.

        :param how_many int: How many points to generate
        :rtype np.ndarray: A set of points, has dim [how_many, 3]
        """
        areas = []
        for t in self.tables:
            x, y, _ = t.dims
            areas.append(x * y)
        # We want to evenly sample points from each of the tables
        # First define which point will correspond to which tabletop
        table_choices = np.random.choice(
            np.arange(len(self.tables)),
            size=how_many,
            p=np.array(areas) / np.sum(areas),
        )
        # Sample points from the surface and then
        # only take those on the top
        # This relies on the fact that the tabletop is horizontal
        pointsets = []
        for t in self.tables:
            x_min, y_min, _ = np.min(t.corners, axis=0)
            x_max, y_max, z_max = np.max(t.corners, axis=0)
            # Always sample with points on the top surface
            pointsets.append(
                np.random.uniform(
                    [x_min, y_min, z_max],
                    [x_max, y_max, z_max],
                    size=(how_many, 3),
                )
            )
        return np.stack(pointsets)[table_choices, np.arange(how_many), :]

    def setup_tables(self):
        """
        Generate the random tables. Table setup can be L-shaped or l-shaped, but will always include
        an obstacle free base table under the robot. Additionally, objects are only placed
        within a randomly generated workspace within the table. The `self.tables` object
        has the tables that can have stuff on them. The `self.clear_tables` will have no objects
        placed on them.
        """
        table_height = np.random.choice(
            (np.random.uniform(0, 0.4), 0.0), p=[0.65, 0.35]
        )
        z = (table_height + -0.02) / 2
        dim_z = table_height + 0.02
        # Setup front table
        front_x_min = np.random.uniform(0.275, 0.375)
        front_x_max = np.random.uniform(1.275, 1.375)
        front_y_max = np.random.uniform(1.5, 1.65)
        has_side_table = np.random.uniform() < 0.5
        if has_side_table:
            front_y_min = np.random.uniform(-0.75, -1.0)
            pass
        else:
            front_y_min = np.random.uniform(-0.55, -0.75)

        whole_front_table = Cuboid(
            center=[
                (front_x_min + front_x_max) / 2,
                (front_y_min + front_y_max) / 2,
                z,
            ],
            dims=[
                (front_x_max - front_x_min),
                (front_y_max - front_y_min),
                dim_z,
            ],
            quaternion=[1, 0, 0, 0],
        )

        corners = whole_front_table.corners
        front_task_table_scalar = np.random.uniform(0.55, 0.65)
        front_task_table = deepcopy(whole_front_table)
        front_task_table._pose._xyz[
            1
        ] = front_task_table_scalar * whole_front_table.dims[1] / 2 + np.min(
            whole_front_table.corners[:, 1]
        )
        front_task_table._dims[1] = front_task_table_scalar * whole_front_table.dims[1]
        self.tables = [front_task_table]

        front_free_table = deepcopy(whole_front_table)
        front_free_table._dims[1] = whole_front_table.dims[1] - front_task_table.dims[1]
        front_free_table._pose._xyz[1] = (
            np.max(whole_front_table.corners, axis=0)[1]
            + np.max(front_task_table.corners, axis=0)[1]
        ) / 2
        self.clear_tables = [front_free_table]

        if has_side_table:
            side_y_max = np.random.uniform(-0.275, -0.325)
            side_y_min = min(front_task_table.corners[:, 1])
            side_x_max = min(front_task_table.corners[:, 0])
            side_x_min = side_x_max - np.random.uniform(0, 1.375)
            whole_side_table = Cuboid(
                [(side_x_max + side_x_min) / 2, (side_y_max + side_y_min) / 2, z],
                [side_x_max - side_x_min, side_y_max - side_y_min, dim_z],
                quaternion=[1, 0, 0, 0],
            )
            side_y = (side_y_max + min(front_task_table.corners[:, 1])) / 2
            side_x = (
                np.random.uniform(-1.275, -1.375) + min(front_task_table.corners[:, 0])
            ) / 2
            corners = whole_side_table.corners
            side_task_table_scalar = np.random.uniform(0.55, 0.65)
            side_task_table = deepcopy(whole_side_table)
            side_task_table._dims[0] *= side_task_table_scalar
            side_task_table._pose._xyz[0] = (
                np.max(corners[:, 0]) - side_task_table.dims[0] / 2
            )
            self.tables.append(side_task_table)

            side_free_table = deepcopy(whole_side_table)
            side_free_table._dims[0] = (
                whole_side_table.dims[0] - side_task_table.dims[0]
            )
            side_free_table._pose._xyz[0] = (
                np.min(whole_side_table.corners[:, 0])
                + np.min(side_task_table.corners[:, 0])
            ) / 2
            self.clear_tables.append(side_free_table)

        # Setup table that robot is mounted to
        mount_table = Cuboid.random(
            center_range=[[-0.02, -0.02, -0.01], [0.02, 0.02, -0.01]],
            dimension_range=[
                [1, 0.9, 0.02],
                [1, 0.94, 0.02],
            ],
            quaternion=False,
        )
        xdim_mean = 2 * (np.min(front_task_table.corners[:, 0]) - mount_table.center[0])
        xdim = min(np.random.uniform(xdim_mean - 0.03, xdim_mean + 0.03), xdim_mean)
        mount_table._dims[0] = xdim_mean
        if has_side_table:
            ydim_mean = 2 * (
                mount_table.center[1] - np.max(side_task_table.corners[:, 1])
            )
            ydim = min(np.random.uniform(ydim_mean - 0.03, ydim_mean + 0.03), ydim_mean)
            mount_table._dims[1] = ydim_mean

        self.clear_tables.append(mount_table)

    def _gen_additional_candidate_sets(
        self, how_many: int, selfcc: FrankaSelfCollisionChecker
    ) -> List[List[TaskOrientedCandidate]]:
        """
        Problems in the tabletop environment are symmetric, meaning that all candidates
        generated on the surface of the table are valid start/end poses. However, not all
        environments are symmetric, so this function is implemented here to match the
        general environment interface. This creates two sets of `how_many` candidates that
        are intended to be used as start/end respectively. Take the cartesian product of these
        two sets and you'll have a bunch of valid problems.

        :param how_many int: How many candidates to generate in each candidate set (the result
                             is guaranteed to match this number or the function will run forever)
        :param selfcc FrankaSelfCollisionChecker: Checks for self collisions using spheres that
                                                  mimic the internal Franka collision checker.
        :rtype List[List[TaskOrientedCandidate]]: A list of candidate sets, where each has `how_many`
                                      candidates on the table.
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

    def gen_candidate(self, selfcc: FrankaSelfCollisionChecker, tool: Tool) -> Optional[TaskOrientedCandidate]:
        points = self.random_points_on_table(100)
        sim = Bullet(gui=False)
        sim.load_primitives(self.obstacles)
        gripper = sim.load_robot(FrankaGripper)
        arm = sim.load_robot(FrankaRobot)
            
        q = None
        pose = None
        for p in points:
            # Determine base height based on surface type
            base_height = p[2]
            for o in self.objects:
                if o.sdf(p) <= 0.01:
                    if isinstance(o, Cuboid):
                        base_height = o.center[2] + o.half_extents[2]
                    elif isinstance(o, Cylinder):
                        base_height = o.center[2] + o.height / 2
            
            # MODIFIED HEIGHT SAMPLING: Account for primitive length
            min_safe_height = np.random.uniform(0.03, 0.2)
            max_safe_height = min_safe_height + 0.15  # Up to 15cm above the primitive length

            # Sample height with bias toward lower values but ensuring primitive clearance
            p[2] = base_height + random_linear_decrease() * (max_safe_height - min_safe_height) + min_safe_height
            
            # MODIFIED ORIENTATION SAMPLING: Avoid pointing primitive into table
            # Keep roll mostly downward-facing but with tighter bounds
            roll = np.random.uniform(4 * np.pi / 5, 6 * np.pi / 5)  # 144°-216°
            
            # Constrain pitch to avoid horizontal orientations that might collide
            pitch = np.random.uniform(-np.pi / 12, np.pi / 12)  # ±15°
            
            # Yaw can be mostly free but avoid orientations where primitive hits objects
            yaw = np.random.uniform(-np.pi / 2, np.pi / 2)
            
            pose = SE3(xyz=p, so3=SO3.from_rpy(roll, pitch, yaw))
            # Check collision for gripper with attached primitive
            gripper.marionette(pose)

            # Check tool collision using the new helper method
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
            Generates a complex tool shape with randomized dimensions, offsets, and orientations
            for its constituent primitives to create more variety. It can now also generate
            simple, single-primitive tools like a bar, driller, or box.
            """
            # tool_type = np.random.choice(["T_shape", "L_shape", "U_shape", "bar", "driller"])
            tool_type = np.random.choice(["U_shape"])
            primitives = []

            # Helper to create small random rotations
            def random_rotation_quat():
                rpy = np.random.uniform(-np.pi / 18, np.pi / 18, 3)  # +/- 10 degrees
                # CORRECTED LINE: Changed .coeffs to .quaternion
                return SO3.from_rpy(rpy[0], rpy[1], rpy[2]).wxyz

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
                        "offset_quaternion": random_rotation_quat(),
                    },
                    { # Horizontal bar
                        "dims": bar_dims,
                        "offset": [
                            np.random.uniform(-0.01, 0.01),
                            np.random.uniform(-0.01, 0.01),
                            stem_dims[2] + bar_dims[2] / 2 - 0.01,
                        ],
                        "offset_quaternion": random_rotation_quat(),
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
                        "offset_quaternion": random_rotation_quat(),
                    },
                    { # Horizontal extension
                        "dims": horiz_dims,
                        "offset": [
                            horiz_dims[0] / 2 - 0.01,
                            0,
                            vert_dims[2] - horiz_dims[2] / 2,
                        ],
                        "offset_quaternion": random_rotation_quat(),
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
                        "offset_quaternion": random_rotation_quat(),
                    },
                    { # Left arm
                        "dims": arm_dims,
                        "offset": [
                            -base_width / 2 + arm_dims[0] / 2,
                            0,
                            arm_dims[2] / 2,
                        ],
                        "offset_quaternion": random_rotation_quat(),
                    },
                    { # Right arm
                        "dims": arm_dims,
                        "offset": [
                            base_width / 2 - arm_dims[0] / 2,
                            0,
                            arm_dims[2] / 2,
                        ],
                        "offset_quaternion": random_rotation_quat(),
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
                    "offset_quaternion": [1, 0, 0, 0],
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
                    "offset_quaternion": [1, 0, 0, 0],
                }]
                
            elif tool_type == "box":
                box_side = np.random.uniform(0.06, 0.09)
                box_dims = [box_side, box_side, box_side]
                primitives = [{
                    "dims": box_dims,
                    "offset": [0, 0, box_dims[2] / 2],
                    "offset_quaternion": [1, 0, 0, 0],
                }]

            self._tools = Tool(primitive_type="composite", primitives=primitives)
        
    def random_object(
        self, x: float, y: float, table_top: float, dim_min: float, dim_max: float
    ) -> Union[Cuboid, Cylinder]:
        """
        Generate a random object on the table top. If a cylinder, will always be oriented
        so that the round face is parallel to the tabletop.

        :param x float: The x position of the object in the world frame
        :param y float: The y position of the object in the world frame
        :param table_top float: The height of the tabletop
        :param dim_min float: The minimum value to use for either the radius or the x or y dimension
        :param dim_max float: The maximum value to use for either the radius or the x or y dimension.
                              The value is clamped at 0.15.
        :rtype Union[Cuboid, Cylinder]: The primitive
        """
        xy_dim_max = min(dim_max, 0.15)
        if np.random.rand() < 0.3:
            c = Cylinder.random(
                center_range=None,
                radius_range=[dim_min, xy_dim_max],
                height_range=[0.05, 0.35],
                quaternion=False,
            )
            c.center = [x, y, c.height / 2 + table_top]
        else:
            c = Cuboid.random(
                center_range=None,
                dimension_range=[
                    [dim_min, dim_min, 0.05],
                    [xy_dim_max, xy_dim_max, 0.35],
                ],
                quaternion=False,
            )
            c.center = [x, y, c.half_extents[2] + table_top]
            c.pose._so3 = SO3.from_rpy(0, 0, np.random.uniform(0, np.pi / 2))
        return c