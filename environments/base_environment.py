from abc import (
    ABC,
    abstractmethod,
)
from typing import List, Union, List, Any, Optional
from robofin.collision import FrankaSelfCollisionChecker
from geometrout.primitive import Cuboid, Cylinder
from dataclasses import dataclass, field
from geometrout.transform import SO3, SE3
import numpy as np
from prob_types import Obstacles


def radius_sample(center: float, radius: float):
    """
    Helper function to draw a uniform sample with a fixed radius around a center

    :param center float: The center of the distribution
    :param radius float: The radius of the distribution
    """
    return np.random.uniform(center - radius, center + radius)


@dataclass
class Tool:
    """
    Represents a tool in the environment
    """
    primitive_type: str
    dims: List[float]
    offset: List[float]
    offset_quaternion: List[float]


@dataclass
class Candidate:
    """
    Represents a configuration and the corresponding end-effector pose
    (in the right_gripper frame).
    """

    pose: SE3
    config: np.ndarray
    negative_volumes: Obstacles
    tool: Optional[Tool] = None


@dataclass
class TaskOrientedCandidate(Candidate):
    """
    Represents a configuration and the corresponding end-effector pose
    (in the right_gripper frame) for a task oriented pose.
    """

    pass


@dataclass
class NeutralCandidate(Candidate):
    """
    Represents a configuration and the corresponding end-effector pose
    (in the right_gripper frame) for a neutral pose.
    """

    pass


@dataclass
class FreeSpaceCandidate(Candidate):
    """
    Represents a configuration and the corresponding end-effector pose
    (in the right_gripper frame) for a free space candidate.
    A very random candidate that is not task oriented or neutral.
    """

    pass


class Environment(ABC):
    def __init__(self):
        self.generated = False
        self.demo_candidates = []
        self._tools = None  # Renamed from self.tools to self._tools
        pass

    @property
    @abstractmethod
    def obstacles(self) -> List[Union[Cuboid, Cylinder]]:
        """
        Returns all obstacles in the scene.
        :rtype List[Union[Cuboid, Cylinder]]: The list of obstacles in the scene
        """
        pass
    
    @property
    def tools(self) -> Optional[Tool]:
        return self._tools
    
    @property
    @abstractmethod
    def cuboids(self) -> List[Cuboid]:
        """
        Returns just the cuboids in the scene
        :rtype List[Cuboid]: The list of cuboids in the scene
        """
        pass

    @property
    @abstractmethod
    def cylinders(self) -> List[Cylinder]:
        """
        Returns just the cylinders in the scene
        :rtype List[Cylinder]: The list of cylinders in the scene
        """
        pass
    
    @staticmethod
    def _check_tool_collision(gripper_pose: SE3, obstacles: List[Union[Cuboid, Cylinder]], tool: Tool, buffer=0.03) -> bool:
        """
        Check if the tool (attached primitive) collides with any obstacles.
        
        :param gripper_pose: The pose of the gripper
        :param obstacles: List of obstacles to check against
        :return: True if collision detected, False otherwise
        """
        num_surface_points = 50  # Points on the surface
        
        # Create offset transformation relative to gripper frame
        offset_transform = SE3(
            xyz=tool.offset, 
            so3=SO3(quaternion=tool.offset_quaternion)
        )
        
        # Combine with the gripper pose: primitive_pose = gripper_pose * offset_transform
        primitive_pose = gripper_pose @ offset_transform
        
        # Create the primitive cuboid at the correct pose
        primitive_cuboid = Cuboid(
            center=primitive_pose.xyz,
            dims=tool.dims,
            quaternion=primitive_pose.so3.wxyz
        )
        
        # Sample multiple points in the volume for collision checking
        primitive_corners = primitive_cuboid.corners
        surface_points = primitive_cuboid.sample_surface(num_surface_points, noise=0.0)
        all_check_points = np.vstack([primitive_corners, surface_points])
        
        # Check collision with all obstacles
        for obstacle in obstacles:
            # Check all sampled points for collision
            for point in all_check_points:
                if obstacle.sdf(point) < buffer:  # Buffer: point is too close or inside
                    return True

            # Additional check: test the center as well
            center_sdf = obstacle.sdf(primitive_pose.xyz)
            if center_sdf < buffer:
                return True
            
        return False

    def gen(self, selfcc: FrankaSelfCollisionChecker, **kwargs: Any) -> bool:
        """
        Generates an environment and a pair of start/end candidates

        :param selfcc FrankaSelfCollisionChecker: Checks for self collisions using spheres that
                                                  mimic the internal Franka collision checker.
        :rtype bool: Whether the environment was successfully generated
        """
        self.generated = self._gen(selfcc, **kwargs)
        if self.generated:
            assert len(self.demo_candidates) == 2
            cand1, cand2 = self.demo_candidates
            assert cand1 is not None and cand2 is not None
        return self.generated

    def gen_additional_candidate_sets(
        self, how_many: int, selfcc: FrankaSelfCollisionChecker
    ) -> List[List[TaskOrientedCandidate]]:
        """
        This creates two sets of `how_many` candidates that
        are intended to be used as start/end respectively. Take the cartesian product of these
        two sets and you'll have a bunch of valid problems.

        :param how_many int: How many candidates to generate in each candidate set (the result
                             is guaranteed to match this number or the function will run forever)
        :param selfcc FrankaSelfCollisionChecker: Checks for self collisions using spheres that
                                                  mimic the internal Franka collision checker.
        :rtype List[List[TaskOrientedCandidate]]: A list of candidate sets, where each has `how_many`
                                      candidates on the table.
        """
        assert (
            self.generated
        ), "Must run generate the environment before requesting additional candidates"
        return self._gen_additional_candidate_sets(how_many, selfcc)

    def gen_neutral_candidates(
        self, how_many: int, selfcc: FrankaSelfCollisionChecker
    ) -> List[NeutralCandidate]:
        """
        Generate a set of collision free neutral poses and corresponding configurations
        (represented as Candidate object)

        :param how_many int: How many neutral poses to generate
        :param selfcc FrankaSelfCollisionChecker: Checks for self collisions using spheres that
                                                  mimic the internal Franka collision checker.
        :rtype List[NeutralCandidate]: A list of neutral poses
        """
        assert (
            self.generated
        ), "Must run generate the environment before requesting additional candidates"
        return self._gen_neutral_candidates(how_many, selfcc)
    
    def gen_free_space_candidates(
        self, how_many: int, selfcc: FrankaSelfCollisionChecker
    ) -> List[FreeSpaceCandidate]:
        """
        Generate a set of collision free free space poses and corresponding configurations
        (represented as FreeSpaceCandidate object)

        :param how_many int: How many free space poses to generate
        :param selfcc FrankaSelfCollisionChecker: Checks for self collisions using spheres that
                                                  mimic the internal Franka collision checker.
        :rtype List[FreeSpaceCandidate]: A list of free space poses
        """
        assert (
            self.generated
        ), "Must run generate the environment before requesting additional candidates"
        return self._gen_free_space_candidates(how_many, selfcc)

    @abstractmethod
    def _gen(self, selfcc: FrankaSelfCollisionChecker) -> bool:
        """
        The internal implementation of the gen function.

        :param selfcc FrankaSelfCollisionChecker: Checks for self collisions using spheres that
                                                  mimic the internal Franka collision checker.
        :rtype bool: Whether the environment was successfully generated
        """
        pass

    @abstractmethod
    def _gen_additional_candidate_sets(
        self, how_many: int, selfcc: FrankaSelfCollisionChecker
    ) -> List[List[TaskOrientedCandidate]]:
        """
        This creates two sets of `how_many` candidates that
        are intended to be used as start/end respectively. Take the cartesian product of these
        two sets and you'll have a bunch of valid problems.

        :param how_many int: How many candidates to generate in each candidate set (the result
                             is guaranteed to match this number or the function will run forever)
        :param selfcc FrankaSelfCollisionChecker: Checks for self collisions using spheres that
                                                  mimic the internal Franka collision checker.
        :rtype List[List[TaskOrientedCandidate]]: A list of candidate sets, where each has `how_many`
                                      candidates on the table.
        """
        pass

    @abstractmethod
    def _gen_neutral_candidates(
        self, how_many: int, selfcc: FrankaSelfCollisionChecker
    ) -> List[NeutralCandidate]:
        """
        Generate a set of collision free neutral poses and corresponding configurations
        (represented as NeutralCandidate object)

        :param how_many int: How many neutral poses to generate
        :param selfcc FrankaSelfCollisionChecker: Checks for self collisions using spheres that
                                                  mimic the internal Franka collision checker.
        :rtype List[NeutralCandidate]: A list of neutral poses
        """
        pass

    @abstractmethod
    def _gen_free_space_candidates(
        self, how_many: int, selfcc: FrankaSelfCollisionChecker
    ) -> List[FreeSpaceCandidate]:
        """
        Generate a set of collision free free space poses and corresponding configurations
        (represented as FreeSpaceCandidate object)

        :param how_many int: How many free space poses to generate
        :param selfcc FrankaSelfCollisionChecker: Checks for self collisions using spheres that
                                                  mimic the internal Franka collision checker.
        :rtype List[FreeSpaceCandidate]: A list of free space poses
        """
        pass
