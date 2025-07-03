from dataclasses import dataclass, field
from typing import List, Union, Optional, Dict, Sequence
import numpy as np
from geometrout.transform import SE3
from geometrout.primitive import Cuboid, Cylinder, Sphere

Obstacles = List[Union[Cuboid, Cylinder, Sphere]]
Trajectory = Sequence[Union[Sequence, np.ndarray]]


@dataclass
class PlanningProblem:
    """
    Defines a common interface to describe planning problems
    """

    target: SE3  # The target in the `right_gripper` frame
    target_volume: Union[Cuboid, Cylinder]
    q0: np.ndarray  # The starting configuration
    obstacles: Optional[Obstacles] = None  # The obstacles in the scene
    obstacle_point_cloud: Optional[np.ndarray] = None
    target_negative_volumes: Obstacles = field(default_factory=lambda: [])


ProblemSet = Dict[str, Dict[str, List[PlanningProblem]]]
