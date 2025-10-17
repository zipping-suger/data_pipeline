#!/usr/bin/env python3

import os
import sys
import h5py
import time
import argparse
import numpy as np
import pybullet as p
from pathlib import Path

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robofin.bullet import Bullet, BulletFranka
from robofin.robots import FrankaRobot, FrankaGripper
from geometrout.primitive import Cuboid, Cylinder
from geometrout.transform import SE3, SO3


def load_obstacles_from_hdf5(file_path, trajectory_idx=0):
    """
    Load obstacles (cuboids and cylinders) from an HDF5 file
    """
    obstacles = []

    with h5py.File(file_path, "r") as f:
        # Load cuboids
        if "cuboid_dims" in f:
            cuboid_dims = f["cuboid_dims"][trajectory_idx]
            cuboid_centers = f["cuboid_centers"][trajectory_idx]
            cuboid_quats = f["cuboid_quaternions"][trajectory_idx]

            for i in range(len(cuboid_dims)):
                if np.all(cuboid_dims[i] == 0):
                    continue
                obstacles.append(
                    Cuboid(
                        center=cuboid_centers[i],
                        dims=cuboid_dims[i],
                        quaternion=cuboid_quats[i],
                    )
                )

        # Load cylinders
        if "cylinder_radii" in f:
            cylinder_radii = f["cylinder_radii"][trajectory_idx]
            cylinder_heights = f["cylinder_heights"][trajectory_idx]
            cylinder_centers = f["cylinder_centers"][trajectory_idx]
            cylinder_quats = f["cylinder_quaternions"][trajectory_idx]

            for i in range(len(cylinder_radii)):
                if cylinder_radii[i][0] == 0 or cylinder_heights[i][0] == 0:
                    continue
                obstacles.append(
                    Cylinder(
                        center=cylinder_centers[i],
                        radius=float(cylinder_radii[i][0]),
                        height=float(cylinder_heights[i][0]),
                        quaternion=cylinder_quats[i],
                    )
                )

    return obstacles


def load_tool_from_hdf5(file_path, trajectory_idx=0, tool_type="start"):
    """
    Load tool primitives from an HDF5 file
    
    Args:
        file_path: Path to the HDF5 file
        trajectory_idx: Index of the trajectory
        tool_type: Either "start" or "target" tool
        
    Returns:
        List of tool primitives as dictionaries with keys: dims, offset, quaternion
    """
    tool_primitives = []
    
    with h5py.File(file_path, "r") as f:
        # Check if tool data exists
        dims_key = f"{tool_type}_tool_dims"
        offset_key = f"{tool_type}_tool_offset" 
        quat_key = f"{tool_type}_tool_quaternion"
        num_primitives_key = f"{tool_type}_tool_num_primitives"
        
        if (dims_key in f and offset_key in f and 
            quat_key in f and num_primitives_key in f):
            
            num_primitives = int(f[num_primitives_key][trajectory_idx])
            
            if num_primitives > 0:
                dims = f[dims_key][trajectory_idx]
                offsets = f[offset_key][trajectory_idx]
                quats = f[quat_key][trajectory_idx]
                
                for i in range(num_primitives):
                    primitive = {
                        "dims": dims[i],
                        "offset": offsets[i],
                        "quaternion": quats[i]
                    }
                    tool_primitives.append(primitive)
    
    return tool_primitives


def visualize_tool(sim, gripper_pose, tool_primitives, color=[0.8, 0.2, 0.2, 1.0]):
    """
    Visualize a tool composed of multiple primitives attached to the gripper
    
    Args:
        sim: Bullet simulation instance
        gripper_pose: SE3 pose of the gripper
        tool_primitives: List of tool primitive dictionaries
        color: RGBA color for the tool visualization
    """
    tool_body_ids = []
    
    for primitive in tool_primitives:
        # Create the primitive pose relative to gripper
        offset_pose = SE3(
            xyz=primitive["offset"],
            so3=SO3(quaternion=primitive["quaternion"])
        )
        
        # Transform to world frame
        primitive_pose = gripper_pose @ offset_pose
        
        # Create and load the primitive
        cuboid = Cuboid(
            center=primitive_pose.xyz,
            dims=primitive["dims"],
            quaternion=primitive_pose.so3.wxyz
        )
        
        # Use load_primitive (singular) for single primitive
        body_id = sim.load_cuboid(cuboid, color=color)
        tool_body_ids.append(body_id)
    
    return tool_body_ids


def visualize_all_trajectories(
    file_path,
    start_idx=0,
    max_trajectories=None,
    delay=0.1,
    pause_between=1.0,
    key=None,
    show_tool=True,
):
    """
    Visualize all trajectories from an HDF5 file

    Args:
        file_path: Path to the HDF5 file
        start_idx: Index to start visualization from
        max_trajectories: Maximum number of trajectories to visualize (None for all)
        delay: Time delay between frames (seconds)
        pause_between: Time to pause between trajectories (seconds)
        key: The HDF5 key for the trajectory data (e.g., 'global_solutions')
        show_tool: Whether to visualize the tool attached to the gripper
    """
    # Setup PyBullet simulation (only once)
    sim = Bullet(gui=True)
    robot = sim.load_robot(FrankaRobot)
    gripper = sim.load_robot(FrankaGripper)

    # Open file and get number of trajectories
    with h5py.File(file_path, "r") as f:
        # Determine the correct key for trajectories
        trajectory_key = key
        if trajectory_key is None:
            # Auto-detect if no key is provided
            if "hybrid_solutions" in f:
                trajectory_key = "hybrid_solutions"
            elif "global_solutions" in f:
                trajectory_key = "global_solutions"
            else:
                print(
                    f"Error: No 'hybrid_solutions' or 'global_solutions' key found in {file_path}"
                )
                print("Please specify a key with the --key argument.")
                return
        elif trajectory_key not in f:
            # Handle case where the specified key doesn't exist
            print(f"Error: Specified key '{trajectory_key}' not found in {file_path}")
            return

        num_trajectories = f[trajectory_key].shape[0]
        print(
            f"Found {num_trajectories} trajectories in {file_path} under key '{trajectory_key}'"
        )

        # Determine end index
        end_idx = (
            num_trajectories
            if max_trajectories is None
            else min(start_idx + max_trajectories, num_trajectories)
        )

        # Visualize each trajectory
        for idx in range(start_idx, end_idx):
            # Clear previous obstacles and tools
            sim.clear_all_obstacles()

            # Load obstacles for this trajectory
            obstacles = load_obstacles_from_hdf5(file_path, idx)
            sim.load_primitives(obstacles)

            # Load tool primitives for this trajectory
            start_tool_primitives = []
            target_tool_primitives = []
            if show_tool:
                start_tool_primitives = load_tool_from_hdf5(file_path, idx, "start")
                target_tool_primitives = load_tool_from_hdf5(file_path, idx, "target")
                
                print(f"  Start tool primitives: {len(start_tool_primitives)}")
                print(f"  Target tool primitives: {len(target_tool_primitives)}")

            # Load trajectory
            trajectory = f[trajectory_key][idx]

            # Print info
            print(f"\nTrajectory {idx}/{num_trajectories-1}:")
            print(f"  Shape: {trajectory.shape}")
            print(f"  Obstacles: {len(obstacles)}")

            # Execute trajectory
            tool_body_ids = []
            for q in trajectory:
                robot.marionette(q)
                
                # Update gripper pose and visualize tool if enabled
                if show_tool and (start_tool_primitives or target_tool_primitives):
                    # Clear previous tool visualization
                    for body_id in tool_body_ids:
                        p.removeBody(body_id)
                    tool_body_ids = []
                    
                    # Get current gripper pose
                    gripper_pose = FrankaRobot.fk(q, eff_frame="right_gripper")
                    gripper.marionette(gripper_pose)
                    
                    # Visualize start tool (red)
                    if start_tool_primitives:
                        tool_body_ids.extend(
                            visualize_tool(sim, gripper_pose, start_tool_primitives, [0.8, 0.2, 0.2, 1.0])
                        )
                    
                    # For target tool, we might want to visualize it differently
                    # or at the target pose. For now, we'll skip target tool visualization
                    # during trajectory execution since it's attached to target candidate
                
                time.sleep(delay)

            # Clear tool visualization at the end of trajectory
            for body_id in tool_body_ids:
                p.removeBody(body_id)

            # Pause between trajectories
            if idx < end_idx - 1:  # Don't pause after the last trajectory
                print(f"Pausing for {pause_between} seconds before next trajectory...")
                time.sleep(pause_between)

    print("\nAll trajectories visualized.")


def visualize_trajectory(file_path, trajectory_idx=0, delay=0.1, key=None, show_tool=True):
    """
    Visualize a single trajectory from an HDF5 file
    """
    trajectory_key = None
    with h5py.File(file_path, "r") as f:
        # Determine the correct key for trajectories
        trajectory_key = key
        if trajectory_key is None:
            # Auto-detect if no key is provided
            if "hybrid_solutions" in f:
                trajectory_key = "hybrid_solutions"
            elif "global_solutions" in f:
                trajectory_key = "global_solutions"
            else:
                print(
                    f"Error: No 'hybrid_solutions' or 'global_solutions' key found in {file_path}"
                )
                print("Please specify a key with the --key argument.")
                return
        elif trajectory_key not in f:
            # Handle case where the specified key doesn't exist
            print(f"Error: Specified key '{trajectory_key}' not found in {file_path}")
            return

        num_trajectories = f[trajectory_key].shape[0]
        if trajectory_idx >= num_trajectories:
            print(
                f"Error: Trajectory index {trajectory_idx} out of range (0-{num_trajectories-1}) for key '{trajectory_key}'"
            )
            return

        trajectory = f[trajectory_key][trajectory_idx]

    obstacles = load_obstacles_from_hdf5(file_path, trajectory_idx)
    start_tool_primitives = load_tool_from_hdf5(file_path, trajectory_idx, "start")
    target_tool_primitives = load_tool_from_hdf5(file_path, trajectory_idx, "target")

    sim = Bullet(gui=True)
    robot = sim.load_robot(FrankaRobot)
    gripper = sim.load_robot(FrankaGripper)
    sim.load_primitives(obstacles)

    print(
        f"Visualizing trajectory {trajectory_idx} (from key '{trajectory_key}') from {file_path}"
    )
    print(f"Trajectory shape: {trajectory.shape}")
    print(f"Number of obstacles: {len(obstacles)}")
    print(f"Start tool primitives: {len(start_tool_primitives)}")
    print(f"Target tool primitives: {len(target_tool_primitives)}")

    tool_body_ids = []
    for q in trajectory:
        robot.marionette(q)
        
        # Update tool visualization if enabled
        if show_tool and start_tool_primitives:
            # Clear previous tool visualization
            for body_id in tool_body_ids:
                p.removeBody(body_id)
            tool_body_ids = []
            
            # Get current gripper pose and visualize tool
            gripper_pose = FrankaRobot.fk(q, eff_frame="right_gripper")
            gripper.marionette(gripper_pose)
            tool_body_ids.extend(
                visualize_tool(sim, gripper_pose, start_tool_primitives, [0.8, 0.2, 0.2, 1.0])
            )
        
        time.sleep(delay)

    # Clean up tool visualization
    for body_id in tool_body_ids:
        p.removeBody(body_id)

    input("Press Enter to exit...")
    p.disconnect()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize motion planning trajectories from HDF5 files"
    )
    parser.add_argument("file_path", type=str, help="Path to the HDF5 file")
    parser.add_argument(
        "--trajectory",
        "-t",
        type=int,
        default=None,
        help="Index of the trajectory to visualize (default: None, visualize all)",
    )
    parser.add_argument(
        "--key",
        "-k",
        type=str,
        default=None,
        help="Explicitly specify the trajectory key (e.g., global_solutions). If not provided, it will auto-detect.",
    )
    parser.add_argument(
        "--delay",
        "-d",
        type=float,
        default=0.1,
        help="Time delay between frames in seconds (default: 0.1)",
    )
    parser.add_argument(
        "--start",
        "-s",
        type=int,
        default=0,
        help="Starting trajectory index when visualizing multiple (default: 0)",
    )
    parser.add_argument(
        "--max",
        "-m",
        type=int,
        default=None,
        help="Maximum number of trajectories to visualize (default: all)",
    )
    parser.add_argument(
        "--pause",
        "-p",
        type=float,
        default=1.0,
        help="Pause time between trajectories in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--no-tool",
        action="store_true",
        help="Disable tool visualization",
    )
    args = parser.parse_args()

    if not Path(args.file_path).exists():
        print(f"Error: File {args.file_path} does not exist")
        return

    show_tool = not args.no_tool

    if args.trajectory is not None:
        visualize_trajectory(args.file_path, args.trajectory, args.delay, key=args.key, show_tool=show_tool)
    else:
        visualize_all_trajectories(
            args.file_path,
            start_idx=args.start,
            max_trajectories=args.max,
            delay=args.delay,
            pause_between=args.pause,
            key=args.key,
            show_tool=show_tool,
        )


if __name__ == "__main__":
    main()