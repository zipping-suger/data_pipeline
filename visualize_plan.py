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
from robofin.robots import FrankaRobot
from geometrout.primitive import Cuboid, Cylinder
from geometrout.transform import SE3

def load_obstacles_from_hdf5(file_path, trajectory_idx=0):
    """
    Load obstacles (cuboids and cylinders) from an HDF5 file
    """
    obstacles = []
    
    with h5py.File(file_path, 'r') as f:
        # Load cuboids
        if 'cuboid_dims' in f:
            cuboid_dims = f['cuboid_dims'][trajectory_idx]
            cuboid_centers = f['cuboid_centers'][trajectory_idx]
            cuboid_quats = f['cuboid_quaternions'][trajectory_idx]
            
            for i in range(len(cuboid_dims)):
                if np.all(cuboid_dims[i] == 0):
                    continue
                obstacles.append(Cuboid(
                    center=cuboid_centers[i],
                    dims=cuboid_dims[i],
                    quaternion=cuboid_quats[i]
                ))
        
        # Load cylinders
        if 'cylinder_radii' in f:
            cylinder_radii = f['cylinder_radii'][trajectory_idx]
            cylinder_heights = f['cylinder_heights'][trajectory_idx]
            cylinder_centers = f['cylinder_centers'][trajectory_idx]
            cylinder_quats = f['cylinder_quaternions'][trajectory_idx]
            
            for i in range(len(cylinder_radii)):
                if cylinder_radii[i][0] == 0 or cylinder_heights[i][0] == 0:
                    continue
                obstacles.append(Cylinder(
                    center=cylinder_centers[i],
                    radius=float(cylinder_radii[i][0]),
                    height=float(cylinder_heights[i][0]),
                    quaternion=cylinder_quats[i]
                ))
    
    return obstacles

def visualize_all_trajectories(file_path, start_idx=0, max_trajectories=None, delay=0.1, pause_between=1.0):
    """
    Visualize all trajectories from an HDF5 file
    
    Args:
        file_path: Path to the HDF5 file
        start_idx: Index to start visualization from
        max_trajectories: Maximum number of trajectories to visualize (None for all)
        delay: Time delay between frames (seconds)
        pause_between: Time to pause between trajectories (seconds)
    """
    # Setup PyBullet simulation (only once)
    sim = Bullet(gui=True)
    robot = sim.load_robot(FrankaRobot)
    
    # Open file and get number of trajectories
    with h5py.File(file_path, 'r') as f:
        if 'global_solutions' not in f:
            print(f"Error: No trajectories found in {file_path}")
            return
            
        num_trajectories = f['global_solutions'].shape[0]
        print(f"Found {num_trajectories} trajectories in {file_path}")
        
        # Determine end index
        end_idx = num_trajectories if max_trajectories is None else min(start_idx + max_trajectories, num_trajectories)
        
        # Visualize each trajectory
        for idx in range(start_idx, end_idx):
            # Clear previous obstacles
            sim.clear_all_obstacles()
            
            # Load obstacles for this trajectory
            obstacles = load_obstacles_from_hdf5(file_path, idx)
            sim.load_primitives(obstacles)
            
            # Load trajectory
            trajectory = f['global_solutions'][idx]
            
            # Print info
            print(f"\nTrajectory {idx}/{num_trajectories-1}:")
            print(f"  Shape: {trajectory.shape}")
            print(f"  Obstacles: {len(obstacles)}")
            
            # Execute trajectory
            for q in trajectory:
                robot.marionette(q)
                time.sleep(delay)
            
            # Pause between trajectories
            if idx < end_idx - 1:  # Don't pause after the last trajectory
                print(f"Pausing for {pause_between} seconds before next trajectory...")
                time.sleep(pause_between)
    
    print("\nAll trajectories visualized.")

def visualize_trajectory(file_path, trajectory_idx=0, delay=0.1):
    """
    Visualize a single trajectory from an HDF5 file
    """
    with h5py.File(file_path, 'r') as f:
        if 'global_solutions' not in f:
            print(f"Error: No trajectories found in {file_path}")
            return
            
        num_trajectories = f['global_solutions'].shape[0]
        if trajectory_idx >= num_trajectories:
            print(f"Error: Trajectory index {trajectory_idx} out of range (0-{num_trajectories-1})")
            return
            
        trajectory = f['global_solutions'][trajectory_idx]
    
    obstacles = load_obstacles_from_hdf5(file_path, trajectory_idx)
    
    sim = Bullet(gui=True)
    robot = sim.load_robot(FrankaRobot)
    sim.load_primitives(obstacles)
    
    print(f"Visualizing trajectory {trajectory_idx} from {file_path}")
    print(f"Trajectory shape: {trajectory.shape}")
    print(f"Number of obstacles: {len(obstacles)}")
    
    for q in trajectory:
        robot.marionette(q)
        time.sleep(delay)
    
    input("Press Enter to exit...")
    p.disconnect()

def main():
    parser = argparse.ArgumentParser(description='Visualize motion planning trajectories from HDF5 files')
    parser.add_argument('file_path', type=str, help='Path to the HDF5 file')
    parser.add_argument('--trajectory', '-t', type=int, default=None, 
                        help='Index of the trajectory to visualize (default: None, visualize all)')
    parser.add_argument('--delay', '-d', type=float, default=0.1,
                        help='Time delay between frames in seconds (default: 0.1)')
    parser.add_argument('--start', '-s', type=int, default=0,
                        help='Starting trajectory index when visualizing multiple (default: 0)')
    parser.add_argument('--max', '-m', type=int, default=None,
                        help='Maximum number of trajectories to visualize (default: all)')
    parser.add_argument('--pause', '-p', type=float, default=1.0,
                        help='Pause time between trajectories in seconds (default: 1.0)')
    args = parser.parse_args()
    
    if not Path(args.file_path).exists():
        print(f"Error: File {args.file_path} does not exist")
        return
    
    if args.trajectory is not None:
        visualize_trajectory(args.file_path, args.trajectory, args.delay)
    else:
        visualize_all_trajectories(
            args.file_path, 
            start_idx=args.start, 
            max_trajectories=args.max, 
            delay=args.delay,
            pause_between=args.pause
        )

if __name__ == "__main__":
    main()
