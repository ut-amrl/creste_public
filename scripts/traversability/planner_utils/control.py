# This is template code for generating ackeermann steering trajectories
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import directed_hausdorff
from scipy.interpolate import make_interp_spline

# set rnadom seed
# np.random.seed(1337)


def getControls(s, c, v, w, dt=0.1):
    """
    Inputs:
        c - curvature   [Tx3]
        v - speed       [T,]
        w - wheelbase length [1]
    """
    xdot = v * np.cos(s[:, 2])
    ydot = v * np.sin(s[:, 2])
    thetadot = v * c

    dx = xdot * dt
    dy = ydot * dt
    dtheta = thetadot * dt

    return np.stack([dx, dy, dtheta], axis=-1)


def sampleRange(size, min=-1, max=1):
    return (np.random.rand(size) * (max - min)) + min


def hausdorffDistance(trajectories, expert_idx=0):
    """
    Compute the average Hausdorff distance between the first trajectory
    and all other trajectories.

    Args:
    - trajectories (np.array): A NumPy array of shape [#steps, #trajectories, 3] 
                               where 3 represents (x, y, theta).

    Returns:
    - average_hausdorff (float): The average Hausdorff distance between the first trajectory
                                 and the rest of the trajectories.
    """
    # Get the number of trajectories
    num_trajectories, num_steps, _ = trajectories.shape

    # Extract the first trajectory as the reference
    ref_trajectory = trajectories[expert_idx, :, :]  # [#steps, 3]

    # Initialize a list to store the Hausdorff distances
    hausdorff_distances = []

    # Compute the Hausdorff distance between the first trajectory and all others
    for i in range(0, num_trajectories):
        other_trajectory = trajectories[i, :, :]  # [#steps, 3]

        # Compute directed Hausdorff distance in both directions
        d_1 = directed_hausdorff(ref_trajectory, other_trajectory)[0]
        d_2 = directed_hausdorff(other_trajectory, ref_trajectory)[0]

        # The true Hausdorff distance is the max of the two directed distances
        hausdorff_distance = max(d_1, d_2)

        # Store the Hausdorff distance
        hausdorff_distances.append(hausdorff_distance)

    # Compute the average of the Hausdorff distances
    distances = np.array(hausdorff_distances)

    return distances

def sampleEpsilonTrajectory(expert_traj, num_traj, num_iter, num_samples, epsilon):
    """ Samples n points along expert trajectory and perturbs them by epsilon min max range"""
    DEGREES=3
    # Divide [0, epsilon] into num_traj equal parts
    perturb_range = np.linspace(0, epsilon, num_traj//2+1)
    perturb_pairs = np.array([[perturb_range[i], perturb_range[i+1]] for i in range(len(perturb_range)-1)])

    spline_traj = np.zeros((num_traj, num_iter, 3))
    for pair_idx in range(len(perturb_pairs)):
        ppair = perturb_pairs[pair_idx]
        path_left = perturb_path(expert_traj[0], side="left", magnitude=ppair, num_points=num_samples)
        path_right = perturb_path(expert_traj[0], side="right", magnitude=ppair, num_points=num_samples)
        # sort by x
        path_left = path_left[path_left[:, 0].argsort()]
        path_right = path_right[path_right[:, 0].argsort()]

        # Create a spline interpolation of the path
        tspline_left = make_interp_spline(path_left[:, 0], path_left[:, 1], k=DEGREES)
        tspline_right = make_interp_spline(path_right[:, 0], path_right[:, 1], k=DEGREES)
        spline_traj[2*pair_idx, :, 0] = np.linspace(path_left[0, 0], path_left[-1, 0], num_iter)
        spline_traj[2*pair_idx, :, 1] = tspline_left(spline_traj[pair_idx, :, 0])
        spline_traj[2*pair_idx+1, :, 0] = np.linspace(path_right[0, 0], path_right[-1, 0], num_iter)
        spline_traj[2*pair_idx+1, :, 1] = tspline_right(spline_traj[2*pair_idx+1, :, 0])

    return spline_traj

def sampleTrajectory(num_traj, num_iter, cmin, cmax, vmin, vmax, w, dt, epsilon=10):
    trajectories = np.tile(
        np.array([[0, 0, 0]]), (num_traj, 1, 1))  # x, y, theta

    for it in range(0, num_iter-1):
        # Stochastically sample curvature and velocity
        c = sampleRange((num_traj), cmin, cmax)
        v = sampleRange((num_traj), vmin, vmax)

        next_state = trajectories[:, it] + \
            getControls(trajectories[:, it], c, v, w, dt=dt)
        next_state = next_state[:, None]
        trajectories = np.concatenate((trajectories, next_state), axis=1)

    return trajectories

def transformToLocal(trajectories, center=(12.8, 12.8), res=0.1):
    """Transforms trajectories from x,y,theta in bev cordinates to (x,y) in meters."""
    B, T, _ = trajectories.shape  # B: Batch size, T: Number of time steps

    T_world_ego = np.eye(3)
    T_world_ego[:2, 2] = [c / res for c in center]
    T_world_ego[:2, :2] = np.array([[-1, 0], [0, -1]])

    trajectories_homo = np.ones((B, T, 3))
    trajectories_homo[:, :, :2] = trajectories[:, :, :2]

    transformed = np.matmul(trajectories_homo, T_world_ego.T)
    transformed[:, :, :2] = transformed[:, :, :2] * res
    return transformed

def transformToBEV(trajectories, center=(12.8, 12.8), res=0.1):
    """Transforms trajectories from (x,y,theta) in meters to x,y in BEV pixel coordinates.
    Also centers points at center of image."""

    B, T, _ = trajectories.shape  # B: Batch size, T: Number of time steps

    T_ego_world = np.eye(3)
    T_ego_world[:2, 2] = center
    T_ego_world[:2, :2] = np.array([[-1, 0], [0, -1]])

    trajectories_homo = np.ones((B, T, 3))
    # Copy (x, y), ignore theta for now
    trajectories_homo[:, :, :2] = trajectories[:, :, :2]

    # Perform batched matrix multiplication
    # Result is (B, T, 3)
    transformed = np.matmul(trajectories_homo, T_ego_world.T)
    transformed = transformed / res

    # Return only the (x, y) coordinates in BEV space
    return transformed[:, :, :2]  # Return transformed (x, y) BEV coordinates

def perturb_path(path, side="left", magnitude=[0.8, 1.0], num_points=10):
    """
    Perturb random points on one side of the true path.
    
    Args:
        path (np.array) [T, 3] x y theta
        side (str): Side to perturb points ('left' or 'right').
        magnitude (float): Maximum distance for the perturbation.
        num_points (int): Number of points to perturb.
    
    Returns:
        np.array: Perturbed path as an array of shape (N, 2).
    """
    T, _ = path.shape
    perturbed_path = []
    indices = np.linspace(0, len(path), num_points+2).astype(int)
    indices = indices[1:-1]

    for idx in indices:
        # Calculate direction vector
        direction_vector = path[idx + 1] - path[idx]
        direction_vector /= np.linalg.norm(direction_vector)
        # Calculate perpendicular vector
        perpendicular_vector = np.array([-direction_vector[1], direction_vector[0]])
        if side == "right":
            perpendicular_vector = -perpendicular_vector

        # Apply random perturbation along perpendicular direction
        perturbation = np.random.uniform(magnitude[0], magnitude[1]) * perpendicular_vector
        perturbed_pt = [path[idx][0] + perturbation[0], path[idx][1] + perturbation[1], path[idx][2]]
        perturbed_path.append(perturbed_pt)

    perturbed_path = [path[0].tolist()] + perturbed_path + [path[-1].tolist()]
    return np.array(perturbed_path)

def main():
    w = 1.0     # wheelbase length
    cmin = -2
    cmax = 2
    vmax = 1       # m / s
    num_vel = 1
    num_curv = 20
    num_traj = num_vel * num_curv
    dt = 0.1    # s
    num_iter = 50  # ul
    epsilon = 1
    traj_config = {
        # "num_vel": num_vel,
        # "num_curv": num_curv,
        "num_traj": num_traj,
        "num_iter": num_iter,
        "cmin": cmin,
        "cmax": cmax,
        "vmin": vmax,
        "vmax": vmax,
        "w": w,
        "dt": dt
    }

    # Create dummy straight line trajectories
    expert_traj = np.zeros((1, num_iter, 3))
    expert_traj[:, :, 0] = np.linspace(0, 10.8, num_iter)
    # traj_config['expert_traj'] = expert_traj[0]
    # trajectories = sampleEpsilonTrajectory(**traj_config)
    path_points = perturb_path(expert_traj[0], side="left", magnitude=[0.1, 0.4], num_points=3)

    from planner_utils.planner_original import AStarPlanner, Node
    map_size = [25.6, 25.6]
    map_res = [0.1, 0.1]
    grid_size = [int(map_size[0] / map_res[0]), int(map_size[1] / map_res[1])]
    # cost_map = np.zeros(grid_size)
    cost_map = np.zeros(grid_size)
    planner_configs = {
        "cost_map": cost_map,
        "cost_weight": 2,
        "map_size": map_size,
        "map_res": map_res,
        "max_v": 1.0,
        "max_w": np.pi/2,
        "max_dv": 0.2,
        "max_dw": np.pi/4,
        "partitions": 11,
        "planning_dt": 1.0,
        "heuristic_multiplier": 0,
        "linear_acc_weight": 0,
        "angular_acc_weight": 0,
        "random_motion_cost": False,
        "heuristic_annealing": 1
    }
    # planner = AStarPlanner(**planner_configs)
    # start = Node(x=path_points[0][0], y=path_points[0][1], theta=0, t_idx=0, v=0.2, w=0)
    # full_path = []
    # for point_idx in range(1, len(path_points)):
    #     goal = Node(path_points[point_idx][0], path_points[point_idx][1], 0, t_idx=0, v=0, w=0)
    #     path = planner.plan(start, goal, max_time=0, goal_radius=0.1, max_expand_node_num=100000)
    #     print("Start: ", start.x, start.y)
    #     print("Goal: ", goal.x, goal.y)
    #     if path is None:
    #         print("No path found")
    #         import pdb; pdb.set_trace()
    #         continue
    #     full_path.extend(path)
    #     start = Node(x=path[-1][0], y=path[-1][1], theta=path[-1][2], t_idx=0, v=0.2, w=0)
    # trajectories = np.array([[node[0], node[1], node[2]] for node in full_path])[None,...]
    # print("Full path: ", full_path)
    trajectories = path_points[None,...]
    unique_ids = np.unique(trajectories[:, :, 0], return_index=True)[1]
    trajectories = trajectories[:, unique_ids]
    tspline = make_interp_spline(trajectories[0, :, 0], trajectories[0, :, 1], k=3)
    spline_traj = np.zeros((1, num_iter, 3))
    spline_traj[0, :, 0] = np.linspace(trajectories[0, 0, 0], trajectories[0, -1, 0], num_iter)
    spline_traj[0, :, 1] = tspline(spline_traj[0, :, 0])
    trajectories = spline_traj

    # trajectories = sampleTrajectory(**traj_config)
    # Use first trajectory as expert
    distances = hausdorffDistance(trajectories, expert_idx=0)  # [N,]
    # trajectories = transformToBEV(trajectories)

    # Plot states
    plt.figure()
    for tidx in range(1):
        if tidx == 0:
            # This is the expert trajectory
            label = "Expert"
        else:
            # This is a perturbed trajectory
            label = f"Trajectory {tidx}"

        if distances[tidx] < epsilon:
            plt.plot(trajectories[tidx, :, 0],
                     trajectories[tidx, :, 1], '-o', label=label)
        else:
            print(f"Exceeded epsilon for trajectory {tidx}")
            plt.plot(trajectories[tidx, :, 0],
                     trajectories[tidx, :, 1], '-x', label=label)
        
        # Plot expert trajectories
        plt.plot(expert_traj[0, :, 0], expert_traj[0, :, 1], '-o', label="Expert")

    plt.xlabel('m')
    plt.ylabel('m')
    plt.legend()  # Add legend to the plot
    plt.savefig("trajectories.png")


if __name__ == "__main__":
    main()
