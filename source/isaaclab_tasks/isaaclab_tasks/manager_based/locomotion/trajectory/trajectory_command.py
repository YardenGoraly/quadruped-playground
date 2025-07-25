import torch
from dataclasses import dataclass, field

from isaaclab.assets import Articulation
from isaaclab.utils.configclass import configclass
from isaaclab.managers import CommandTerm, CommandTermCfg
import isaaclab.sim as sim_utils

@configclass
class TrajectoryCommandCfg(CommandTermCfg):
    asset_name: str = "robot"
    waypoints: list[list[float]] = field(default_factory=list)
    desired_speed: float = 1.0
    arrival_threshold: float = 0.5
    lookahead_distance: float = 0.8
    heading_control_stiffness: float = 0.8

    def __post_init__(self):
        """Post-initialization checks."""
        # Since this command is state-based, resampling is not used.
        # We set a default value here to satisfy the base class requirements.
        self.resampling_time_range = (1.0, 1.0)
    

class TrajectoryCommand(CommandTerm):
    cfg: TrajectoryCommandCfg

    def __init__(self, cfg: TrajectoryCommandCfg, env):
        super().__init__(cfg, env)

        self.device = env.device
        self._asset: Articulation = env.scene[cfg.asset_name]

        self.waypoints = torch.tensor(self.cfg.waypoints, device=self.device)
        self.num_waypoints = self.waypoints.shape[0]

        self.target_waypoint_idx = torch.ones(self.num_envs, dtype=torch.long, device=self.device)
        self.command_buffer = torch.zeros(self.num_envs, 3, device=self.device)

    def reset(self, env_ids: torch.Tensor):
        self.target_waypoint_idx[env_ids] = 1

    def compute(self, dt: float):
        robot_pos = self._asset.data.root_pos_w[:, :2]
        robot_yaw = self._asset.data.root_quat_w

        # For each env, get distance from robot to target waypoint
        target_waypoints = self.waypoints[self.target_waypoint_idx]
        distance_to_target = torch.linalg.norm(target_waypoints - robot_pos, dim=1)
        
        # If env arrived at a waypoint, update
        arrived_envs = torch.where(distance_to_target < self.cfg.arrival_threshold)[0]
        if arrived_envs.numel() > 0:
            self.target_waypoint_idx[arrived_envs] += 1
            self.target_waypoint_idx[arrived_envs] = torch.clamp(self.target_waypoint_idx[arrived_envs], max=self.num_waypoints - 1)

        prev_waypoint_idx = torch.clamp(self.target_waypoint_idx - 1, min=0)
        start_points = self.waypoints[prev_waypoint_idx]
        end_points = self.waypoints[self.target_waypoint_idx]

        traj_vecs = end_points - start_points
        traj_lens = torch.linalg.norm(traj_vecs, dim=1)
        nonzero_traj_lens = torch.where(traj_lens > 1e-6, traj_lens, torch.ones_like(traj_lens))
        traj_dirs = traj_vecs / nonzero_traj_lens.unsqueeze(1)

        vec_to_start = robot_pos - start_points
        proj_dist = torch.sum(vec_to_start * traj_dir, dim=1)
        proj_dist = torch.clamp(proj_dist, 0.0, traj_lens)
        closest_point_on_traj = start_points + proj_dist.unsqueeze(1) * traj_dirs
        lookahead_point = closest_point_on_traj + self.cfg.lookahead_distance * traj_dirs

        desired_velocity_world = lookahead_point - robot_pos
        desired_velocity_world_norm = torch.linalg.norm(desired_velocity_world, dim=1)
        nonzero_norm = torch.where(desired_velocity_world_norm > 1e-6, desired_velocity_world_norm, torch.ones_like(desired_velocity_world_norm))
        desired_velocity_world_dir = desired_velocity_world / nonzero_norm.unsqueeze(1)
        lin_vel_world = desired_velocity_world_dir * self.cfg.desired_speed

        _, _, yaw = sim_utils.get_euler_xyz_from_quaternion(robot_yaw)
        cos_yaw, sin_yaw = torch.cos(-yaw), torch.sin(-yaw)
        lin_vel_x_base = lin_vel_world[:, 0] * cos_yaw - lin_vel_world[:, 1] * sin_yaw
        lin_vel_y_base = lin_vel_world[:, 0] * sin_yaw + lin_vel_world[:, 1] * cos_yaw

        # Update the command buffer
        self.command_buffer[:, 0] = lin_vel_x_base
        self.command_buffer[:, 1] = lin_vel_y_base
        self.command_buffer[:, 2] = desired_heading

        return self.command_buffer
