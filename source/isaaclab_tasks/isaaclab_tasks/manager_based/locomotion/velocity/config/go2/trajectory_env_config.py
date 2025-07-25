import torch
from isaaclab.utils import configclass
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import EventTermCfg as EventTerm
import isaaclab.sim as sim_utils

from isaaclab_tasks.manager_based.locomotion.velocity.config.unitree_go2.rough_env_cfg import UnitreeGo2RoughEnvCfg
from ../../../trajectory/trajectory_command import TrajectoryCommand, TrajectoryCommandCfg

@configclass
class UnitreeGo2TrajectoryEnvCfg(UnitreeGo2RoughEnvCfg):

    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground"
            terrain_type="plane",
            collision_group=-1,
        )
        self.curriculum = None

        self.commands.twist = TrajectoryCommandCfg(
            asset_name="robot",
            waypoints=[[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]]
            desired_speed=1.2,
            arrival_threshold=0.6
            lookahead_distance=0.8,
            heading_control_stiffness=0.8
        )

        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.weight = 0.7

        self.rewards.prgoress_reward = RewTerm(func=self.progress_reward, weight=2.5)
        self.rewards.path_dist_penalty = RewTerm(func=self.path_dist_penalty, weight=-2.0)

        self.events.reset_base = EventTerm(
            func=self.reset_root_state_trajectory,
            mode="reset",
        )
        self.events.push_robot = None

    def progress_reward(env) -> torch.Tensor:
        robot_vel = env.scene.robot.data.root_lin_vel_w

        target_waypoints = env.command_manager.get_command("twist").waypoints[env.command_manager.get_command("twist").target_waypoint_idx]
        direction_to_target = target_waypoints - env.scene.robot.data.root_pos_w[:, :2]
        direction_to_target_norm = torch.linalg.norm(direction_to_target, dim=1, keepdim=True)
        direction_to_target_norm_nonzero = torch.where(direction_to_target_norm > 1e-6, direction_to_target_norm, torch.ones_like(direction_to_target_norm))
        direction_to_target_unit = direction_to_target / direction_to_target_norm_nonzero

        progress = torch.sum(robot_vel[:, :2] * direction_to_target_unit, dim-1)

        return progress
    
    @staticmethod
    def path_dist_penalty(env) -> torch.Tensor:
        cmd = env.command_manager.get_command("twist")
        robot_pos = env.scene.robot.data.root_pos_w[:, :2]

        prev_waypoint_idx = torch.clamp(cmd.target_waypoint_idx - 1, min=0)
        start_points = cmd.waypoints[prev_waypoint_idx]
        end_points = cmd.waypoints[cmd.target_waypoint.idx]

        line_vec = end_points - start_points
        point_vec = robot_pos - start_points
        line_len_sq = torch.sum(line_vec**2, dim=1)
        line_len_sq_nonzero = torch.where(line_len_sq > 1e-6, line_len_sq, torch.ones_like(line_len_sq))

        t = torch.sum(point_vec * line_vec, dim=1) / line_len_sq_nonzero
        t_clamped = torch.clamp(t, 0.0, 1.0)

        projection = start_points + t_clamped.unsqueeze * line_vec
        distance_error = torch.linalg.norm(robot_pos - projection, dim=1)

        return distance_error
    
    @staticmethod
    def reset_root_state_trajectory(env, env_ids: torch.Tensor):
        start_pos_2d = env.command_manager.get_command("twist").waypoints[0]

        rand_offset = (torch.rand(len(env_ids), 3, device=env.device) - 0.5) * 2
        rand_offset[:, 0] *= 0.5
        rand_offset[:, 1] *= 0.5
        rand_offset[:, 2] = 0.0

        root_pos[:, :2] = start_pos_2d
        root_pos += rand_offset
        root_pos[:, 2] += env.scene.robot.cfg.meta_info.base_height

        second_waypoint = env.command_manager.get_command("twist").waypoints[1]
        traj_dir = second_waypoint - start_pos_2d
        yaw = torch.atan2(traj_dir[1], traj_dir[0])
        root_rot = sim_utils.quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)

        root_vel = torch.zeros(len(env_ids), 6, device=env.device)

        env.scene.robot.write_root_state_to_sim(
            torch.cat([root_pos, root_rot, root_vel], dim=1), env_ids=env_ids
        )

        env.command_manager.reset(env_ids)