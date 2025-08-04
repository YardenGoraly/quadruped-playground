import torch
import torch.nn as nn
from rsl_rl.modules import ActorCritic

class TrajectoryFollowerPolicy(ActorCritic):
    def __init__(self, num_actor_obs, num_critic_obs, num_actions, **kwargs):
        custom_params = kwargs.pop("params")
        pretrained_policy_path = custom_params.pop("pretrained_policy_path")
        super().__init__(num_actor_obs, num_critic_obs, num_actions, **kwargs)

        pretrained_state_dict = torch.load(pretrained_policy_path, map_location=self.device, weights_only=True)

        self.load_state_dict(pretrained_state_dict["model_state_dict"])

        # Freeze pre-trained layers
        for param in self.actor.parameters():
            param.requires_grad = False
        for param in self.critic.parameters():
            param.requires_grad = False
        self.std.requires_grad = False
        
        # Add new head for trajector. Size must math that of the last hidden layer + command size
        feature_size = 128
        command_size = 3 
        trajectory_head_input_size = feature_size + command_size

        self.trajectory_actor_head = nn.Sequential(
            nn.Linear(trajectory_head_input_size, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, num_actions)
        ).to(self.device)

        torch.nn.init.uniform_(self.trajectory_actor_head[-1].weight, a=-0.01, b=0.01)
        torch.nn.init.zeros_(self.trajectory_actor_head[-1].bias)
        
        self.distribution = None

    @property
    def entropy(self):
        """The entropy of the last computed action distribution."""
        return self.distribution.entropy()

    def get_distribution(self, observations):
        """Helper function to compute the action distribution."""
        with torch.no_grad():
            walking_actions_mean = self.actor(observations)
            frozen_features = self.actor[:-1](observations)
        trajectory_command_obs = observations[:, -3:]
        combined_input = torch.cat((frozen_features, trajectory_command_obs), dim=1)
        action_offset = self.trajectory_actor_head(combined_input)
        final_actions_mean = walking_actions_mean + action_offset
        covariance = torch.diag(self.std.pow(2))
        distribution = torch.distributions.MultivariateNormal(final_actions_mean, scale_tril=covariance)
        return distribution

    def act(self, observations, **kwargs):
        """Called during rollouts to generate actions."""
        self.distribution = self.get_distribution(observations)
        return self.distribution.sample()

    def act_inference(self, observations):
        """Called during playback to generate deterministic actions."""
        distribution = self.get_distribution(observations)
        return distribution.mean

    def evaluate(self, critic_observations, observations=None, actions=None, **kwargs):
        """
        This method is called by the PPO algorithm in different ways.
        - During rollout, it's called with just critic_observations to get the value.
        - During update, it's called with all arguments to get log_prob and value.
        """
        # If we are in the main update loop, we need to compute the log_prob
        if observations is not None and actions is not None:
            self.distribution = self.get_distribution(observations)
            actions_log_prob = self.distribution.log_prob(actions)
            value = self.critic(critic_observations)
            return actions_log_prob, value
        # If we are just getting the value during rollout
        else:
            value = self.critic(critic_observations)
            return value
    
    @property
    def device(self):
        return next(self.parameters()).device