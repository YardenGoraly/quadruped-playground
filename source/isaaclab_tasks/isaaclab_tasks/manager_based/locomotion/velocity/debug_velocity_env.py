import torch
from isaaclab.envs import ManagerBasedRLEnv

class DebugEnv(ManagerBasedRLEnv):
    """
    A custom environment that inherits from ManagerBasedRLEnv
    and prints observation data during each step.
    """

    def step(self, actions: torch.Tensor):
        """
        Overrides the default step method to add printing logic.
        """
        obs, rew, terminated, truncated, info = super().step(actions)

        if self.num_envs > 0:
            print("======================================================")
            print(f"Time: {self.sim.current_time * self.sim.get_rendering_dt():.2f}s (Step: {self.sim.current_time})")

            print("\n-- Observations --")
            # Get the observation data for the first environment (env_idx = 0)
            obs_data_env0 = self.observation_manager.get_active_iterable_terms(env_idx=0)
            print("Observations for Environment 0:")
            # Loop through the active observation terms and print their values
            for term_name, term_tensor in obs_data_env0:
                # term_tensor is a tensor, we get the first (and only) element
                print(f"  > {term_name}: {term_tensor}")
            


            base_height_state = self.scene["robot"].data.root_pos_w[:, 2][0].item()
            print("\n-- Monitored States --")
            print(f"  {'base_height (meters)':<20}: {base_height_state:.4f}")



            reward_data = self.reward_manager.get_active_iterable_terms(env_idx=0)
            print("\n-- Reward Term Values --")
            for term_name, term_tensor in reward_data:
                # Check if the term you're interested in is present
                if "base_height" in term_name:
                    print(f"  {term_name}: {term_tensor}  <-- HEIGHT REWARD")
                else:
                    print(f"  {term_name}: {term_tensor}")

            print("======================================================\n")


        # Return the results as normal
        return obs, rew, terminated, truncated, info
