# Isaac Lab - Spot Navigation

This repository is a fork of the original Isaac Lab, configured for training a navigation policy for a Boston Dynamics Spot quadruped. The high-level navigation policy is trained using PPO, while a frozen pre-trained low-level locomotion policy is used to handle the low-level actions.

***

## 🛠️ Setup

1.  **Install Isaac Lab:** Follow the official Isaac Lab [installation instructions](https://isaaclab.nvidia.com/install.html). This will guide you through setting up the simulation environment and creating the necessary Conda environment.

2. **Install Nav Suite:**  Follow the Nav Suite [installation instructions](https://github.com/leggedrobotics/nav-suite/tree/main). 

3. **Add this repository as a reference and pull** Once the necessary packages are installed, run the following in the IsaacLab directory:
      ```bash
         git remote add tiamat git@github.com:JonasFrey96/tiamat.git
         git pull tiamat navigation
      ```

4. **Download locomotion policy file:** For some reason, the policy.pt file is unable to download properly when pulling, so you must download it manually from here:
      ```bash
         https://github.com/JonasFrey96/tiamat/tree/navigation/source/isaaclab_tasks/isaaclab_tasks/manager_based/navigation/config/spot/policies/height_scan
      ```
   and place it in the same location on your local repository.
***

## 🏃 Training

To train the PPO-based navigation policy, use the `isaaclab.sh` script with the provided configuration inside the IsaacLab directory. This command will start the training in a headless (non-GUI) mode. The CLI argument flags `--num_envs` and `--max_iterations` should be modified as desired.

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train_nav.py --task NavTasks-DepthImgNavigation-PPO-Spot-TRAIN --num_envs 1000 --max_iterations 10000 --headless
```

## ✅ Evaluation

To play the navigation policy, use the `isaaclab.sh` script with the provided configuration. This command will start the policy in the IsaacLab GUI. the `--checkpoint` argument should be replaced with the path to the model file you'd like to run. 

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py --task NavTasks-DepthImgNavigation-PPO-Spot-PLAY --checkpoint logs/rsl_rl/nav_tasks_depth_nav/[run directory]/[model iteration]
```

## 📈 Logging

After training, please add your results to [the policy log](https://docs.google.com/spreadsheets/d/1fgb_1slth_QkNokUy5vHiRXQaOKT2muVEwQ0kVv5bQU/edit?usp=sharing)

## License

The Isaac Lab framework is released under [BSD-3 License](LICENSE). The `isaaclab_mimic` extension and its corresponding standalone scripts are released under [Apache 2.0](LICENSE-mimic). The license files of its dependencies and assets are present in the [`docs/licenses`](docs/licenses) directory.

## Acknowledgement

Isaac Lab development initiated from the [Orbit](https://isaac-orbit.github.io/) framework. We would appreciate if you would cite it in academic publications as well:

```
@article{mittal2023orbit,
   author={Mittal, Mayank and Yu, Calvin and Yu, Qinxi and Liu, Jingzhou and Rudin, Nikita and Hoeller, David and Yuan, Jia Lin and Singh, Ritvik and Guo, Yunrong and Mazhar, Hammad and Mandlekar, Ajay and Babich, Buck and State, Gavriel and Hutter, Marco and Garg, Animesh},
   journal={IEEE Robotics and Automation Letters},
   title={Orbit: A Unified Simulation Framework for Interactive Robot Learning Environments},
   year={2023},
   volume={8},
   number={6},
   pages={3740-3747},
   doi={10.1109/LRA.2023.3270034}
}
```
