# Repository for the code of my master’s thesis

This repository contains code for training and evaluating PPO agents that play games from the Arcade Learning Environment (ALE) using object-centric input representations. These object-centric input representations are variants of [OCAtari's](https://github.com/k4ntz/OC_Atari) Semantic Vector or of [OCCAM's masks](https://github.com/VanillaWhey/OCAtariWrappers). In addition, the object focus of agents trained on raw pixel inputs (DQN-like) or the masks can be compared.

The code in this repository is based on [OC-CleanRL](https://github.com/BluemlJ/oc_cleanrl/tree/master) and [OCCAM](https://github.com/VanillaWhey/OCAtariWrappers).

The trained agents are not contained in this repository; instead, they were uploaded to [Google Drive](https://drive.google.com/drive/folders/1ndNVYIUkW7nXyAWYF1vgSCfMrRH1eKbl?usp=drive_link). After downloading the agents and placing them in a folder named `agents` at the repository root, they can be evaluated using `eval_all_agents.py`. The evaluation results can then be visualized using `generate_bar_plots.py` and displayed in tables using `generate_performance_tables.py`, both located in the `python_scripts` folder.

The tables for the object focus comparison are generated with `generate_object_focus_tables.py` from the same folder. If trained agents are available, the JSON files containing object focus data can be produced by `object_focus_comparison.py`.


---

## Getting Started with the repository

### Prerequisites
- Python 3.10
- pip

### Running Experiments Locally

1. **Clone the repository**

   ```bash
   git clone https://github.com/spassmitfop/mthesis_final.git && cd mthesis_final
   ```

2. **Install dependencies**

   ```bash
   # Core dependencies
   pip install -r requirements.txt
   ```

3. **Enable OCAtari/HackAtari**

   ```bash
   cd submodules/OC_Atari
   pip install -e .
   cd ..
   cd HackAtari
   pip install -e .
   cd ..
   cd ..
   ```

4. **Start a training run**

   ```bash
   python cleanrl/ppo_atari_oc.py --env-id ALE/Pong-v5 --obs_mode obj --architecture PPO_OBJ --backend OCAtari --masked_wrapper ext_obj --use_angle
   ```

---

