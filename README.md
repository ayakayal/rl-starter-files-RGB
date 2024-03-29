# RL Starter Files

RL starter files in order to immediatly train, visualize and evaluate an agent **without writing any line of code**.

<p align="center">
    <img width="300" src="README-rsrc/visualize-keycorridor.gif">
</p>

These files are suited for [`gym-minigrid`](https://github.com/maximecb/gym-minigrid) environments and [`torch-ac`](https://github.com/lcswillems/torch-ac) RL algorithms. They are easy to adapt to other environments and RL algorithms.

## Features

- **Script to train**, including:
  - Log in txt, CSV and Wandb
  - Save model
  - Stop and restart training
  - Use PPO algorithm not A2C 
- **Script to visualize**, including:
  - Act by sampling or argmax
  - Save as Gif
- **Script to evaluate**, including:
  - Act by sampling or argmax
  - List the worst performed episodes

## Installation

1. Clone this repository.

2. Install `gym-minigrid` environments and `torch-ac` RL algorithms:

```
pip3 install -r requirements.txt
```

## Example of use

Train, visualize and evaluate an agent on the `MiniGrid-DoorKey-5x5-v0` environment:

<p align="center"><img src="README-rsrc/doorkey.png"></p>

1. Train the agent on the `MiniGrid-DoorKey-16x16-v0` environment with PPO algorithm (Grid encoding obervations) and add specific intrinsic reward algorithms. Data is saved in /scratch/rmapkay/folder-name and logged to wandb.

```
python -m scripts.train --algo ppo --env MiniGrid-DoorKey-16x16-v0 --folder-name Doorkey_grid_encodings --frames 40000000 --entropy-coef 0.0005 --ir-coef 0 --seed 1 --RGB False --singleton False --pretraining False --save-heatmaps False

python -m scripts.train --algo ppo_state_count --env MiniGrid-DoorKey-16x16-v0 --folder-name Doorkey_grid_encodings --frames 40000000 --entropy-coef 0.0005 --ir-coef 0.005 --seed 1 --RGB False --singleton False --pretraining False --save-heatmaps False

python -m scripts.train --algo ppo_entropy --env MiniGrid-DoorKey-16x16-v0 --folder-name Doorkey_grid_encodings --frames 40000000 --entropy-coef 0.0005 --ir-coef 0.0005 --seed 1 --RGB False --singleton False --pretraining False --save-heatmaps False

python -m scripts.train --algo ppo_icm_alain --env MiniGrid-DoorKey-16x16-v0 --folder-name Doorkey_grid_encodings --frames 40000000 --entropy-coef 0.0005 --ir-coef 0.05 --seed 1 --RGB False --singleton False --pretraining False --save-heatmaps False

python -m scripts.train --algo ppo_diayn --env MiniGrid-DoorKey-16x16-v0 --folder-name Doorkey_grid_encodings --frames 40000000 --entropy-coef 0.0005 --ir-coef 0.01 --seed 1 --RGB False --singleton False --pretraining True --save-heatmaps False

python -m scripts.train --algo ppo_diayn --env MiniGrid-DoorKey-16x16-v0 --folder-name Doorkey_grid_encodings --frames 40000000 --entropy-coef 0.0005 --ir-coef 0 --seed 1 --RGB False --singleton False --pretraining False --save-heatmaps False --pretrained-model-name MiniGrid-DoorKey-16x16-v0_ppo_diayn_seed1_ir0.01_ent0.0005_sk10_dis0.0003  --folder-name-pretrained-model Doorkey_grid_encodings

```
2. Train the agent on the `MiniGrid-DoorKey-8x8-v0` environment with PPO algorithm (RGB obervations) and add SimHash2 instead of State Count as intrinsic reward because it is not possible to count raw pixels. you can use all other intrinsic rewards as they are with RGB=True.
```
python -m scripts.train --algo ppo_simhash2 --env MiniGrid-DoorKey-8x8-v0 --folder-name Doorkey_RGB --frames 40000000 --entropy-coef 0.0005 --ir-coef 0.005 --seed 1 --RGB True --singleton False --pretraining False --save-heatmaps False
```

3. Train the agent on the `MiniGrid-DoorKey-8x8-v0` environment with PPO algorithm (RGB obervations) and SimHash2 but on singleton (not procedurally generated environment) to visualize heatmaps. Set singleton and save_heatmaps to True. Heatmaps are saved every 50 PPO updates and in a folder /heatmaps. If you would like to plot a grid of heatmaps check plot_heatmaps_grid.py, but change the path.
```
python -m scripts.train --algo ppo_simhash2 --env MiniGrid-DoorKey-8x8-v0 --folder-name Doorkey_RGB --frames 40000000 --entropy-coef 0.0005 --ir-coef 0.005 --seed 1 --RGB True --singleton True --pretraining False --save-heatmaps True
```
<p align="center"><img src="README-rsrc/train-terminal-logs.png"></p>

3. Visualize agent's behavior:

```
python -m scripts.visualize_edited --env MiniGrid-DoorKey-16x16-v0 --folder-name test --episodes 5 --model MiniGrid-DoorKey-16x16-v0 _ppo_seed1_ir0.0_ent0.0005 --gif Doorkey16_sing
```

<p align="center"><img src="README-rsrc/visualize-doorkey.gif"></p>


<p align="center"><img src="README-rsrc/evaluate-terminal-logs.png"></p>

**Note:** ppo_simhash_better_rep is for the case of Doorkey only, appending whether the key is picked up or not to the hashed key

**Note:** More details on the commands are given below.

## Other examples



## Files

This package contains:
- scripts to:
  - train an agent \
  in `script/train.py` ([more details](#scripts-train))
  - visualize agent's behavior \
  in `script/visualize.py` ([more details](#scripts-visualize))
  - evaluate agent's performances \
  in `script/evaluate.py` ([more details](#scripts-evaluate))
- a default agent's model \
in `model.py` ([more details](#model))
- utilitarian classes and functions used by the scripts \
in `utils`

These files are suited for [`gym-minigrid`](https://github.com/maximecb/gym-minigrid) environments and [`torch-ac`](https://github.com/lcswillems/torch-ac) RL algorithms. They are easy to adapt to other environments and RL algorithms by modifying:
- `model.py`
- `utils/format.py`

<h2 id="scripts-train">scripts/train.py</h2>

An example of use:

```bash
python3 -m scripts.train --algo ppo --env MiniGrid-DoorKey-5x5-v0 --folder-name DoorKey --save-interval 10 --frames 80000
```

The script loads the model in `sratch/rmapkay/folder-name` or creates it if it doesn't exist, then trains it with the PPO algorithm on the MiniGrid DoorKey environment, and saves it every 10 updates. It stops after 80 000 frames.

**Note:** You can define a different storage location in the environment variable `PROJECT_STORAGE`.

More generally, the script has 2 required arguments:
- `--algo ALGO`: name of the RL algorithm used to train
- `--env ENV`: name of the environment to train on

and a bunch of optional arguments among which:
- `--recurrence N`: gradient will be backpropagated over N timesteps. By default, N = 1. If N > 1, a LSTM is added to the model to have memory.
- `--text`: a GRU is added to the model to handle text input.
- ... (see more using `--help`)

During training, logs are printed in your terminal (and saved in text and CSV format):

<p align="center"><img src="README-rsrc/train-terminal-logs.png"></p>

**Note:** `U` gives the update number, `F` the total number of frames, `FPS` the number of frames per second, `D` the total duration, `rR:μσmM` the mean, std, min and max reshaped return per episode, `F:μσmM` the mean, std, min and max number of frames per episode, `H` the entropy, `V` the value, `pL` the policy loss, `vL` the value loss and `∇` the gradient norm.

During training, logs are also plotted in Tensorboard:



<h2 id="scripts-visualize">scripts/visualize.py</h2>

An example of use:

```
python3 -m scripts.visualize --env MiniGrid-DoorKey-5x5-v0 --model DoorKey
```

<p align="center"><img src="README-rsrc/visualize-doorkey.gif"></p>

In this use case, the script displays how the model in `storage/DoorKey` behaves on the MiniGrid DoorKey environment.

More generally, the script has 2 required arguments:
- `--env ENV`: name of the environment to act on.
- `--model MODEL`: name of the trained model.

and a bunch of optional arguments among which:
- `--argmax`: select the action with highest probability
- ... (see more using `--help`)

<h2 id="scripts-evaluate">scripts/evaluate.py</h2>

An example of use:

```
python3 -m scripts.evaluate --env MiniGrid-DoorKey-5x5-v0 --model DoorKey
```

<p align="center"><img src="README-rsrc/evaluate-terminal-logs.png"></p>

In this use case, the script prints in the terminal the performance among 100 episodes of the model in `storage/DoorKey`.

More generally, the script has 2 required arguments:
- `--env ENV`: name of the environment to act on.
- `--model MODEL`: name of the trained model.

and a bunch of optional arguments among which:
- `--episodes N`: number of episodes of evaluation. By default, N = 100.
- ... (see more using `--help`)

<h2 id="model">model.py</h2>

The default model is discribed by the following schema:

<p align="center"><img src="README-rsrc/model.png"></p>

By default, the memory part (in red) and the langage part (in blue) are disabled. They can be enabled by setting to `True` the `use_memory` and `use_text` parameters of the model constructor.

This model can be easily adapted to your needs.
