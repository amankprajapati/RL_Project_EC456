
# Downfall – Reinforcement Learning Agent

This project trains a reinforcement learning (RL) agent to play **Downfall**, a Fall Guys–style 3D obstacle course built in Godot. The agent learns to control the player character and automatically clear levels using PPO (Proximal Policy Optimization) from Stable-Baselines3.

---

## 1. What is the task?

The task is a **goal-directed navigation problem**:

- The agent controls a player in a 3D level with:
  - Platforms and tiles (some may fall or move).
  - Dynamic hazards (e.g., bombs, rotating obstacles, moving objects).
  - A finish gate / goal area.
- An **episode** starts at a spawn point and ends when:
  - The agent reaches the end gate (success),
  - The agent falls off / dies / is hit by a critical hazard (failure), or
  - A maximum step/time limit is reached (timeout).
- The objective is to **maximize cumulative reward**, which corresponds to:
  - Reaching the goal reliably,
  - Doing it quickly,
  - Avoiding deaths and unnecessary wandering.

The Godot scripts (`player.gd`, `ai_controller.gd`, `falling_tile.gd`, `bomb.gd`, `end_gate.gd`, `…`) define all in-game logic, hazards, and the link between the game and the RL agent.

---

## 2. What is the training dataset?

There is **no static dataset** (no CSVs, no images).  
The “dataset” is generated online by interacting with the environment.

At each step during training, the PPO algorithm collects transitions of the form:

\[
(s_t, a_t, r_t, s_{t+1}, \text{done}_t)
\]

where:

- \( s_t \) (state / observation) is provided by `ai_controller.gd` via `get_obs()` and typically includes:
  - Local environment information (e.g., raycast distances to nearby obstacles/ground),
  - Relative distance and direction to the goal,
  - Boolean flags such as “is the player on the ground?”,
  - Level/phase indicators (e.g., one-hot encoding of current level).
- \( a_t \) (action) is a continuous vector mapped to:
  - Player movement (e.g., forward/back, left/right),
  - Rotation/turning,
  - Jump (continuous value thresholded to on/off in `set_action()`).
- \( r_t \) (reward) is a scalar accumulated inside the Godot scripts (mainly `player.gd` and hazard scripts) and read by `get_reward()` in `ai_controller.gd`.
- `done_t` signals whether the episode has ended (success, failure, or timeout).

The PPO trainer in Python batches these transitions (roughly `n_steps × n_parallel` per update) and uses them to update the policy and value networks.

---

## 3. Steps to build and train the RL model

### 3.1. Environment setup (Godot side)

1. **Player script (`player.gd`)**
   - Extends `CharacterBody3D` and implements:
     - Movement, jumping, gravity, and animations.
     - Links to the AI controller (e.g., `ai_controller` reference).
     - Game events like `level_complete()`, `hit_by_bomb()`, `game_over()`, etc.
   - Modifies a shared reward variable on important events (progress, success, death).

2. **AI controller (`ai_controller.gd`)**
   - Extends the Godot RL plugin base (e.g., `AIController3D`).
   - Implements:
     - `get_obs()` → returns a flattened observation vector.
     - `get_obs_space()` → defines the observation space (box with given size).
     - `get_action_space()` → describes continuous actions for move/turn/jump.
     - `set_action(action)` → applies actions to the `player.gd` script.
     - `get_reward()` → returns accumulated reward since last step and resets it.
     - Episode bookkeeping: step counting, `done`, `needs_reset`, `reset()`.

3. **Hazards and goals**
   - Scripts like `bomb.gd`, `falling_tile.gd`, `spike_roller.gd`, `swiper.gd`, `end_gate.gd`, `spawn_box.gd` define:
     - When the player is hit, falls, or leaves the safe region.
     - When the level is completed (e.g., entering the end gate’s trigger).
     - They call methods on `player.gd` that adjust rewards and/or terminate episodes.

### 3.2. Training setup (Python side)

1. **Wrap the Godot environment**

```python
from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv
from stable_baselines3.common.vec_env import VecMonitor

env = StableBaselinesGodotEnv(
    env_path=ARGS.env_path,    # path to Downfall.x86_64 / .exe
    show_window=ARGS.viz,
    seed=ARGS.seed,
    n_parallel=ARGS.n_parallel,
    speedup=ARGS.speedup,
)
env = VecMonitor(env)  # monitors rewards, episode lengths
```

2. **Create the PPO model**

```python
from stable_baselines3 import PPO

model = PPO(
    "MultiInputPolicy",        # for dict/flat obs from Godot
    env,
    ent_coef=0.0001,           # entropy bonus
    n_steps=32,                # rollout length per env before update
    learning_rate=0.0003,      # or a linear schedule
    tensorboard_log=ARGS.experiment_dir,
    verbose=2,
)
```

3. **Train**

```python
model.learn(
    total_timesteps=ARGS.timesteps,
    tb_log_name=ARGS.experiment_name,
    callback=checkpoint_callback_if_any,
)
```

4. **Save / export**
   - Save Stable-Baselines model (`.zip`) if `--save_model_path` is provided.
   - Optionally export to ONNX via `export_model_as_onnx` for deployment.

5. **Inference (testing visually)**
   - Use `--inference` mode to run the trained policy for a number of steps in the Godot environment (with `--viz` to see it play).

---

## 4. Reward model

The **reward logic lives in the Godot scripts**, mainly `player.gd` and the hazard/goal scripts. Conceptually:

- **Per-step shaping:**
  - Small negative reward per step (time penalty) to encourage fast completion.
  - Positive shaping reward when the agent moves closer to the goal (e.g., based on change in distance).
- **Positive terminal rewards:**
  - Large positive reward when the player reaches the end gate (`level_complete()`).
  - Optionally additional rewards for reaching intermediate checkpoints or levels.
- **Negative terminal rewards:**
  - Significant negative reward when the player:
    - Falls off the map (e.g., y < some threshold),
    - Is hit by critical hazards (bombs, spikes, etc.),
    - Possibly times out without reaching the goal.

All these events update a shared `reward` variable, which `ai_controller.gd` reads and resets in `get_reward()` each decision step. The RL algorithm only sees the final scalar reward per step; all shaping happens inside the Godot game logic.

---

## 5. Training progression (screenshot)

Training metrics are logged automatically via TensorBoard.

1. **Run training**, for example:

```bash
python train_downfall_ppo.py   --env_path path/to/Downfall.x86_64   --experiment_dir logs/sb3   --experiment_name downfall_ppo   --timesteps 1000000
```

2. **Launch TensorBoard**:

```bash
tensorboard --logdir logs/sb3
```

3. **Open the TensorBoard URL** (e.g., `http://localhost:6006`) and inspect:
   - `rollout/ep_rew_mean` – mean episode reward over time.
   - `rollout/ep_len_mean` – mean episode length.
   - Optional: `train/value_loss`, `train/policy_loss`, etc.

4. **Training progression screenshot**:
   - Capture a plot with:
     - X-axis: timesteps,
     - Y-axis: `rollout/ep_rew_mean`.
   - As the agent learns, you should see mean episode reward increase and then stabilize.

This plot can be directly included in reports as the **training curve** for the Downfall agent.

---

## 6. Accuracy testing / evaluation

After training, evaluation is done by **running the trained agent in the environment** and measuring performance over many episodes.

Typical evaluation procedure:

1. **Load the trained model** and create an evaluation environment (usually with `n_parallel = 1` for cleaner logging).

2. **Run multiple episodes** with **deterministic actions**:
   - For each episode:
     - Reset env,
     - Run until done,
     - Accumulate the total reward,
     - Track whether the episode ended in success (reaching the goal) or failure.

3. **Compute metrics** such as:
   - **Mean episode reward** – average return per episode.
   - **Success rate** – fraction of episodes where the agent reaches the end gate.
   - **Average time to success** – mean number of steps taken in successful episodes.

Example (conceptual) evaluation code:

```python
import numpy as np

n_eval_episodes = 100
episode_rewards = []
successes = 0

for ep in range(n_eval_episodes):
    obs = env.reset()
    done = False
    total_reward = 0.0
    success = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward

        # Option 1: if the env provides a success flag in info:
        # if info.get("success", False):
        #     success = True

    episode_rewards.append(total_reward)

    # Option 2: define success via reward threshold or custom logic:
    # if success or total_reward > some_threshold:
    #     successes += 1

print("Mean reward:", np.mean(episode_rewards))
print("Std reward:", np.std(episode_rewards))
print("Success rate:", successes / n_eval_episodes)
```

These metrics serve as the “accuracy” of the RL agent in the Downfall game, how reliably and efficiently it clears levels.
