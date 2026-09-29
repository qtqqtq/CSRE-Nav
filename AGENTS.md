# Repository Guidelines

## Project Structure & Module Organization

`SAC/` contains the active reinforcement-learning implementation. `train.py` is the CNN-SAC training entry point; `cnn_network.py`, `replay_buffer.py`, and `settings.py` define the agent, storage, and shared configuration. `Sparrow_V2.py` provides the vectorized simulator, while `cspace_cost_rays.py` and `global_planner.py` implement configuration-space perception and optional planning. Use `play_cspace.py` for interactive evaluation. Standalone testing, profiling, and sweep scripts live in `SAC/script/`. ROS/Gazebo packages, messages, worlds, and URDF/Xacro files live under `catkin_ws/src/`. Treat `SAC/runs/`, `SAC/results/`, and `SAC/pytorch_models/` as generated artifacts.

## Build, Test, and Development Commands

Run Python commands from `SAC/` because modules use local imports:

```bash
# Build the project container with bash build.bash; Python dependencies are in dockerfile.
cd catkin_ws && source /opt/ros/noetic/setup.bash && catkin_make
cd ../SAC && python3 train.py
python3 play_cspace.py --ModelName dense_sample_cspace_1 --dvc cuda
python3 global_planner.py
python3 cspace_cost_rays.py
python3 cspace_cost_rays.py --bench
```

The first two commands install dependencies and build ROS packages. The remaining commands train, run a policy, execute planner/cost-ray self-tests, and benchmark perception. `bash build.bash` performs the catkin build inside the project container mounted at `/DRL-robot-navigation`.

## Coding Style & Naming Conventions

Use four-space indentation and PEP 8 conventions for new Python code. Prefer `snake_case` for modules, functions, and variables; `PascalCase` for classes; and `UPPER_SNAKE_CASE` for shared constants. Add shape and unit information to tensor-heavy docstrings or comments, for example `(N, 36, 30)` and `cm/s`. Keep tunable values in `SAC/settings.py` instead of duplicating them. Follow existing ROS/C++ conventions in the package being edited. No formatter or linter is currently enforced.

## Testing Guidelines

There is no centralized pytest suite or coverage threshold. Run the relevant executable self-test after each change. For simulator changes, also run `python3 Sparrow_V2.py`. Record device, seed, and key settings when reporting training regressions.

## Commit & Pull Request Guidelines

Recent commits use short imperative prefixes such as `fix:`, `feat:`, and `perf:`; follow that pattern and keep each commit focused. Pull requests should explain behavior changes, list verification commands, link related issues, and include screenshots or TensorBoard metrics for rendering or learning changes. Do not commit checkpoints, event logs, build products, or large generated result files unless explicitly required.
