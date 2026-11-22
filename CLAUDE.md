# PDM4AR exercises — working notes

- Python env: `.venv/` (Poetry, Python 3.11) is created by the cloud environment's setup script.
  Always use `.venv/bin/python`. `MPLBACKEND=agg` and `DISABLE_CONTRACTS=1` are set in the environment.
- Run an exercise with the official report: `.venv/bin/python src/pdm4ar/main.py -e 13` → `out/13/index.html`
  (public cases selected in `src/pdm4ar/exercises_def/ex13/get_config.py`).
- Only `src/pdm4ar/exercises/exNN/` is submitted; do not modify `exercises_def/` (simulator, evaluation,
  official configs).

## Exercise 13 (robust spaceship docking)

Solution: `src/pdm4ar/exercises/ex13/`
- `spaceship.py` SymPy dynamics; `discretization.py` adds `BatchFirstOrderHold` (vectorised FOH, used by the planner;
  the template `FirstOrderHold` B matrices are inaccurate)
- `geometry.py` ship circle cover, analytic obstacles + gradients, exact shapely clearance
- `planner.py` SCvx (cvxpy/CLARABEL, DPP-compiled once), multi-start initial plan, warm-started replans
- `tracker.py` TV-LQR + disturbance observer/feed-forward; `agent.py` replanning policy

Tests / tools (`tests/ex13/`, need `pytest`, `pytest-xdist`):
- fast unit tests: `.venv/bin/python -m pytest tests/ex13 -m "not slow" -n 4`
- closed-loop scenarios: `.venv/bin/python -m pytest tests/ex13 -m slow -n 4` (several minutes)
- headless runner with scores: `.venv/bin/python tests/ex13/run_scenarios.py [configs...]` (default: public cases)
- one scenario verbosely (replans, collisions): `.venv/bin/python tests/ex13/diagnose.py cfg.yaml`
- regenerate local configs: `.venv/bin/python tests/ex13/make_configs.py`;
  random disturbance stress set: `... make_configs.py --stress 48 --out <dir>`

Gotcha: dg_commons' `Simulator` keeps `disabled_players` etc. as class attributes — after a collision every later
simulation in the same process is broken. `tests/ex13/scenario_utils.run_simulation` resets them.
