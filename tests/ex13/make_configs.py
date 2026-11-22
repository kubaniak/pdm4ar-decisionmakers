"""
Generate additional local ex13 scenarios in tests/ex13/configs/ (deterministic).

Families:
  dist_*   : public cases 2-4 with randomised disturbance bursts (seed, strength, start, duration),
             in the range of the public example and somewhat beyond (up to ~2x sigma)
  dock_*   : docking with various dock orientations / positions / start poses (+ disturbance)
  static_* : scenario-1-like static goals with different planets and headings
  ast_*    : asteroid crossings timed to intersect the direct route
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
DEF = HERE.parents[1] / "src" / "pdm4ar" / "exercises_def" / "ex13"
OUT = HERE / "configs"

BOUNDARY = {"corners": [[-11, -11], [-11, 11], [11, 11], [11, -11], [-11, -11]]}


def load(i):
    with open(DEF / f"config_{i}_public.yaml") as f:
        return yaml.safe_load(f)


def dump(name, cfg):
    OUT.mkdir(exist_ok=True)
    with open(OUT / f"{name}.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def disturbance(rng, scale=1.0):
    return {
        "start_time": float(np.round(rng.uniform(1.0, 9.0), 2)),
        "duration": float(np.round(rng.uniform(0.5, 3.0), 2)),
        "linear_acc_sigma": float(np.round(scale * rng.uniform(0.1, 0.2), 3)),
        "angular_acc_sigma": float(np.round(scale * rng.uniform(0.02, 0.04), 3)),
        "seed": int(rng.integers(0, 10000)),
    }


def ship(x, y, psi):
    return {"x": x, "y": y, "psi": psi, "vx": 0.0, "vy": 0.0, "dpsi": 0.0, "delta": 0.0, "m": 2.5}


def dock_goal(x, y, psi):
    return {
        "type": "dock",
        "state": {"x": x, "y": y, "psi": psi, "vx": 0.0, "vy": 0.0, "dpsi": 0.0},
        "pos_tolerance": 0.5,
        "dir_tolerance": 0.1,
        "vel_tolerance": 0.2,
        "add_land_space": 0.5,
        "arms_length": 1.3,
        "offset": 1.8,
    }


def main():
    rng = np.random.default_rng(2026)
    # ---- public cases with random disturbances
    for base in (2, 3, 4):
        for k in range(4):
            cfg = load(base)
            cfg["dynamics_disturbance"] = disturbance(rng, scale=1.0 if k < 3 else 2.0)
            dump(f"dist_p{base}_{k}", cfg)
    # public 4 with the public burst but different seeds (direction/strength of the sampled acceleration)
    for k, seed in enumerate([1, 42, 777, 2024]):
        cfg = load(4)
        cfg["dynamics_disturbance"] = dict(cfg["dynamics_disturbance"], seed=seed)
        dump(f"dist_p4seed_{k}", cfg)

    # ---- docking geometry variations
    docks = [
        ((5.5, 0.0, 0.0), (-9.0, 0.0, 0.0)),  # opening facing away from the start -> must go around the dock
        ((0.0, 8.0, -np.pi / 2), (0.0, -9.0, np.pi / 2)),
        ((-8.0, 8.0, -np.pi / 4), (8.0, -8.0, 3 * np.pi / 4)),
        ((6.0, -7.0, 2.4), (-8.0, 7.0, -1.0)),
        ((-6.0, -2.0, 1.57), (7.0, 6.0, 3.0)),
    ]
    planets_sets = [
        {"Namek": {"center": [0.0, 0.0], "radius": 2.0}},
        {"Namek": {"center": [0.0, 1.0], "radius": 1.5}, "Vegeta": {"center": [-4.0, -5.0], "radius": 1.2}},
        {"Namek": {"center": [1.0, -1.0], "radius": 2.5}},
        {"Namek": {"center": [-2.0, 2.0], "radius": 1.8}, "Vegeta": {"center": [3.0, -2.0], "radius": 1.0}},
        {"Namek": {"center": [2.0, 3.0], "radius": 1.5}},
    ]
    for k, ((gx, gy, gpsi), (sx, sy, spsi)) in enumerate(docks):
        cfg = {
            "agents": {"PDM4ARSpaceship": {"color": "blue", "state": ship(sx, sy, spsi), "goal": dock_goal(gx, gy, float(gpsi))}},
            "planets": planets_sets[k],
            "boundary": copy.deepcopy(BOUNDARY),
            "seed": 98,
            "dynamics_disturbance": disturbance(rng) if k % 2 == 0 else None,
        }
        dump(f"dock_{k}", cfg)

    # ---- static goals
    statics = [
        ((8.0, 8.0, 1.57), (-8.0, -8.0, 0.0), {"A": {"center": [0.0, 0.0], "radius": 3.0}}),
        ((-8.0, 7.0, 3.0), (8.0, -7.0, 0.0), {"A": {"center": [2.0, -1.0], "radius": 1.5}, "B": {"center": [-3.0, 3.0], "radius": 1.5}}),
        ((0.0, 9.0, -1.57), (0.0, -9.0, 1.57), {"A": {"center": [0.0, 0.0], "radius": 2.0}, "B": {"center": [5.0, 5.0], "radius": 1.0}, "C": {"center": [-5.0, -5.0], "radius": 1.0}}),
        ((3.0, 2.0, 0.0), (-3.0, -2.0, 3.14), {}),
    ]
    for k, ((gx, gy, gpsi), (sx, sy, spsi), planets) in enumerate(statics):
        cfg = {
            "agents": {"PDM4ARSpaceship": {"color": "blue", "state": ship(sx, sy, spsi), "goal": {
                "type": "static",
                "state": {"x": gx, "y": gy, "psi": gpsi, "vx": 0.0, "vy": 0.0, "dpsi": 0.0},
                "pos_tolerance": 0.5, "dir_tolerance": 0.5, "vel_tolerance": 1.0}}},
            "planets": planets,
            "boundary": copy.deepcopy(BOUNDARY),
            "seed": 98,
        }
        if k == 1:
            cfg["dynamics_disturbance"] = disturbance(rng)
        dump(f"static_{k}", cfg)

    # ---- asteroid crossings: asteroids timed to cross the straight route
    for k in range(4):
        cfg = load(3)
        r = np.random.default_rng(100 + k)
        ast = {}
        for j in range(3):
            # crossing point on the route y=0 and time of crossing
            xc = float(np.round(r.uniform(-5, 5), 2))
            tc = float(np.round(r.uniform(3, 10), 2))
            speed = float(np.round(r.uniform(0.6, 1.5), 2))
            sign = 1 if r.random() < 0.5 else -1
            ast[f"A{j}"] = {
                "start": [xc, float(-sign * speed * tc)],
                "radius": float(np.round(r.uniform(0.5, 1.0), 2)),
                "velocity": [0.0, float(sign * speed)],
                "orientation": 0.0,
            }
        cfg["asteroids"] = ast
        cfg["dynamics_disturbance"] = disturbance(r) if k >= 2 else None
        dump(f"ast_{k}", cfg)


if __name__ == "__main__":
    main()
    print("\n".join(sorted(p.name for p in OUT.glob("*.yaml"))))
