"""
Headless closed-loop runner for ex13 scenarios (no video / report), in parallel.

    python tests/ex13/run_scenarios.py                       # public cases
    python tests/ex13/run_scenarios.py tests/ex13/configs/*.yaml
    python tests/ex13/run_scenarios.py --verbose config_4_public.yaml

Prints one line per scenario and the average score.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent))


def _run_one(args):
    path, verbose = args
    logging.disable(logging.WARNING)
    import warnings

    warnings.filterwarnings("ignore")
    from pdm4ar.exercises.ex13 import agent as agent_module

    agent_module.Config.VERBOSE = verbose
    from scenario_utils import run, PLAYER

    try:
        ctx, avg, score = run(path, verbose=False)
    except Exception as e:  # report and continue with the other scenarios
        return {"name": Path(path).name, "score": 0.0, "goal": 0.0, "collision": True, "T": 0.0, "dist": 0.0,
                "effort": 0.0, "d2g": 0.0, "comp": 0.0, "error": f"{type(e).__name__}: {e}"}
    collided = avg.collision_rate > 0
    return {
        "name": Path(path).name,
        "score": score,
        "goal": avg.goal_success_rate,
        "collision": collided,
        "T": avg.avg_episode_duration,
        "dist": avg.avg_distance_travelled,
        "effort": avg.avg_actuation_effort,
        "d2g": avg.avg_distance2goal,
        "comp": avg.avg_computation_time,
    }


def resolve(p: str) -> Path:
    from scenario_utils import EX13_DEF

    path = Path(p)
    if path.exists():
        return path
    return EX13_DEF / p


def main(argv=None):
    from scenario_utils import PUBLIC

    ap = argparse.ArgumentParser()
    ap.add_argument("configs", nargs="*")
    ap.add_argument("-j", "--jobs", type=int, default=os.cpu_count())
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    paths = [resolve(p) for p in args.configs] or PUBLIC
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        results = list(ex.map(_run_one, [(str(p), args.verbose) for p in paths]))
    for r in results:
        flag = "OK " if r["goal"] >= 1 and not r["collision"] else "FAIL"
        print(
            f"{flag} {r['name']:<40} score={r['score']:7.1f} T={r['T']:5.2f} dist={r['dist']:5.1f} "
            f"eff={r['effort']:.2f} d2g={r['d2g']:.2f} comp={r['comp']:.4f}"
            + (" COLLISION" if r["collision"] else "")
            + (f" ERROR {r['error']}" if "error" in r else "")
        )
    scores = [r["score"] for r in results]
    print(f"average score {sum(scores) / len(scores):.1f} over {len(scores)} scenarios; "
          f"failures: {sum(1 for r in results if r['goal'] < 1 or r['collision'])}")
    return results


if __name__ == "__main__":
    main()
