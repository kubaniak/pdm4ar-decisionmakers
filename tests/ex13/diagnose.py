"""Run one scenario verbosely and print replans and collision reports: python tests/ex13/diagnose.py cfg.yaml"""
import logging
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
logging.disable(logging.WARNING)
warnings.filterwarnings("ignore")

from pdm4ar.exercises.ex13 import agent as agent_module  # noqa: E402

agent_module.Config.VERBOSE = True
from scenario_utils import PLAYER, run  # noqa: E402

for p in sys.argv[1:]:
    ctx, avg, score = run(p)
    for cr in ctx.collision_reports[:3]:
        print("  collision at", float(cr.at_time), list(cr.players.keys()))
    model = ctx.models[PLAYER]
    if hasattr(model, "sampled_disturbance"):
        d = model.disturbance
        print("  disturbance", d.start_time, d.duration, "acc=", [round(v, 3) for v in model.sampled_disturbance])
