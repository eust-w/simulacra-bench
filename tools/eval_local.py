"""Exact (un-noised) local scoring of a submission directory on sandbox data.

    python tools/eval_local.py --submission submissions/v2_stack \
        --sandbox starter-kit/_sandbox --phase 1 [--instruments unicef unhcr]

Imports main.py in-process (no venv, no network cut -- use starter-kit/score.py
for the official dry run) and reports skill per instrument, the mean, and time.
"""

import argparse
import importlib.util
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
KIT = os.path.join(HERE, "..", "starter-kit")
sys.path.insert(0, KIT)

from make_sandbox import load_config, load_schema  # noqa: E402
from score import (check, floored, load_frames, marginal_reference,  # noqa: E402
                   sample_rows, score)

INSTRUMENTS = ("unicef", "world_bank", "unhcr")


def load_submission(path):
    spec = importlib.util.spec_from_file_location("submission_main",
                                                  os.path.join(path, "main.py"))
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, path)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", required=True)
    parser.add_argument("--sandbox", default=os.path.join(KIT, "_sandbox"))
    parser.add_argument("--phase", type=int, default=1)
    parser.add_argument("--instruments", nargs="*", default=list(INSTRUMENTS))
    parser.add_argument("--json", default=None, help="append result as JSON line")
    args = parser.parse_args()

    config = load_config(os.path.join(KIT, "config.yml"))
    floor = config["scoring"]["floor"]
    t0 = time.time()
    module = load_submission(args.submission)
    t_import = time.time() - t0

    results = {}
    for name in args.instruments:
        schema = load_schema(os.path.join(KIT, "data", name + ".json"), config)
        respondents = load_frames(os.path.join(args.sandbox, name), schema)
        frame, cells, truth = sample_rows(schema, respondents, args.phase)
        t1 = time.time()
        vectors = module.predict(frame.copy(), json.loads(json.dumps(schema)))
        dt = time.time() - t1
        check(schema, vectors, cells)
        vectors = floored(vectors, schema, cells, floor)
        visible = frame.iloc[:len(frame) - len({c[0] for c in cells})]
        ref = marginal_reference(schema, visible, cells, truth, floor)
        out = score(schema, config, vectors, truth, cells, ref)
        results[name] = {"skill": out["skill"], "se": out["std_error"],
                         "log_score": out["log_score"], "seconds": dt}
        print("%-11s skill %+.4f  (se %.4f)  logscore %.4f  ref %.4f  %.1fs"
              % (name, out["skill"], out["std_error"], out["log_score"], ref, dt),
              flush=True)

    mean = float(np.mean([r["skill"] for r in results.values()]))
    total = time.time() - t0
    print("MEAN skill %+.4f   import %.1fs   total %.1fs" % (mean, t_import, total))
    if args.json:
        with open(args.json, "a") as fh:
            fh.write(json.dumps({"submission": args.submission,
                                 "sandbox": args.sandbox, "phase": args.phase,
                                 "results": results, "mean": mean,
                                 "total_seconds": total,
                                 "time": time.strftime("%Y-%m-%d %H:%M")}) + "\n")


if __name__ == "__main__":
    main()
