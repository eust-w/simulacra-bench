"""A second practice generator, harder to overfit than the official one.

The official sandbox has one hidden trait and GIVEN items that only carry
information through it. Real surveys look different: country and demographics
shift answers directly, effects interact (country x gender, ...), and several
latent attitudes sit behind the answers. This writes the same file layout as
make_sandbox.py (respondents.parquet with a role column) so score.py and
tools/eval_local.py read it unchanged.

    python tools/make_sandbox_v2.py --schema starter-kit/data/unicef.json \
        --out starter-kit/_sandbox_v2/unicef --seed 1 --direct 0.8
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
KIT = os.path.join(HERE, "..", "starter-kit")
sys.path.insert(0, KIT)

from make_sandbox import (_generation_order, assign_roles,  # noqa: E402
                          generated_items, load_config, load_schema)


def make(schema, gated_value, seed, direct, interact, traits, trait_scale):
    rng = np.random.default_rng(seed)
    n = schema["dataset"]["n_rows"]
    records = schema["items"]
    items = generated_items(schema)
    order = _generation_order(schema, items)
    given = [i for i in order if records[i]["class"] == "GIVEN"]
    order = given + [i for i in order if records[i]["class"] != "GIVEN"]
    first = given[0]

    z = rng.normal(size=(n, traits))
    columns, codes = {}, {}
    for name in order:
        options = list(records[name]["values"])
        width = len(options)
        base = np.log(rng.dirichlet(np.full(width, 0.9)) + 1e-12)
        logits = np.tile(base, (n, 1))
        if records[name]["class"] == "PREDICT":
            # Direct main effects of every GIVEN column, plus one interaction
            # of the first GIVEN column (country / wave / arm) with another.
            for g in given:
                card = len(records[g]["values"]) + 1
                table = rng.normal(scale=direct / np.sqrt(len(given)),
                                   size=(card, width))
                logits += table[codes[g]]
            other = given[rng.integers(1, len(given))] if len(given) > 1 else first
            c1 = len(records[first]["values"]) + 1
            c2 = len(records[other]["values"]) + 1
            table = rng.normal(scale=interact, size=(c1 * c2, width))
            logits += table[codes[first] * c2 + codes[other]]
            loadings = rng.normal(scale=trait_scale, size=(traits, width))
            loadings *= (rng.random(traits) < 0.6)[:, None]
            logits += z @ loadings
        else:
            # GIVEN columns depend on the first GIVEN column and on the traits a
            # little, so demographics carry some information about attitudes.
            if name != first:
                c1 = len(records[first]["values"]) + 1
                logits += rng.normal(scale=0.8, size=(c1, width))[codes[first]]
            logits += z @ rng.normal(scale=0.5, size=(traits, width))
        w = np.exp(logits - logits.max(axis=1, keepdims=True))
        w /= w.sum(axis=1, keepdims=True)
        chosen = (w.cumsum(axis=1) > rng.random((n, 1))).argmax(axis=1)
        values = np.asarray(options, dtype=object)[chosen]
        code = chosen.copy()
        gate = records[name].get("gate")
        if gate:
            parent = np.asarray(columns[gate["parent"]], dtype=object)
            skipped = ~np.isin(parent, np.asarray(gate["observed_if"], dtype=object))
            values = np.where(skipped, gated_value, values)
            code = np.where(skipped, width, code)
        columns[name], codes[name] = values, code

    frame = pd.DataFrame({name: columns[name] for name in items})
    frame.insert(0, "respondent_id", ["R%06d" % i for i in range(1, n + 1)])
    return frame.astype(object)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--schema", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--direct", type=float, default=1.0)
    parser.add_argument("--interact", type=float, default=0.4)
    parser.add_argument("--traits", type=int, default=3)
    parser.add_argument("--trait-scale", type=float, default=0.8)
    args = parser.parse_args()
    config = load_config(os.path.join(KIT, "config.yml"))
    schema = load_schema(args.schema, config)
    frame = make(schema, config["gated_value"], args.seed, args.direct,
                 args.interact, args.traits, args.trait_scale)
    frame = assign_roles(schema, frame)
    os.makedirs(args.out, exist_ok=True)
    frame.to_parquet(os.path.join(args.out, "respondents.parquet"), index=False)
    print("wrote", args.out, frame.shape)


if __name__ == "__main__":
    main()
