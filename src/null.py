"""Null for P-source at the H3 layer: 200 shuffled-label directions and 200 random Gaussian directions.
Run 10 showed a single shuffled direction can score anywhere from 0.12 to 0.91 within chat, because eval-chat and
real-chat differ along many directions. So P-source and the synthetic position are compared with this spread, not 0.5.

    python src/null.py        # after probes.py; writes results/null.json
"""
import json
import os
import sys

import numpy as np

sys.argv = sys.argv[:1]
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "probes.py"), encoding="utf-8").read()
exec(src[: src.index("PROBES = {")])  # reuse probes.py's loading, item split and helpers exactly

LAYER = json.load(open(os.path.join(RES, "probes.json")))["h3"]["layer"]
POS, NEG = ["E-bench-rw", "E-chat-rw"], ["R-bench-rw", "R-chat-rw"]
g = np.random.default_rng(1)


def score(d):
    p = {n: part(n, False, LAYER) @ d for n in ["E-chat-rw", "R-chat-rw", "SYN-chat", "R-chat-orig"]}
    den = p["E-chat-rw"].mean() - p["R-chat-rw"].mean()
    return [auc(p["E-chat-rw"], p["R-chat-rw"]), (p["SYN-chat"].mean() - p["R-chat-rw"].mean()) / den,
            (p["R-chat-orig"].mean() - p["R-chat-rw"].mean()) / den]


real = [float(x) for x in score(direction(POS, NEG, LAYER)[0])]
out = {"layer": LAYER, "real": {"auc": real[0], "syn_pos": real[1], "orig_pos": real[2]}}
dim = S["E-chat-rw"]["X"].shape[2]
for name, make in [("shuffled", lambda: direction(POS, NEG, LAYER, shuffle=True)[0]),
                   ("gaussian", lambda: g.standard_normal(dim).astype(np.float32))]:
    a = np.array([score(make()) for _ in range(200)])
    sep = np.maximum(a[:, 0], 1 - a[:, 0])  # a random direction's sign is arbitrary
    ok = np.abs(a[:, 0] - 0.5) > 0.25       # position is only meaningful where the direction separates the cells
    out[name] = {"sep_median": float(np.median(sep)), "sep_95": float(np.percentile(sep, 95)), "sep_max": float(sep.max()),
                 "frac_ge_real": float((sep >= real[0]).mean()), "n_separating": int(ok.sum()),
                 "syn_pos_median": float(np.median(a[ok, 1])), "syn_pos_5_95": np.percentile(a[ok, 1], [5, 95]).tolist(),
                 "orig_pos_median": float(np.median(a[ok, 2])), "orig_pos_5_95": np.percentile(a[ok, 2], [5, 95]).tolist()}
json.dump(out, open(os.path.join(RES, "null.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
