"""Difference-of-means probes per layer (P-naive, P-source, P-format), controls, and where SYN-chat lands.
Split by item (70/30, seed 0) so a request never sits in train and test in different formats.
Writes results/probes.json and results/fig_*.png.

    python src/probes.py                  # results/acts
    python src/probes.py results/acts_test
"""
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
ACTS = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results", "acts")
RES = os.path.dirname(ACTS) if len(sys.argv) > 1 else os.path.join(ROOT, "results")
TAG = "" if len(sys.argv) == 1 else "_test"
rng = np.random.default_rng(0)
B = 1000  # bootstrap resamples


# Run 09 spot check: dropped from every cell. Moral-scenario rewrites lost both scenarios (3/3); real-to-benchmark
# rewrites that cut a long question to under 40% (9); plus the two the hand check marked N or left blank.
EXCLUDE = {"mmlu24", "mmlu46", "mmlu66", "wc8", "wc37", "wc39", "wc74", "wc103", "wc113", "wc123", "wc149", "wc190",
           "tqa36", "syn52"}


def load(name):
    meta = json.load(open(os.path.join(ACTS, name + ".json")))
    k = np.array([i not in EXCLUDE for i in meta["items"]])
    meta = {key: [v for v, keep in zip(val, k) if keep] for key, val in meta.items()}
    return dict(X=np.load(os.path.join(ACTS, name + ".npy"))[k].astype(np.float32), text=meta["text"],
                items=np.array(meta["items"]), ntok=np.array(meta["ntok"], dtype=np.float32))



S = {n: load(n) for n in ["E-bench-rw", "E-chat-rw", "R-chat-rw", "R-bench-rw", "R-chat-orig", "SYN-chat"]}
L = S["E-bench-rw"]["X"].shape[1]


def split(items):
    u = np.unique(items)
    test = set(rng.permutation(u)[: int(round(0.3 * len(u)))])
    return np.array([i not in test for i in items])


TR = {}
for src in ("E", "R"):
    tr = dict(zip(S[f"{src}-bench-rw"]["items"], split(S[f"{src}-bench-rw"]["items"])))
    for n in S:
        if n.startswith(src):
            TR[n] = np.array([tr.get(i, True) for i in S[n]["items"]])
TR["SYN-chat"] = np.zeros(len(S["SYN-chat"]["items"]), bool)  # scored only


def part(n, train, layer=None):
    m = TR[n] if train else ~TR[n]
    return S[n]["X"][m] if layer is None else S[n]["X"][m, layer]


def auc(pos, neg):
    r = rankdata(np.concatenate([pos, neg]))
    return (r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def auc_ci(pos, neg):
    bs = [auc(rng.choice(pos, len(pos)), rng.choice(neg, len(neg))) for _ in range(B)]
    return [auc(pos, neg), *np.percentile(bs, [2.5, 97.5])]


def direction(pos_sets, neg_sets, layer, shuffle=False):
    P = np.concatenate([part(n, True, layer) for n in pos_sets])
    N = np.concatenate([part(n, True, layer) for n in neg_sets])
    if shuffle:
        A = rng.permutation(np.concatenate([P, N]))
        P, N = A[: len(P)], A[len(P):]
    return P.mean(0) - N.mean(0), (P.mean(0) + N.mean(0)) / 2


PROBES = {"naive": (["E-bench-rw"], ["R-chat-rw"]),
          "source": (["E-bench-rw", "E-chat-rw"], ["R-bench-rw", "R-chat-rw"]),
          "format": (["E-bench-rw", "R-bench-rw"], ["E-chat-rw", "R-chat-rw"])}
# Test contrasts per probe: (name, positive test set, negative test set)
TESTS = {"naive": [("own", "E-bench-rw", "R-chat-rw"), ("crossed", "E-chat-rw", "R-bench-rw")],
         "source": [("within_bench", "E-bench-rw", "R-bench-rw"), ("within_chat", "E-chat-rw", "R-chat-rw"),
                    ("shuffled_within_chat", "E-chat-rw", "R-chat-rw")],
         "format": [("eval_src", "E-bench-rw", "E-chat-rw"), ("real_src", "R-bench-rw", "R-chat-rw")]}

out = {"n": {n: [int(TR[n].sum()), int((~TR[n]).sum())] for n in S}, "layers": L, "probes": {}}
for p, (pos, neg) in PROBES.items():
    res = {t[0]: [] for t in TESTS[p]}
    res["pct_eval"] = []
    for l in range(L):
        d, mid = direction(pos, neg, l)
        ds, mids = direction(pos, neg, l, shuffle=True)
        for name, a, b in TESTS[p]:
            dd, mm = (ds, mids) if name.startswith("shuffled") else (d, mid)
            res[name].append(auc_ci((part(a, False, l) - mm) @ dd, (part(b, False, l) - mm) @ dd))
        # Devbunova-style: % of each test cell above the train midpoint
        res["pct_eval"].append({n: float(((part(n, False, l) - mid) @ d > 0).mean() * 100)
                                for n in ["E-bench-rw", "E-chat-rw", "R-chat-rw", "R-bench-rw", "R-chat-orig", "SYN-chat"]})
    out["probes"][p] = res
    print(p, {k: round(max(v, key=lambda x: x[0])[0], 3) for k, v in res.items() if k != "pct_eval"}, flush=True)

# Controls
out["length_only"] = {"within_chat": auc_ci(S["E-chat-rw"]["ntok"], S["R-chat-rw"]["ntok"]),
                      "within_bench": auc_ci(S["E-bench-rw"]["ntok"], S["R-bench-rw"]["ntok"]),
                      "syn_vs_real_chat": auc_ci(S["SYN-chat"]["ntok"], S["R-chat-rw"]["ntok"])}
def surface(texts):
    """What a probe could exploit without reading meaning: length, words, question marks, first person, digits."""
    return np.array([[len(t), len(t.split()), t.count("?"), len(re.findall(r"\b(i|i'm|my|me)\b", t, re.I)),
                      sum(c.isdigit() for c in t)] for t in texts], dtype=np.float32)


def surface_auc(a, b):
    Xa, Xb = surface(S[a]["text"]), surface(S[b]["text"])
    clf = LogisticRegression(max_iter=1000).fit(np.concatenate([Xa[TR[a]], Xb[TR[b]]]),
                                                np.r_[np.ones(TR[a].sum()), np.zeros(TR[b].sum())])
    return auc_ci(clf.decision_function(Xa[~TR[a]]), clf.decision_function(Xb[~TR[b]]))


out["surface_only"] = {"within_chat": surface_auc("E-chat-rw", "R-chat-rw"),
                       "within_bench": surface_auc("E-bench-rw", "R-bench-rw")}
out["authorship"] = []  # raw WildChat vs llama3 paraphrase of the same WildChat
for l in range(L):
    d, mid = direction(["R-chat-rw"], ["R-chat-orig"], l)
    out["authorship"].append(auc_ci((part("R-chat-rw", False, l) - mid) @ d, (part("R-chat-orig", False, l) - mid) @ d))


# H3: pick the best layer on an inner split of the train items, never on test
def pick_layer(p, a, b):
    best, best_auc = None, -1
    for l in range(1, L):
        P = [part(n, True, l) for n in PROBES[p][0]]
        N = [part(n, True, l) for n in PROBES[p][1]]
        half = lambda X, k: X[k::2]  # ponytail: alternate rows as inner split; fine since rows are shuffled items
        d = np.concatenate([half(x, 0) for x in P]).mean(0) - np.concatenate([half(x, 0) for x in N]).mean(0)
        s = auc(half(part(a, True, l), 1) @ d, half(part(b, True, l), 1) @ d)
        if s > best_auc:
            best, best_auc = l, s
    return best


def position(p, layer):
    """(mean(set) - mean(real chat)) / (mean(eval chat) - mean(real chat)) on held-out chat cells."""
    d, _ = direction(*PROBES[p], layer)
    proj = {n: part(n, False, layer) @ d for n in ["E-chat-rw", "R-chat-rw", "R-chat-orig", "SYN-chat"]}
    pos = lambda pr, n: (pr[n].mean() - pr["R-chat-rw"].mean()) / (pr["E-chat-rw"].mean() - pr["R-chat-rw"].mean())
    o = {}
    for n in ["SYN-chat", "R-chat-orig"]:
        bs = [pos({k: rng.choice(v, len(v)) for k, v in proj.items()}, n) for _ in range(B)]
        o[n] = [float(pos(proj, n)), *np.percentile(bs, [2.5, 97.5]).tolist()]
    return o, {k: v.tolist() for k, v in proj.items()}


# Position only makes sense for P-source; for P-naive the comparable number is pct_eval["SYN-chat"].
l = pick_layer("source", "E-chat-rw", "R-chat-rw")
pos_, proj = position("source", l)
h3 = {"layer": l, "position": pos_, "auc_within_chat_at_layer": out["probes"]["source"]["within_chat"][l],
      "naive_pct_eval_at_its_best_layer": out["probes"]["naive"]["pct_eval"][pick_layer("naive", "E-bench-rw", "R-chat-rw")]}
plt.figure(figsize=(6, 3.5))
for n, v in proj.items():
    plt.hist(v, bins=25, alpha=0.5, label=n)
plt.legend(); plt.xlabel(f"projection on P-source direction, layer {l}"); plt.tight_layout()
plt.savefig(os.path.join(RES, f"fig_h3_hist{TAG}.png"), dpi=130); plt.close()
# Check: H3 on the shortest third of synthetic prompts only (synthetic is ~2.7x longer than real, Run 06)
keep = TR["SYN-chat"]
TR["SYN-chat"] = S["SYN-chat"]["ntok"] > np.percentile(S["SYN-chat"]["ntok"], 100 / 3)  # ~TR = shortest third
h3["position_shortest_third"] = position("source", l)[0]["SYN-chat"]
TR["SYN-chat"] = keep
out["h3"] = h3
print("h3", json.dumps(h3))

fig, ax = plt.subplots(1, 3, figsize=(13, 3.5), sharey=True)
for a, (p, res) in zip(ax, out["probes"].items()):
    for k, v in res.items():
        if k == "pct_eval":
            continue
        v = np.array(v)
        a.plot(v[:, 0], label=k); a.fill_between(range(L), v[:, 1], v[:, 2], alpha=0.2)
    if p == "naive":
        v = np.array(out["authorship"]); a.plot(v[:, 0], "k--", lw=1, label="authorship ctrl")
    a.axhline(0.5, color="grey", lw=0.5); a.set_title(f"P-{p}"); a.set_xlabel("layer"); a.legend(fontsize=7)
ax[0].set_ylabel("AUROC (held-out items)")
plt.tight_layout(); plt.savefig(os.path.join(RES, f"fig_auroc{TAG}.png"), dpi=130)
json.dump(out, open(os.path.join(RES, f"probes{TAG}.json"), "w"), indent=1)
