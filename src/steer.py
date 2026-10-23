"""Day 6 part C: add alpha * M * unit(direction) to the residual stream after decoder layer 19 (= hidden_states[20],
the layer the probes use) at every position while Qwen answers XSTest in chat format. M is P-source's
eval-minus-real gap on held-out chat cells, the same magnitude for every direction. Resumable; writes
results/steer.jsonl with format = "<direction>_<alpha>" so judge.py can label it.

    python src/steer.py
"""
import json
import os
import sys

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.argv = sys.argv[:1]
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "probes.py"), encoding="utf-8").read()
exec(src[: src.index("PROBES = {")])
LAYER, OUT = 20, os.path.join(RES, "steer.jsonl")

unit = lambda v: v / np.linalg.norm(v)
d_src = unit(direction(["E-bench-rw", "E-chat-rw"], ["R-bench-rw", "R-chat-rw"], LAYER)[0])
M = float(part("E-chat-rw", False, LAYER).mean(0) @ d_src - part("R-chat-rw", False, LAYER).mean(0) @ d_src)
g = np.random.default_rng(2)
DIRS = {"psource": d_src, "framing": unit(np.load(os.path.join(RES, "framing_dir.npy"))[LAYER]),
        "rand1": unit(g.standard_normal(d_src.shape[0])), "rand2": unit(g.standard_normal(d_src.shape[0]))}
CONDS = [("none", 0.0)] + [(k, a) for k in DIRS for a in (-2.0, 2.0)]
print("M =", M, flush=True)

xs = list(load_dataset("Paul/XSTest", split="train", revision="f600c994b256f12867dfa5b3eb3d545a3e62f8b5"))
xs = ([x for x in xs if x["type"] in ("safe_contexts", "figurative_language")]
      + [x for x in xs if x["label"] == "unsafe"][::4])  # the same 50 unsafe prompts as Day 5

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B", quantization_config=q, torch_dtype=torch.float16,
                                             device_map="cuda").eval()
add = {"v": None}


def hook(_, __, output):
    if add["v"] is None:
        return output
    if isinstance(output, tuple):
        return (output[0] + add["v"],) + output[1:]
    return output + add["v"]


model.model.layers[LAYER - 1].register_forward_hook(hook)  # hidden_states[20] is decoder layer 19's output

# Check the push enters the residual stream. transformers 4.57 records hidden_states inside each layer's wrapped
# forward, i.e. BEFORE this hook runs (Run 22), so hidden_states[LAYER] never shows it; the next layer's output must.
probe_ids = tok("hello", return_tensors="pt").input_ids.to(model.device)
with torch.no_grad():
    h0 = model(probe_ids, output_hidden_states=True).hidden_states[LAYER + 1][0, -1].float()
    u = torch.tensor(DIRS["rand1"], device=model.device).float()
    add["v"] = (10 * u).half()
    h1 = model(probe_ids, output_hidden_states=True).hidden_states[LAYER + 1][0, -1].float()
    add["v"] = None
shift = h1 - h0  # fp16 residuals are large, so check the push's size and direction, not exact equality
print("layer check: projection", float(shift @ u), "cosine", float(torch.cosine_similarity(shift, u, dim=0)), flush=True)
# one layer later the +10u is carried by the residual connection, plus layer 20's (small) reaction to it
assert 7 < float(shift @ u) < 13 and float(torch.cosine_similarity(shift, u, dim=0)) > 0.5, "push not in residual stream"

done = {(r["id"], r["format"]) for r in map(json.loads, open(OUT, encoding="utf-8"))} if os.path.exists(OUT) else set()
with open(OUT, "a", encoding="utf-8") as f:
    for name, a in CONDS:
        key = f"{name}_{a:+.0f}" if name != "none" else "none_0"
        add["v"] = None if name == "none" else torch.tensor(a * M * DIRS[name], dtype=torch.float16, device=model.device)
        for x in xs:
            if (x["id"], key) in done:
                continue
            p = tok.apply_chat_template([{"role": "user", "content": x["prompt"]}], tokenize=False,
                                        add_generation_prompt=True, enable_thinking=False)
            enc = tok(p, return_tensors="pt").to(model.device)
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=40, do_sample=False)
            reply = tok.decode(gen[0, enc.input_ids.shape[1]:], skip_special_tokens=True)
            f.write(json.dumps(dict(id=x["id"], format=key, label=x["label"], type=x["type"], prompt=x["prompt"],
                                    reply=reply), ensure_ascii=False) + "\n")
            f.flush()
        print(key, "done", flush=True)
