"""Day 5: Qwen3-4B answers XSTest (250 safe + 50 unsafe) in two wrappers that differ only in format.
Saves the reply and the last-prompt-token activations. Resumable: finished rows in results/behave.jsonl are skipped.

    python src/behave.py          # all 300 x 2
    python src/behave.py 3        # smoke test: first 3 prompts, writes results/behave_test.jsonl
"""
import json
import os
import sys

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
N = int(sys.argv[1]) if len(sys.argv) > 1 else None
TAG = "_test" if N else ""
OUT = os.path.join(ROOT, "results", f"behave{TAG}.jsonl")
ACT = os.path.join(ROOT, "results", f"behave_acts{TAG}")
REV = "f600c994b256f12867dfa5b3eb3d545a3e62f8b5"
WRAP = {"chat": "{p}", "bench": "Question: {p}\nAnswer:"}

# Loader copied from attack/white_box_gcg/src/judge.py (Judge.__init__), as in acts.py.
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B", quantization_config=q, torch_dtype=torch.float16,
                                             device_map="cuda").eval()

xs = list(load_dataset("Paul/XSTest", split="train", revision=REV))
# ponytail: every 4th unsafe prompt (50 of 200) as the refusal-rule check; all 250 safe ones are the test
xs = [x for x in xs if x["label"] == "safe"] + [x for x in xs if x["label"] == "unsafe"][::4]
xs = xs[:N]
done = set()
if os.path.exists(OUT):
    done = {(r["id"], r["format"]) for r in map(json.loads, open(OUT, encoding="utf-8"))}
os.makedirs(ACT, exist_ok=True)

with open(OUT, "a", encoding="utf-8") as f:
    for n, x in enumerate(xs):
        for fmt, w in WRAP.items():
            if (x["id"], fmt) in done:
                continue
            p = tok.apply_chat_template([{"role": "user", "content": w.format(p=x["prompt"])}], tokenize=False,
                                        add_generation_prompt=True, enable_thinking=False)
            enc = tok(p, return_tensors="pt").to(model.device)
            ids = enc.input_ids
            with torch.no_grad():
                hs = model(ids, output_hidden_states=True).hidden_states
                gen = model.generate(**enc, max_new_tokens=40, do_sample=False)  # refusals show in the first sentence
            np.save(os.path.join(ACT, f"{x['id']}_{fmt}.npy"), torch.stack([h[0, -1] for h in hs]).cpu().half().numpy())
            reply = tok.decode(gen[0, ids.shape[1]:], skip_special_tokens=True)
            f.write(json.dumps(dict(id=x["id"], format=fmt, label=x["label"], type=x["type"], prompt=x["prompt"],
                                    reply=reply), ensure_ascii=False) + "\n")
            f.flush()
        if n % 25 == 0:
            print(f"{n}/{len(xs)}", flush=True)
