"""Day 6 part B: does Qwen3-4B *say* a prompt looks like a test? Thinking on, 25 prompts per set, greedy, up to 300
new tokens. Same normalised text the probes saw. Resumable; writes results/think.jsonl.

    python src/think.py
"""
import json
import os
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.argv = sys.argv[:1]
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "probes.py"), encoding="utf-8").read()
exec(src[: src.index("PROBES = {")])  # S (normalised texts, EXCLUDE applied) and the item split TR
OUT = os.path.join(RES, "think.jsonl")
SETS, K = ["E-bench-rw", "E-chat-rw", "R-chat-rw", "SYN-chat"], 25

pick = {}
for n in SETS:
    idx = np.flatnonzero(~TR[n])  # held-out items (all of SYN-chat), never used to train a probe
    pick[n] = np.random.default_rng(0).choice(idx, K, replace=False)

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B", quantization_config=q, torch_dtype=torch.float16,
                                             device_map="cuda").eval()
done = {(r["set"], r["item"]) for r in map(json.loads, open(OUT, encoding="utf-8"))} if os.path.exists(OUT) else set()
with open(OUT, "a", encoding="utf-8") as f:
    for n in SETS:
        for i in pick[n]:
            item, text = str(S[n]["items"][i]), S[n]["text"][i]
            if (n, item) in done:
                continue
            p = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                        add_generation_prompt=True, enable_thinking=True)
            enc = tok(p, return_tensors="pt").to(model.device)
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=300, do_sample=False)
            out = tok.decode(gen[0, enc.input_ids.shape[1]:], skip_special_tokens=False)
            f.write(json.dumps(dict(set=n, item=item, prompt=text, out=out), ensure_ascii=False) + "\n")
            f.flush()
        print(n, "done", flush=True)
