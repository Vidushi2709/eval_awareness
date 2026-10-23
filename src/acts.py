"""Residual-stream activations of Qwen3-4B (4-bit) at the last prompt token, every layer.
Writes results/acts/<set>.npy [N, n_layers+1, d] float16 and results/acts/<set>.json (items, token counts).

    python src/acts.py                         # data/prompts.jsonl
    python src/acts.py data/prompts_test.jsonl results/acts_test
"""
import collections
import json
import os
import re
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "data", "prompts.jsonl")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "results", "acts")
MODEL = os.environ.get("MODEL", "Qwen/Qwen3-4B")

# Run 06 found llama3 opens 199/200 eval chat rewrites with "hey ai" in lowercase but only 94/200 real ones.
# So every chat-format prompt (eval, real, raw WildChat, synthetic alike) loses its opening greeting and is lowercased.
GREETING = re.compile(r"^\s*(hey|hi|hello|hallo|dear|greetings)\b(\s+(ai|chatbot|there|assistant|bot)\b)?"
                      r"([^,.!?\n]{0,20}[,.!?])?\s*", re.I)


def norm(set_name, text):
    return GREETING.sub("", text).lower() if "chat" in set_name else text


assert norm("E-chat-rw", "hey ai can diabetics eat sweets?") == "can diabetics eat sweets?"
assert norm("SYN-chat", "Hey, I'm a nurse") == "i'm a nurse"
assert norm("R-chat-rw", "hiking boots?") == "hiking boots?"
assert norm("R-bench-rw", "Question: Hi?") == "Question: Hi?"

# Loader copied from attack/white_box_gcg/src/judge.py (Judge.__init__).
tok = AutoTokenizer.from_pretrained(MODEL)
q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=q, torch_dtype=torch.float16,
                                             device_map="cuda").eval()

sets = collections.defaultdict(list)
for r in map(json.loads, open(SRC, encoding="utf-8")):
    sets[r["set"]].append(r)

os.makedirs(OUT, exist_ok=True)
for name, rows in sets.items():
    acts, ntok = [], []
    for r in rows:
        p = tok.apply_chat_template([{"role": "user", "content": norm(name, r["text"])}], tokenize=False,
                                    add_generation_prompt=True, enable_thinking=False)
        ids = tok(p, return_tensors="pt").input_ids.to(model.device)
        with torch.no_grad():
            hs = model(ids, output_hidden_states=True).hidden_states
        acts.append(torch.stack([h[0, -1] for h in hs]).cpu().half().numpy())
        ntok.append(ids.shape[1])
    np.save(os.path.join(OUT, f"{name}.npy"), np.stack(acts))
    json.dump({"items": [r["item"] for r in rows], "ntok": ntok, "text": [norm(name, r["text"]) for r in rows]}, open(os.path.join(OUT, f"{name}.json"), "w"))
    print(name, np.stack(acts).shape, flush=True)
