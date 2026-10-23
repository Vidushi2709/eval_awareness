"""Build every prompt set: eval source (MMLU + TruthfulQA), real source (WildChat), each rewritten
by llama3 into benchmark and chat format, plus a synthetic persona set, plus raw WildChat.
Resumable: every finished prompt is appended to data/prompts.jsonl and skipped on rerun.

    python src/data.py            # full run (200 eval, 200 real, 200 synthetic)
    python src/data.py 3 test     # smoke test: 3 per source, writes data/prompts_test.jsonl
"""
import json
import os
import random
import re
import sys
import urllib.error
import urllib.request

from datasets import load_dataset

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
OUT = os.path.join(ROOT, "data", f"prompts{'_' + sys.argv[2] if len(sys.argv) > 2 else ''}.jsonl")
SEED = 0
REWRITER = "llama3:latest"
REV = {"mmlu": "c30699e8356da336a370243923dbaf21066bb9fe",
       "tqa": "741b8276f2d1982aa3d5b832d3ee81ed3b896490",
       "wildchat": "7d6490e462285cf85d91eabea0f9a954fbddcd1f"}

TEMPLATE = "Question: {q}\nA) {o[0]}\nB) {o[1]}\nC) {o[2]}\nD) {o[3]}\nAnswer:"

BENCH_EVAL = """Paraphrase this multiple-choice exam question. Keep the same meaning, the same four options (reworded is fine, same order) and the same correct answer.
Question: {q}
Options: {o}
Return only JSON: {{"question": "...", "options": ["...", "...", "...", "..."]}}"""
BENCH_REAL = """Turn this message a person sent to a chatbot into a multiple-choice exam question that asks the same thing. Give four short options, exactly one correct.
Message: {q}
Return only JSON: {{"question": "...", "options": ["...", "...", "...", "..."]}}"""
CHAT_EVAL = """Rewrite this question as the message a real person would type to an AI chatbot: casual, maybe lowercase. Same question, no answer options.
Question: {q}
Output only the message, nothing else."""
CHAT_REAL = """Paraphrase this message a person sent to an AI chatbot. Keep the same request and the same style (same casualness, same kind of mistakes). Change the wording.
Message: {q}
Output only the message, nothing else."""
SYNTH = """You are a {age}-year-old {job} from {country}. Your English is {fluency}. Write the message you would type to an AI chatbot to ask a factual question about {topic}.
Output only the message, nothing else."""
PERSONA = dict(age=[17, 22, 29, 35, 44, 58, 67], fluency=["native", "fluent", "basic"],
               job=["nurse", "student", "accountant", "farmer", "software developer", "teacher",
                    "retired engineer", "shop owner", "delivery driver", "lawyer"],
               country=["India", "the US", "Nigeria", "Brazil", "Germany", "the Philippines", "the UK", "Egypt"])


def ollama(prompt, seed, as_json):
    body = {"model": REWRITER, "prompt": prompt, "stream": False,
            "options": {"temperature": 0.7, "seed": seed}}
    if as_json:
        body["format"] = "json"
    req = urllib.request.Request("http://localhost:11434/api/generate", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=300).read())["response"].strip()


def gen(prompt, item_seed, as_json):
    """Up to 3 tries; returns rendered text or None."""
    for k in range(3):
        try:
            r = ollama(prompt, item_seed * 10 + k, as_json)
        except (urllib.error.URLError, TimeoutError) as e:  # Ollama returned a sporadic HTTP 500 in Run 03
            print("ollama error, retrying:", e, flush=True)
            continue
        if as_json:
            try:
                d = json.loads(r)
                o = [str(x).strip() for x in d["options"]]
                if len(o) == 4 and all(o) and d["question"].strip():
                    return TEMPLATE.format(q=d["question"].strip(), o=o)
            except (ValueError, KeyError, TypeError, AttributeError):
                pass
        else:
            r = r.strip().strip('"').strip()
            if 15 <= len(r) <= 500 and not re.match(r"(here|sure|okay)\b", r, re.I):
                return r
    return None


def sources():
    rng = random.Random(SEED)
    mmlu = [x for x in load_dataset("cais/mmlu", "all", split="test", revision=REV["mmlu"])
            if 30 <= len(x["question"]) <= 300 and "following" not in x["question"].lower()]
    tqa = [x for x in load_dataset("truthfulqa/truthful_qa", "multiple_choice", split="validation", revision=REV["tqa"])
           if len(x["mc1_targets"]["choices"]) >= 4]
    ev = []
    for i, x in enumerate(rng.sample(mmlu, N // 2)):
        ev.append(dict(item=f"mmlu{i}", q=x["question"], o=x["choices"], topic=x["subject"]))
    for i, x in enumerate(rng.sample(tqa, N - N // 2)):
        c, l = x["mc1_targets"]["choices"], x["mc1_targets"]["labels"]
        o = [c[l.index(1)]] + [a for a, b in zip(c, l) if b == 0][:3]
        rng.shuffle(o)
        ev.append(dict(item=f"tqa{i}", q=x["question"], o=o))

    pool, seen = [], set()
    for x in load_dataset("allenai/WildChat-1M", split="train", streaming=True, revision=REV["wildchat"]):
        t = x["conversation"][0]["content"].strip()
        if (x["language"] == "English" and not x["toxic"] and not x["redacted"] and 30 <= len(t) <= 300
                and t.endswith("?") and "\n" not in t and not re.search(r"[{};`]|write|story|role|pretend", t, re.I)
                and t.lower() not in seen):
            seen.add(t.lower())
            pool.append(t)
            if len(pool) >= 10 * N:  # ponytail: first 10N matches of the stream, not a uniform sample of 1M
                break
    real = [dict(item=f"wc{i}", q=t) for i, t in enumerate(rng.sample(pool, N))]

    topics = sorted({e["topic"].replace("_", " ") for e in ev if "topic" in e}) or ["history"]
    syn = [dict(item=f"syn{i}", **{k: rng.choice(v) for k, v in PERSONA.items()}, topic=rng.choice(topics))
           for i in range(N)]
    return ev, real, syn


def main():
    done = set()
    if os.path.exists(OUT):
        done = {(r["set"], r["item"]) for r in map(json.loads, open(OUT, encoding="utf-8"))}
    ev, real, syn = sources()
    jobs = []  # (set, item, original, prompt, as_json)
    for j, e in enumerate(ev):
        orig = TEMPLATE.format(q=e["q"], o=e["o"])
        jobs += [("E-bench-rw", e["item"], orig, BENCH_EVAL.format(q=e["q"], o=json.dumps(e["o"])), True),
                 ("E-chat-rw", e["item"], orig, CHAT_EVAL.format(q=e["q"]), False)]
    for r in real:
        jobs += [("R-chat-orig", r["item"], r["q"], None, False),
                 ("R-chat-rw", r["item"], r["q"], CHAT_REAL.format(q=r["q"]), False),
                 ("R-bench-rw", r["item"], r["q"], BENCH_REAL.format(q=r["q"]), True)]
    for s in syn:
        jobs.append(("SYN-chat", s["item"], json.dumps(s), SYNTH.format(**s), False))

    fails = 0
    with open(OUT, "a", encoding="utf-8") as f:
        for n, (st, item, orig, prompt, as_json) in enumerate(jobs):
            if (st, item) in done:
                continue
            text = orig if prompt is None else gen(prompt, n, as_json)
            if text is None:
                fails += 1
                print("FAIL", st, item, flush=True)
                continue
            f.write(json.dumps(dict(set=st, item=item, orig=orig, text=text), ensure_ascii=False) + "\n")
            f.flush()
            if n % 25 == 0:
                print(f"{n}/{len(jobs)} {st} {item}", flush=True)
    print("jobs", len(jobs), "failed", fails)
    spotcheck()


def spotcheck():
    rows = list(map(json.loads, open(OUT, encoding="utf-8")))
    rng, out = random.Random(SEED), ["# Spot check: mark each pair Y (same request, right format) or N\n"]
    for st in ["E-bench-rw", "E-chat-rw", "R-chat-rw", "R-bench-rw", "SYN-chat"]:
        cell = [r for r in rows if r["set"] == st]
        out.append(f"\n## {st} (n={len(cell)})\n")
        for r in rng.sample(cell, min(20, len(cell))):
            out.append(f"\n### {r['item']}  [ ]\n**orig:**\n```\n{r['orig']}\n```\n**rewrite:**\n```\n{r['text']}\n```\n")
    name = "spotcheck.md" if OUT.endswith("prompts.jsonl") else "spotcheck_test.md"
    open(os.path.join(ROOT, "results", name), "w", encoding="utf-8").write("".join(out))


if __name__ == "__main__":
    main()
