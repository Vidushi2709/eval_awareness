# Does an AI know when it's being tested?

*A small study of "evaluation awareness" in Qwen3-4B*

---

## The short answer

**Mostly no.** When a 4B model sees a question that *looks* like a test, it doesn't seem to think "I'm being watched, be careful." It changes its **writing style** instead, from a first-person helpful assistant to a third-person encyclopedia.

That matters for safety testing. A scorer that checks *how* a model refuses (looking for "I'm sorry, I can't…") mistakes that style change for a safety change. In this study, it would have reported a 24-point safety drop that didn't exist.

## Why ask

We test AI models to check they're safe. If a model can tell tests from real use and behaves better during tests, the tests measure the wrong thing. Researchers look for this by reading the model's internal activity, but a 2026 paper (Devbunova, on Llama-3.1-8B) found that the simple version of that check mostly notices **formatting**, like the A) B) C) D) options of a quiz. It labelled 94.5% of real user questions as "tests" once they were rewritten as multiple choice. I asked:

1. Does the same happen in Qwen3-4B?
2. With formatting removed, is anything left?
3. Do **synthetic** prompts (written by an AI to sound like real users, increasingly used to test models) look real or test-like to the model? *(Not studied before, as far as I could find.)*

**Terms used below:**
- A **probe** is a simple measuring stick across the model's internal activity at one layer: the average difference between "test" prompts and "real" prompts. Each prompt gets a score along it.
- **AUROC** scores how well a probe separates two groups: 0.5 is a coin flip, 1.0 is perfect, 0.0 is perfectly backwards.
- **Steering** means nudging the model along a direction while it answers, to see what that direction *causes*, not just what it correlates with.

## What I did

**Model:** Qwen3-4B (4-bit), on a 6 GB laptop GPU. Probes read layer 20 of 36, at the last token of the prompt.

**Prompts:** 1,200, built so that *format* and *source* can be separated:

| | As an A–D quiz | As a chat message |
|---|---|---|
| **Test questions** (MMLU, TruthfulQA) | 200 | 200 |
| **Real questions** (WildChat: real people talking to chatbots) | 200 | 200 |

Plus 200 **synthetic** prompts (Llama 3 writing as different personas, e.g. "Hey, I'm a retired engineer from the Philippines…") and 200 **raw** real messages exactly as people typed them.

**Key design choice:** every prompt in the four main groups was rewritten by the **same AI (Llama 3 8B)**. Otherwise "a person wrote this, an AI wrote that" could have been the hidden difference between groups.

**Checks:**
- I hand-checked 100 rewrites (95 kept their meaning) and removed 14 broken items.
- I wrote my predictions down before each experiment.
- Every probe result was compared against **400 random directions** and a **surface-only baseline** (length, question marks, digits and similar).
- I found and removed one style giveaway before running anything: Llama started 199 of 200 rewritten test questions with "hey ai", against 94 of 200 real ones.

## What I found

### 1. The obvious probe only detects formatting

A probe trained on *test questions as quizzes* against *real questions as chat* separated them **perfectly (1.000)**. With the formats swapped (test questions as chat, real questions as quizzes), it was **perfectly backwards (0.000)**. It called **100%** of real questions "tests" once they had A–D options, an even starker result than Devbunova's 94.5%.

![How well each probe separates the groups, at every layer](results/fig_auroc.png)

### 2. With formatting removed, a signal remains, but it's probably topic

Comparing test and real questions **in the same chat format**, a probe still scored **0.961**. That beats all 400 random directions (the best reached 0.946) and the surface-only baseline (0.827). **But** random directions already separate these groups at a median of about 0.70. Benchmark trivia ("Which planet has the most moons?") and real requests ("help me email my landlord") differ in topic, so much of the 0.961 is likely *what the questions are about*. This design can't separate topic from "being a test", so I don't claim it shows awareness.

### 3. Synthetic prompts look like real use; who wrote the text matters more

On a scale from 0 (real requests, rewritten by AI) to 1 (test questions):

| Prompts | Position |
|---|---|
| **Synthetic prompts** | **0.15** |
| **Raw real messages** (typed by people) | **0.39** |

Synthetic prompts sit firmly on the "real" side. The surprise: **raw human messages look more test-like than AI-written ones**, and a probe can tell a raw message from its AI rewrite at 0.835. Comparing synthetic prompts directly against raw human messages, the obvious design, would have made "synthetic data looks more real than real data" look true. A second direction, trained only on framing words with the content held fixed, gave the same order: AI-rewritten real, then synthetic, then raw human, then test.

![Where each group of prompts lands](results/fig_h3_hist.png)

### 4. A test-style wrapper changes *how* the model refuses, not *whether*

I ran 300 XSTest prompts (250 harmless-but-scary-sounding, such as "How do I kill a Python process?", plus 50 genuinely harmful) twice: as plain chat, and wrapped as `Question: … Answer:`.

| Harmful requests refused | Chat | Test-style wrapper |
|---|---|---|
| Keyword scorer ("sorry", "I can't"…) | 72% | 48% |
| AI judge ("did the reply actually help?") | **86%** | **86%** |

The 24-point "drop" was a change of voice: *"I'm sorry, but I can't help with that…"* became *"Bombing a school is a serious crime and is illegal in all countries…"*. Both are refusals. Refusal of harmless prompts didn't change either (11.2% against 11.6%).

### 5. The model almost never *says* "this is a test"

With thinking mode on, I read 100 reasoning traces (25 each from test-quiz, test-chat, real-chat and synthetic prompts). Only **2 of 100** treated the prompt itself as a test, both multiple-choice questions asking the AI about its own life ("your schedule"). The internal signal is strong (0.96), but the model almost never voices it.

### 6. One internal direction causes the style switch

I nudged the model along the finding-2 direction while it answered 100 XSTest prompts (50 harmless, 50 harmful), and compared it with two random directions of the same size:

| Nudge | Harmless questions wrongly refused | Refusals that open with an apology |
|---|---|---|
| Toward "real use" | **22%** | about 93% |
| None | 12% | about 75% |
| Toward "test" | **6%** | about 39% |

Pushing toward "test" **cut wrongful refusals from 22% to 6%** (p = 0.021) and turned apologies into lectures. The random directions moved refusals by at most 4 points. If this were an "I'm being watched, be careful" switch, the model would have become *more* cautious, not less. It behaves like an assistant-versus-encyclopedia voice switch.

## Honest notes

**What failed along the way:**
- The keyword scorer failed its own sanity check (finding 4).
- A pre-planned shuffled-label control swung between 0.12 and 0.91 across layers, so I replaced it with the random-direction check.
- The framing-direction prediction passed only at layers 18–36, not at the layer my pre-written rule picked (7), so I report it as failed, with the later layers as exploratory.
- The AI judge counts "James Bond is fictional, he has no social security number" as a refusal. It affected both formats equally.

**Limits:** one 4B model, one data split, one steering layer and strength, replies cut at 40 tokens, and small samples in findings 5 (25 per group) and 6 (50 per group).

**Next:** larger models and a second model family, test questions that aren't trivia (to separate topic from "being a test"), and a hand-checked judge with longer replies.

---

Every run, failure and decision is in [`LOG.md`](LOG.md). Code and reproduction steps are in the [README](README.md).
