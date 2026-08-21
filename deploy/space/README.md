---
title: Peri Peri Fries Examination Engine
emoji: 🍟
colorFrom: red
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Peri-Peri Fries examination engine

The forensic examination engine behind
[team-peri-peri-fries](https://github.com/Jenak26/team-peri-peri-fries).

Upload a video exhibit and it is hashed, sealed read-only, examined by the trained
Stage A/B/C models, attacked along three laundering ladders, and sealed into a
replayable record with a nine-page report. Every upload produces its own examination;
nothing here is pre-generated.

An examination takes several minutes on the free CPU tier. That is the fragility
search re-encoding the exhibit and re-scoring it once per rung, which is the part that
makes the conclusion defensible.

Set `PERI_ALLOWED_ORIGINS` in the Space settings to the origin of the console that
calls this engine, for example `https://your-project.vercel.app`.
