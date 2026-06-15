\# ITS Fine-tuning



Fine-tuning smaller LLMs to generate calibrated buggy Python code for 

Intelligent Tutoring Systems (ITS) based on the Productive Failure 

learning principle (Kapur).



\## Approach

Uses NAT (Negative-example Aware Training) — training examples are prefixed 

with \[NEGATIVE] for buggy code and \[POSITIVE] for correct code.



\## Models

\- \*\*Model A\*\* (`madane007/stratergy2\_qwen`): trained on strategy2 data only — 

&#x20; explicit bug-type conditioning

\- \*\*Model C\*\* (`madane007/qwen-its-combined`): trained on strategy1 + strategy2 — 

&#x20; personality + bug-type conditioning



\## Repo Structure

\- `src/` — reusable Python modules (extraction, prompts, inference)

\- `configs/` — hyperparameters for each training run

\- `notebooks/` — Colab notebooks for data extraction and training

\- `data/` — ignored by git; store SQL dump and JSONL files locally or on Drive



