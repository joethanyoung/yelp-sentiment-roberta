# Calibrated Review Triage & Cost-Optimal Routing (RoBERTa-large)

Fine-tuned `FacebookAI/roberta-large` on 239,870 Yelp reviews for 3-class review classification (Negative 1–2★ / Neutral 3★ / Positive 4–5★), achieving **93.76% accuracy** and **0.9389 weighted F1** on a 40,000-review held-out test set—paired with a reproducible 500-review out-of-fold (OOF) probability calibration benchmark, selective human-review routing under capacity constraints, and a DKK review-cost versus missed-error trade-off calculator.

- **Model Card & Full Benchmark:** [`Zhoyang/review-triage-roberta-large`](https://huggingface.co/Zhoyang/review-triage-roberta-large)
- **Interactive Triage & Cost Trade-Off Space:** [`Zhoyang/review-triage-cost-routing`](https://huggingface.co/spaces/Zhoyang/review-triage-cost-routing)
- **Public 500-Review Evaluation Set:** [`Zhoyang/review-triage-500-eval`](https://huggingface.co/datasets/Zhoyang/review-triage-500-eval)
- **HF Collection:** [Calibrated Review Triage & Cost-Optimal Routing Suite](https://huggingface.co/collections/Zhoyang/calibrated-review-triage-and-cost-optimal-routing-suite-6ad605c8ded5824ee51efbd4)

---

## Project Context

This repository contains the public, non-confidential modeling and evaluation artifacts originally initiated during an NDA-covered analytics internship project with PathSynch Labs through the Georgia Tech OMSA Practicum (Spring 2026), followed by post-internship extensions on probability calibration, selective human-review routing, and review-cost trade-off analysis.

In customer support and reputation management workflows, raw three-class sentiment predictions are most useful when paired with a routing rule: deciding which high-confidence reviews can be routed automatically and which low-margin predictions near the 2–3-star boundary should be sent to human review under a fixed review budget.

Proprietary client data, stakeholder feedback, and client-specific implementation details are excluded under NDA constraints. All experiments, metrics, and benchmarks in this repository and linked Hugging Face artifacts use public Yelp data only.

---

## What This Project Demonstrates

- **Controlled transformer fine-tuning (`roberta-large`, 355M parameters):** Seven iterations from a `bert-base` baseline, where expanding sequence length (`256 → 512` tokens, `+0.012` weighted F1) and applying EDA-driven class weights (`Negative 3.0 / Neutral 8.0 / Positive 1.5`) lifted Neutral-class F1 from `0.612` to `0.728`.
- **Out-of-fold probability calibration:** 5-fold stratified temperature scaling on the pinned 500-review benchmark (`Zhoyang/review-triage-500-eval`) reduces Expected Calibration Error (ECE) from `0.0364` to `0.0147` and NLL from `0.2372` to `0.2184`.
- **Selective prediction & human-in-the-loop (HITL) triage:** Routing the lowest top-two probability-margin reviews to human review reduces the error rate among automatically routed reviews from `7.80%` (at `0%` review) to `4.86%` (at `~10%` review) and `2.75%` (at `~20%` review), while quantifying diminishing marginal returns (`1.85` reviews inspected per error caught at `0%→5%` vs. `4.67` cumulative at `20%`).
- **Decision-theoretic cost trade-off framework:** Parameterizes human review cost per item (benchmarked using Statistics Denmark `LONS20` 2024 standardized hourly earnings for customer service clerks: `246.15 DKK/hour` → `4.10 DKK/review` at 1 minute/review) against hypothetical downstream missed-error loss to evaluate total cost across `0%–30%` routing tiers.

---

## Evaluation 1: Original 40,000-Review Held-Out Test Set

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Negative (1–2★) | 0.9393 | 0.9261 | 0.9327 | 8,327 |
| Neutral (3★) | 0.6934 | 0.7661 | 0.7280 | 4,207 |
| Positive (4–5★) | 0.9789 | 0.9674 | 0.9731 | 27,466 |
| **Overall / Weighted** | **0.9407** | **0.9376 (Acc)** | **0.9389** | **40,000** |

### Confusion Matrix (40,000 Held-Out Reviews)
```text
               Predicted
               Neg    Neu    Pos
Actual  Neg  [ 7712   585    30]
        Neu  [  442  3223   542]
        Pos  [   56   840 26570]
```

### Iteration History

| Version | Key Change | Weighted F1 | Neutral F1 |
|:---|:---|:---:|:---:|
| v1 | BERT-base baseline | 0.881 | 0.514 |
| v3 | RoBERTa-base | 0.892 | 0.573 |
| v4 | `max_len` 256→512 | 0.904 | 0.604 |
| v6 | RoBERTa-large | 0.912 | 0.612 |
| **Final** | **EDA-driven class weights (3.0 / 8.0 / 1.5)** | **0.9389** | **0.728** |

---

## Evaluation 2: Public 500-Review Calibration & Triage Benchmark

To enable full reproducibility of post-hoc calibration and routing thresholds, a fixed 500-review reference set is published at [`Zhoyang/review-triage-500-eval`](https://huggingface.co/datasets/Zhoyang/review-triage-500-eval) (pinned revision `a2aa98f96cc2e829904f47984747667d3be53663`). Errors concentrate sharply at the 2–3-star boundary (`20.00%` error rate on 2★ and `46.00%` on 3★ vs. `1.18%` on 1★ and `1.20%` on 5★).

### Out-of-Fold Threshold Routing (`0%–30%` Human Review Capacity)

| Reference budget | OOF flagged share | Auto-route n | Auto-route error rate | Errors in review queue |
| --- | --- | --- | --- | --- |
| 0% | 0.0% | 500 | 7.80% | 0 |
| 5% | 4.8% | 476 | 5.46% | 13 |
| 10% | 9.4% | 453 | 4.86% | 17 |
| 15% | 15.2% | 424 | 3.77% | 23 |
| 20% | 20.0% | 400 | 2.75% | 28 |
| 25% | 24.6% | 377 | 2.12% | 31 |
| 30% | 30.8% | 346 | 1.73% | 33 |

> **Scope & Boundaries:** Auto-route error rates measure only reviews handled automatically; human reviewer accuracy is unmeasured (the cost trade-off calculator in the [Interactive Space](https://huggingface.co/spaces/Zhoyang/review-triage-cost-routing) treats human review as 100% accurate as a scenario upper bound and uses user-supplied hypothetical cost weights on this 500-review sample). It does not measure customer churn, retention uplift, or production business savings.

---

## Training Details

| Parameter | Value |
|---|---|
| Base model | `FacebookAI/roberta-large` (355M parameters) |
| Training / Held-out test samples | 239,870 / 40,000 (stratified split from 499,229 cleaned Yelp reviews) |
| Max token length | 512 |
| Class weights (Loss) | Negative 3.0 · Neutral 8.0 · Positive 1.5 |
| Hardware | NVIDIA RTX A4000 (Paperspace Gradient) |

---

## Quick Start

```python
from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="Zhoyang/review-triage-roberta-large",
    revision="2e39ef29d577aa84cfe70596020eb40e708b88a0",
    top_k=None,
    truncation=True,
    max_length=512,
)

scores = classifier("The food was okay, but the service was slow.")
print(scores)
```

---

## Notebook & Repository Files

- [`yelp_sentiment_roberta.ipynb`](./yelp_sentiment_roberta.ipynb): Full training and 40,000-review evaluation pipeline (EDA, token-length analysis, weighted-loss training, classification report, and confusion matrix).
- License: [MIT](./LICENSE) · Contributions: [CONTRIBUTING.md](./CONTRIBUTING.md) · Roadmap: [ROADMAP.md](./ROADMAP.md) · Security: [SECURITY.md](./SECURITY.md)
