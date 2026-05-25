# Yelp Sentiment Analysis — Fine-tuned RoBERTa-large

Fine-tuned `roberta-large` on 240K Yelp reviews for 3-class sentiment classification (Negative / Neutral / Positive). Achieved **93.76% accuracy** and **weighted F1 of 0.9389** on a held-out test set of 40,000 samples.

🤗 **Model on Hugging Face:** [Zhoyang/yelp-sentiment-roberta-large](https://huggingface.co/Zhoyang/yelp-sentiment-roberta-large)

---

## Results

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Negative | 0.9393 | 0.9261 | 0.9327 |
| Neutral | 0.6934 | 0.7661 | 0.7280 |
| Positive | 0.9789 | 0.9674 | 0.9731 |
| **Weighted Avg** | **0.9407** | **0.9376** | **0.9389** |

### Confusion Matrix
```
               Predicted
               Neg    Neu    Pos
Actual  Neg  [ 7712   585    30]
        Neu  [  442  3223   542]
        Pos  [   56   840 26570]
```

---

## Dataset

- **Source:** Yelp Academic Dataset (500K reviews)
- **Label mapping:** 1–2 stars → Negative · 3 stars → Neutral · 4–5 stars → Positive
- **Training samples:** 239,870 (stratified split)
- **Test samples:** 40,000 (stratified, held-out)

---

## Training Details

| Parameter | Value |
|---|---|
| Base model | `roberta-large` (355M parameters) |
| Max token length | 512 |
| Class weights (Loss) | Negative 3.0 · Neutral 8.0 · Positive 1.5 |
| Hardware | NVIDIA A4000 (Paperspace Gradient) |

Class-weighted loss was applied to address the natural imbalance in Yelp data (~55% positive reviews).

---

## Quick Start

```python
from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="Zhoyang/yelp-sentiment-roberta-large"
)

id2label = {"LABEL_0": "Negative", "LABEL_1": "Neutral", "LABEL_2": "Positive"}

result = classifier("The food was absolutely amazing!")
print(id2label[result[0]["label"]], round(result[0]["score"], 4))
# Positive 0.9741
```

---

## Notebook

[`yelp_sentiment_roberta.ipynb`](./yelp_sentiment_roberta.ipynb) contains the full evaluation pipeline:
- Data loading and label mapping
- Stratified train/test split
- EDA: star distribution, token length analysis, truncation rates
- Custom PyTorch Dataset and DataLoader
- Batch inference on GPU
- Classification report and confusion matrix

> **Setup:** Set `HF_TOKEN` as an environment variable before running.

---

## Tech Stack

`Python` · `PyTorch` · `Transformers` · `scikit-learn` · `pandas` · `matplotlib`

---

*Developed as part of Georgia Tech OMSA Practicum, Spring 2026 — PathSynch Labs Pod C.*
