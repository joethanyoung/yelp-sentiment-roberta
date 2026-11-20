# Contributing

Thank you for your interest in improving this project. This repository is a
public, non-confidential sentiment modeling project built around the Yelp
Academic Dataset and a fine-tuned RoBERTa-large model.

## Good Contribution Areas

- Reproducibility improvements for the notebook or setup instructions
- Clearer model evaluation outputs and error analysis
- Inference examples for practical text analytics workflows
- Documentation improvements for dataset preparation
- Lightweight scripts that make evaluation easier to rerun
- Responsible reporting of bugs, model limitations, or unsafe outputs

## Before Opening a Pull Request

1. Open an issue describing the problem or proposed improvement.
2. Keep changes focused on one topic.
3. Do not include proprietary client data, private review data, credentials, or
   local environment secrets.
4. If you change model evaluation logic, include the metric outputs you used to
   check the change.
5. Keep examples reproducible with public data or clearly marked synthetic text.

## Development Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the notebook:

```text
yelp_sentiment_roberta.ipynb
```

The Hugging Face model can be loaded with:

```python
from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="Zhoyang/review-triage-roberta-large"
)
```

## Pull Request Checklist

- The change is limited to public, non-confidential material.
- Documentation has been updated if the workflow changed.
- Any added examples can run without private files.
- Any metric claims are backed by reproducible output or clearly marked as
  illustrative.

## Maintainer Review

This is a small maintainer-led project. I review contributions for correctness,
reproducibility, privacy, and whether the change makes the model easier for
others to inspect or reuse.
