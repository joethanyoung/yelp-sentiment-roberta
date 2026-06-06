# Roadmap

This roadmap tracks practical maintenance work that would make the model easier
to inspect, reproduce, and reuse.

## Current Status

- Fine-tuned RoBERTa-large sentiment model is available on Hugging Face.
- Public GitHub repository includes the evaluation notebook, README, and setup
  requirements.
- Model card metadata is available on Hugging Face with MIT license metadata.

## Near-Term Maintenance

- Add a small command-line inference script for batch sentiment scoring.
- Save evaluation outputs as versioned files, including classification report
  and confusion matrix.
- Add a short error-analysis section for neutral-class misclassification.
- Add setup notes for CPU-only inference and lower-memory environments.

## Reproducibility Improvements

- Pin or document tested dependency versions more precisely.
- Add a lightweight smoke test for loading the Hugging Face model.
- Separate preprocessing, evaluation, and inference into reusable scripts.
- Document expected runtime and memory requirements.

## Responsible Use

- Add clearer notes on domain transfer limits outside Yelp-style reviews.
- Add examples of ambiguous neutral reviews and expected model behavior.
- Document privacy expectations for anyone adapting the workflow to real
  customer feedback.

## Longer-Term Ideas

- Evaluate smaller models for faster inference.
- Compare performance on another public review dataset.
- Add model monitoring examples for sentiment drift over time.
- Publish a minimal demo Space if maintenance time allows.
