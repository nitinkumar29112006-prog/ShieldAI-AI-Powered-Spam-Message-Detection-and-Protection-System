# ShieldAI - Models Directory

This directory contains the trained serialised machine learning artifacts.

### Included Artifact:
- `spam_pipeline.joblib`: Serialized Scikit-Learn pipeline dictionary containing:
  - `pipeline`: Feature extraction (word and character TF-IDF) combined with calibrated classifier.
  - `spam_threshold`: Optimal threshold tuned against validation split for maximum F1-score.
  - `minimum_words`: Minimum word limit to ensure high classification confidence.
  - `model_name`: Best-performing model selected during training.
