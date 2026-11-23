# ShieldAI - Data Directory

This directory contains the dataset used to train and evaluate the ShieldAI spam classification model.

### Dataset: UCI SMS Spam Collection
- **Source:** [UCI Machine Learning Repository - SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)
- **Format:** Tab-separated text file with labels (`ham` or `spam`) and message strings.
- **Size:** 5,572 tagged SMS messages (4,516 ham, 642 spam after deduplication).

### Usage
The training script `train_model.py` automatically checks this folder for dataset files matching `.txt`, `.tsv`, or `.csv` formats.
