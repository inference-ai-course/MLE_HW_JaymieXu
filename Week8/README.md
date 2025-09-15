# Week 8 Homework - Reward Model Training and Evaluation

This repository contains the implementation for Week 8 homework on reward modeling for summarization tasks.

## Overview

We trained a reward model using DeBERTa-v3-base to evaluate summary quality based on human preference data. The model learns to distinguish between better and worse summaries for academic paper abstracts.
Model here: https://huggingface.co/Amie69/deberta-rm

## Files

- `reward_data.jsonl` - Training dataset with chosen/rejected summary pairs
- `eval_data.jsonl` - Evaluation dataset for testing the model
- `reward_training.py` - Training script for the reward model
- `eval_notebook.ipynb` - Evaluation notebook with ROUGE, BERTScore, and reward model metrics

## Training

The reward model was trained using:
- **Base Model**: microsoft/deberta-v3-base
- **Training Data**: 20 preference pairs from reward_data.jsonl
- **Framework**: Hugging Face TRL RewardTrainer
- **Epochs**: 3
- **Batch Size**: 8

## Evaluation

The evaluation compares:
- ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L)
- BERTScore metrics (Precision, Recall, F1)
- Reward model scores for chosen vs rejected summaries

## Model Hosting

**The trained reward model is hosted on Hugging Face: https://huggingface.co/Amie69/deberta-rm**

## Usage

1. Train the model:
   ```bash
   python reward_training.py
   ```

2. Run evaluation:
   ```bash
   jupyter notebook eval_notebook.ipynb
   ```

## Results

The reward model successfully learned to prefer chosen summaries over rejected ones, demonstrating effective preference learning for summary quality evaluation.