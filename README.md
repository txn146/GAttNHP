# GAttNHP: Temporal Knowledge Graph Event Prediction  Using Group Attention Neural Hawkes Process

This repository contains the official PyTorch implementation of the **Group Attention Neural Hawkes Process (GAttNHP)**.

GAttNHP is a unified framework for extrapolating future events in Temporal Knowledge Graphs (TKGs). It integrates a **Group-wise Mutual Excitation** mechanism to capture cross-chain dependencies and a **Non-Crossing Quantile (NCQ) Regression** module for robust time prediction.

## Usage

To train and evaluate the model, run the `main.py` script.

**General Command:**

```bash
python main.py --dataset <DATASET_NAME> --model <MODEL_NAME> --time_weight <BETA>

