# Natural Language Interface for Earthquake Analysis: A multimodal sensor-language system
Natural Language Interface for Earthquake Analysis- A multimodal sensor-language system for project in group 3.
This project proposes EQLM (Earthquake Large Language Model), a multimodal seismic-to-language framework that translates three-component seismic waveforms into natural-language earthquake reports.
This repository contains the core code for our project:

## Project Overview
The system combines:
- a pretrained seismic encoder (SeisLM),
- a learnable projection module(CNN/CNN-MOE/MOE/TCN-MOE),
- and a lightweight language decoder (TinyLlama with LoRA).

The training pipeline consists of two stages:
1. Contrastive alignment, which aligns seismic waveform representations with text embeddings
2. Instruction-tuned generation, which generates structured earthquake reports from waveform-conditioned prefix representations

## Repository Structure
### `notebook/`
Main experimental notebooks for different projector architectures:
- `eqlm-cnn-projector.ipynb` — CNN projector experiment
- `eqlm-moe-projector.ipynb` — MoE projector experiment
- `eqlm-multicnn-moe-projector.ipynb` — CNN-MoE projector experiment
- `eqlm-final-tcn-moe.ipynb` — final TCN-MoE experiment (best-performing model)

### `scripts/`
Utility and evaluation code:
- `generation_metrics_cell.py` — evaluation code for SGS, completeness, ROUGE-L, and BERTScore

### `figure/`
Result figures used in the report:
- `fig2.png`
- `losscurves.png`
- `result-ratio.png`

## Evaluation Metrics
The project evaluates generated earthquake reports using:
- SGS (Seismic Grounding Score)
- Completeness
- ROUGE-L
- BERTScore
- Categorical exact-match rates
- Numerical error metrics for magnitude, distance, P-arrival, and S-arrival

## Notes
Large raw datasets and pretrained checkpoints are not included in this repository.  
This repository is intended to provide the core experimental code, notebooks, and result figures for project reproducibility.
Dataset available on Hugging Face: https://huggingface.co/datasets/MIKEYUAN1/eqlm-earthquake-reports

## Authors
- **Linxuan Yuan**
- **Abir Hasan Bhuiyan**

  
