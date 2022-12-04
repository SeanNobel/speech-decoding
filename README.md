# Reimplementation of speech decoding paper by MetaAI

Paper: https://arxiv.org/pdf/2208.12266.pdf

<div align="center"><img src="assets/overview_meta2022.png" width=300></div>

## Status

Works for Gwilliams2022 dataset and Brennan2018 dataset.

## TODOs

- [ ] Full reproducibility support. Will be useful for HP tuning. 
- [ ] Match accuracy to numbers reported in the paper. 

# Usage

## For EEG (Brennan et al. 2022)
Run `python train.py dataset=Brennan2018 rebuild_datasets=True`.
When `rebuild_datasets=False`, existing pre-processed M/EEG and pre-computing embeddings are used. This is useful if you want to run the model on exactly the same data and embeddings several times. Otherwise, the both audio embeddings are pre-computed and M/EEG data are pre-processed before training begins.

## For MEG (Gwilliams et al. 2022)

Run `python train.py dataset=Gwilliams2022 rebuild_datasets=True`
When `rebuild_datasets=False`, existing pre-processed M/EEG and pre-computing embeddings are used. This is useful if you want to run the model on exactly the same data and embeddings several times. It takes ~30 minutes to pre-process Gwilliams2022 and compute embeddings on 20 cores. Set `rebuild_datasets=False` for subsequent runs (or don't specify it, becuase by default `rebuild_datasets=False`). Otherwise, the both audio embeddings are pre-computed and M/EEG data are pre-processed before training begins.

## Monitoring training progress with W&B

To do that, set `entity` and `project` in the `wandb` section of `config.yaml`.

## Datasets

**Gwilliams et al., 2022**

- Paper https://arxiv.org/abs/2208.11488

- Dataset https://osf.io/ag3kj/

**Brennan et al., 2019**

- Paper https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0207741

- Dataset https://deepblue.lib.umich.edu/data/concern/data_sets/bg257f92t

You will need `S01.mat` to `S49.mat` placed under `data/Brennan2018/raw/` to run the code.