# Reimplementation of speech decoding paper by MetaAI

Paper: https://arxiv.org/pdf/2208.12266.pdf

<div align="center"><img src="overview_meta2022.png" width=300></div>

## Status

Works for Gwilliams2022 dataset and Brennan2018 dataset.

## TODOs

* Perfectly align speech and EEG  
  - [x] Encode 3s chunks of speech every iteration, not doing it beforehand
  - [x] Solve the problem of EEG recording having slightly different lengths (find how timings correspond each other)
  - [x] Shift EEG 150ms to the future

* Make subject layer each for subject
  - [x] Basic implementation
  - [ ] Optimize above (unuse for-loop)

- [x] Exclude excluded subjects in the paper

- [x] Robust scaling for EEG

- [x] Correct CLIP Loss (maybe done)

- [ ] The paper says "1x1 convolution" but how do we apply 1x1 convolution to sequences?

- [x] w2v2 expects audio sampled @ 16 kHz

- [ ] It may happen that one batch has more than one segment. This shouldn't break CLIPLoss completely, but it has to deal with it somehow.
- [x] `--wandb` tells the `train.py` to log progress to wandb.
- [x] `--force_recompute` tells `dataset.py` to pre-process EEG and audio again.
- [x] Done for Brennan2018, but should be REVISITED. Upsample embeddings to 120 Hz (instead of ~50 Hz now). FIX: resample exactly to 120 Hz (not 119)
- [ ] Use embeddings at the output of the last 4 transformer blocks instead of the feature extractor.
- [ ] Subject selection based on comprehension score in Brennan2018.

# Usage

## For EEG (Brennan 2018 et al.)
Run `python train.py --config brennan2018.yml --wandb --force_recompute`.

## Dataset

4 datasets were used in the paper (2 EEG and 2 MEG). I implemented preprocessing for Gwilliams2022 (MEG) and Brennan2018 (EEG). As in the paper, the model learns MEG much better.

**Gwilliams et al., 2022**

- Paper https://arxiv.org/abs/2208.11488

- Dataset https://osf.io/ag3kj/

**Brennan et al., 2019**

- Paper https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0207741

- Dataset https://deepblue.lib.umich.edu/data/concern/data_sets/bg257f92t

You will need `S01.mat` to `S49.mat` placed under `data/Brennan2018/raw/` to run the code.

I provide merged version of the audio files [here](https://drive.google.com/file/d/1qXyDFHhIKw7e-llEklLh02D6DuSTTqFg/view?usp=sharing). Place it under `data/Brennan2018/`.

## wav2vec 2.0

`wav2vec2-large-xlsr-53` model was used for speech embedding. You will need to download `xlsr_53_56k.pt` from [here](https://github.com/facebookresearch/fairseq/tree/main/examples/wav2vec) and place it under `weights/`.