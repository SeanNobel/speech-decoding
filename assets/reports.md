# Brennan2018

## Experiments

- Basically couldn't achieve performance in the paper.

<img src="brennan_curves.png">

- Segmenting was done with word onsets.

- Split

  - Sentence
    - 80% sentences were assigned to train

  - Shallow
    - 3s segments were assigned to train (80%) / test (20%) randomly
    - meaning there are actual overlaps between train and test

  - Deep
    - first 80% minutes of each session were assigned to train

- Scaling

  - Global
    - Robust Scaler was applied globally
    - meaning a certain channel with huge noise level can affect other channels

  - Channel-wise
    - Robust Scaler was applied channel-wise

## Questions

- After wav2vec2.0 embedding, audios become something like 50Hz (because wav2vec2.0 requires them to be originally 16kHz and it downsamples them a lot), so we need to upsample them to match brains' 120Hz. Do you actually do that? If so, which method do you use? We've tried linear interpolation by torchaudio and zero-padding but neither worked well.

- Learnable temperature of CLIP loss was not mentioned and absent in equation (2). It was mentioned in the original CLIP paper but do you actually use it?


# Gwilliams2022

## Experiments

