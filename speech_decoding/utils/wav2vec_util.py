import torch
import numpy as np
from tqdm import tqdm
from termcolor import cprint


def process_chunk(wav2vec, audio_chunk, mode="huggingface"):
    with torch.no_grad():
        out = wav2vec(input_values=audio_chunk, output_hidden_states=True)
        out = out.hidden_states[-4:]
    a = [l.detach() for l in out]
    return torch.stack(a).mean(axis=0)


def get_last4layers_avg(wav2vec, waveform: torch.Tensor, atonce=True):
    if atonce:
        return process_chunk(wav2vec, waveform).detach()

    else:
        # NOTE: can't process the entire waveform in one go, so we do chunk-by-chunk
        # FIXME: the number of embeddings is slightly different than if you feed the entire waveform though the feature_extractor
        splits = np.array_split(list(range(waveform.shape[-1])), 10)
        embeddings = []
        pbar = tqdm(splits, desc="Pre-computing W2V embeddings (last 4 layers)")
        for split in pbar:
            out = process_chunk(wav2vec, waveform[0, split].unsqueeze(0))
            embeddings.append(out.detach())
        return torch.vstack(
            [e.squeeze() for e in embeddings]
        ).t()  # before vstack: torch.Size([1, 3617, 1024])
