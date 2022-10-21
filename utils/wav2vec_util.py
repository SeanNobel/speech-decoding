import os
import torch
from omegaconf import DictConfig, open_dict
import fairseq
import numpy as np
from tqdm import tqdm


def load_wav2vec_model(wav2vec_model: str):
    cleared_cp_path = f"weights/{wav2vec_model}_cleared.pt"

    # pop unneeded keys from config dict of newer models so that I can load it without an error
    if (wav2vec_model != "wav2vec_small") and (not os.path.exists(cleared_cp_path)):
        cp_path = f"weights/{wav2vec_model}.pt"
        cp = torch.load(cp_path)
        unneeded_keys = ['eval_wer', 'eval_wer_config', 'eval_wer_tokenizer', 'eval_wer_post_process', 'autoregressive']
        cfg = DictConfig(cp['cfg'])
        with open_dict(cfg):
            for k in unneeded_keys:
                cfg.task.pop(k)
        cp['cfg'] = cfg
        torch.save(cp, cleared_cp_path)  # overwrite .pt

    model, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([cleared_cp_path])

    return model[0]


def getW2VLastFourLayersAvg(wav2vec, waveform, before=True):

    def _process_chunk(wav2vec, audio_chunk, before=before):
        with torch.no_grad():
            out = wav2vec(audio_chunk, features_only=True)
        a = []
        for i, l in enumerate(out['layer_results'][-4:]):
            # NOTE: each layer returns `x, (attn, layer_result)` where layer_result is the output of a layer before
            # NOTE: dropout, adding the residual, and layer_norm. We'll use the ones before.
            if before:
                a.append(l[2].detach())  # to get layer activations before drop, layer_norm, residuals
            else:
                a.append(l[0].detach())  # to get layer activations AFTER drop, layer_norm, residuals
        return torch.stack(a).mean(axis=0)

    # NOTE: can't process the entire waveform in one go, so we do chunk-by-chunk
    # FIXME: the number of embeddings is slightly different than if you feed the entire waveform though the feature_extractor
    splits = np.array_split(list(range(waveform.shape[-1])), 10)
    embeddings = []
    pbar = tqdm(splits, desc="Computing W2V embeddings (last 4 layers)")
    for split in pbar:
        out = _process_chunk(wav2vec, waveform[0, split].unsqueeze(0), before=before)
        embeddings.append(out.detach())
    return torch.vstack(embeddings).permute(1, 2, 0).squeeze()
