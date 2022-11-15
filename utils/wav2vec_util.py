from lib2to3.pytree import HUGE
import os
import torch
import numpy as np
from tqdm import tqdm
from termcolor import cprint


def load_wav2vec_model(wav2vec_model: str):
    """
    Loads either the FAIR implementation of Wav2Vec2 or the Huggingface implementation
    """

    if wav2vec_model == 'xlsr_53_56k':
        cprint("LOADING FAIR'S WAV2VEC2", 'red', 'on_yellow')
        from omegaconf import DictConfig, open_dict
        import fairseq

        cleared_cp_path = f"weights/{wav2vec_model}_cleared.pt"

        # pop unneeded keys from config dict of newer models so that I can load it without an error
        if (wav2vec_model != "wav2vec_small") and (not os.path.exists(cleared_cp_path)):
            cp_path = f"weights/{wav2vec_model}.pt"
            cp = torch.load(cp_path)
            unneeded_keys = [
                'eval_wer', 'eval_wer_config', 'eval_wer_tokenizer', 'eval_wer_post_process', 'autoregressive'
            ]
            cfg = DictConfig(cp['cfg'])
            with open_dict(cfg):
                for k in unneeded_keys:
                    cfg.task.pop(k)
            cp['cfg'] = cfg
            torch.save(cp, cleared_cp_path)  # overwrite .pt

        model, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([cleared_cp_path])

        return model[0]

    if wav2vec_model == 'facebook/wav2vec2-large-xlsr-53':
        cprint("LOADING HUGGINGFACE'S WAV2VEC2", 'red', 'on_yellow')
        from transformers import Wav2Vec2Model
        model = Wav2Vec2Model.from_pretrained(wav2vec_model)
        return model
    else:
        raise ValueError('Wrong model string.')


def getW2VLastFourLayersAvg(wav2vec, waveform, mode='huggingface'):

    def _process_chunk(wav2vec, audio_chunk, mode='huggingface'):
        with torch.no_grad():
            if mode == 'huggingface':
                out = wav2vec(input_values=audio_chunk, output_hidden_states=True)
                out = out.hidden_states[-4:]
            else:
                out = wav2vec(audio_chunk, features_only=True)
                out = out['layer_results'][-4:]
        a = []
        for i, l in enumerate(out):
            # NOTE: each layer returns `x, (attn, layer_result)` where layer_result is the output of a layer before
            # NOTE: dropout, adding the residual, and layer_norm. We'll use the ones before.
            if mode == 'huggingface':
                a.append(l.detach())  # to get layer activations before drop, layer_norm, residuals
            elif mode == 'before':
                a.append(l[2].detach())  # to get layer activations before drop, layer_norm, residuals
            elif mode == 'after':
                a.append(l[0].detach())  # to get layer activations AFTER drop, layer_norm, residuals
            else:
                raise ValueError('Check configs. Must be one of: huggingface, before, after.')
        return torch.stack(a).mean(axis=0)

    # NOTE: can't process the entire waveform in one go, so we do chunk-by-chunk
    # FIXME: the number of embeddings is slightly different than if you feed the entire waveform though the feature_extractor
    splits = np.array_split(list(range(waveform.shape[-1])), 10)
    embeddings = []
    pbar = tqdm(splits, desc="Pre-computing W2V embeddings (last 4 layers)")
    for split in pbar:
        out = _process_chunk(wav2vec, waveform[0, split].unsqueeze(0), mode=mode)
        embeddings.append(out.detach())
    if not mode == 'huggingface':
        return torch.vstack(embeddings).permute(1, 2, 0).squeeze()  # before vstack: torch.Size([3617, 1, 1024])
    else:
        return torch.vstack([e.squeeze() for e in embeddings]).t()  # before vstack: torch.Size([1, 3617, 1024])
