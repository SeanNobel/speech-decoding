import os
import torch
from omegaconf import DictConfig, open_dict
import fairseq


def load_wav2vec_model(wav2vec_model: str):
    cleared_cp_path = f"weights/{wav2vec_model}_cleared.pt"

    # pop unneeded keys from config dict of newer models so that I can load it without an error
    if (wav2vec_model != "wav2vec_small") and (not os.path.exists(cleared_cp_path)):
        cp_path = f"weights/{wav2vec_model}.pt"
        cp = torch.load(cp_path)
        unneeded_keys = ['eval_wer','eval_wer_config', 'eval_wer_tokenizer', 'eval_wer_post_process', 'autoregressive']
        cfg = DictConfig(cp['cfg'])
        with open_dict(cfg):
            for k in unneeded_keys:
                cfg.task.pop(k)
        cp['cfg'] = cfg
        torch.save(cp, cleared_cp_path) # overwrite .pt

    model, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([cleared_cp_path])

    return model[0]