
import numpy as np
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image
from PIL import ImageFile  # 大きな画像もロード
import pickle
import json
import os, sys, random
import torch
import torch.nn as nn
from time import time
from tqdm import tqdm, trange
from termcolor import cprint
import pandas as pd

from torch.utils.data import DataLoader, RandomSampler, BatchSampler
try:
    from meg_decoding.models import get_model, Classifier
    from meg_decoding.utils.get_dataloaders import get_dataloaders, get_samplers
    from meg_decoding.dataclass.god import GODDatasetBase, GODCollator
    from meg_decoding.utils.loggers import Pickleogger
    from meg_decoding.utils.vis_grad import get_grad
    from torch.utils.data.dataset import Subset
    import matplotlib.pyplot as plt
    import seaborn as sns
except ModuleNotFoundError :
    pass




def get_language_model(prompt_dict:dict, savedir):
    if os.path.exists(os.path.join(savedir, 'text_features')):
        
        text_features = torch.load(os.path.join(savedir, 'text_features'))
        with open(os.path.join(savedir, 'prompts.txt'), 'r') as f:
            prompts = f.readlines()
    else:
        import clip
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=device)
        model = model.eval()
        prompts = []
        prefix = prompt_dict['prefix']
        for i, t in prompt_dict.items():
            if i == 'prefix':
                continue
            prompts.append(t+'\n')
        text = clip.tokenize([prefix + s.replace('\n','') for s in prompts]).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text)
        # with open(os.path.join(savedir, 'text_features'), 'wb') as f:
        torch.save(text_features, os.path.join(savedir, 'text_features'))
        with open(os.path.join(savedir, 'prompts.txt'), 'w') as f:
            f.writelines(prompts)
    return text_features, prompts

def evaluate(args, text_features, prompts, savedir, eval_sbj='1'):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from meg_decoding.utils.reproducibility import seed_worker
    # NOTE: We do need it (IMHO).
    if args.reproducible:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        g = torch.Generator()
        g.manual_seed(0)
        seed_worker = seed_worker
    else:
        g = None
        seed_worker = None
    source_dataset = GODDatasetBase(args, 'train', return_label=True)
    outlier_dataset = GODDatasetBase(args, 'val', return_label=True,
                                         mean_X= source_dataset.mean_X, # testデータの統計情報をしれない
                                         mean_Y=source_dataset.mean_Y,
                                         std_X=source_dataset.std_X,
                                         std_Y=source_dataset.std_Y
                                        )
    # import pdb; pdb.set_trace()
    text_features -= source_dataset.mean_Y
    text_features /= source_dataset.std_Y
    text_features = torch.Tensor(text_features).to(device)

    if eval_sbj == '1':
        ind_tr = list(range(100))# list(range(0, 3000)) + list(range(3600, 6600)) #+ list(range(7200, 21600)) # + list(range(7200, 13200)) + list(range(14400, 20400))
        ind_te = list(range(3000,3600)) + list(range(6600, 7200)) # + list(range(13200, 14400)) + list(range(20400, 21600))
        ind_out = list(range(0,50))
    elif eval_sbj == '2':
        ind_tr = list(range(100))# list(range(7200, 7200+3000)) + list(range(10800, 10800+3000)) 
        ind_te = list(range(7200+3000, 7200+3600)) + list(range(10800+3000, 10800+3600))
        ind_out = list(range(50,100))
    elif eval_sbj == '3':
        ind_tr = list(range(100))# list(range(14400, 14400+3000)) + list(range(14400+3600, 14400+6600)) 
        ind_te = list(range(14400+3000,14400+3600)) + list(range(14400+6600, 14400+7200)) 
        ind_out = list(range(100,150))
    else:
        ind_tr = list(range(0, 3000)) + list(range(3600, 6600))  + list(range(7200, 7200+3000))  + list(range(10800, 10800+3000)) + list(range(14400, 14400+3000)) + list(range(14400+3600, 14400+6600))
        ind_te = list(range(3000,3600)) + list(range(6600, 7200))  + list(range(7200+3000, 7200+3600)) + list(range(10800+3000, 10800+3600)) + list(range(14400+3000,14400+3600)) + list(range(14400+6600, 14400+7200)) 
        ind_out = list(range(0,150))
    outlier_dataset = Subset(outlier_dataset, ind_out)
    train_dataset = Subset(source_dataset, ind_tr)
    val_dataset   = Subset(source_dataset, ind_te)
    train_loader = DataLoader(
        train_dataset,
        batch_size= args.batch_size,
        drop_last=True,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
    )
    test_loader = DataLoader(
        # val_dataset, # 
        outlier_dataset,  # val_dataset
        batch_size=50, # args.batch_size,
        drop_last=True,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    brain_encoder = get_model(args).to(device) #BrainEncoder(args).to(device)

    weight_dir = os.path.join(args.save_root, 'weights')
    last_weight_file = os.path.join(weight_dir, "model_last.pt")
    best_weight_file = os.path.join(weight_dir, "model_best.pt")
    if os.path.exists(best_weight_file):
        brain_encoder.load_state_dict(torch.load(best_weight_file))
        print('weight is loaded from ', best_weight_file)
    else:
        brain_encoder.load_state_dict(torch.load(last_weight_file))
        print('weight is loaded from ', last_weight_file)


    classifier = Classifier(args)

    Zs = []
    Ys = []
    Ls = []
    brain_encoder.eval()
    for batch in test_loader:
        with torch.no_grad():

            if len(batch) == 3:
                X, Y, subject_idxs = batch
            elif len(batch) == 4:
                X, Y, subject_idxs, Labels = batch
            else:
                raise ValueError("Unexpected number of items from dataloader.")

            X, Y = X.to(device), Y.to(device)

            Z = brain_encoder(X, subject_idxs)  # 0.96 GB
            Zs.append(Z)
            Ys.append(Y)
            Ls.append(Labels)

    Zs = torch.cat(Zs, dim=0)
    Ys = torch.cat(Ys, dim=0)
    Ls = torch.cat(Ls, dim=0).detach().cpu().numpy()

    # 仮説1:判定に偏りがある。-> あるサンプルのimageの特徴量がMEGの潜在空間ににているかどうかを判定するだけの基準になっているのではないか？
    Zs = Zs - Zs.mean(dim=0, keepdims=True)
    Zs = Zs / Zs.std(dim=0, keepdims=True)
    Zs = Zs - Zs.mean(dim=1, keepdims=True)
    Zs = Zs / Zs.std(dim=1, keepdims=True)


    similarity_meg_text = calc_similarity(Zs, text_features)

    
    text_features *= torch.Tensor(source_dataset.std_Y).to(device)
    Ys *= torch.Tensor(source_dataset.std_Y).to(device)
    text_features += torch.Tensor(source_dataset.mean_Y).to(device)
    Ys += torch.Tensor(source_dataset.mean_Y).to(device)
    similarity_image_text = calc_similarity(Ys, text_features)
    preds_image_text = np.argmax(similarity_image_text, axis=1)
    preds_meg_text = np.argmax(similarity_meg_text, axis=1)
    pred_dict = {'image_text_label': [], 'image_text_similarity':[],'meg_text_label':[], 'meg_text_similarity':[]}

    for i in range(len(preds_image_text)):
        
        p_it = preds_image_text[i]
        p_mt = preds_meg_text[i]
        print('{}th: image2text {}'.format(i, prompts[p_it]), similarity_image_text[i], 'meg2text, {}'.format(prompts[p_mt]), similarity_meg_text[i])
        pred_dict['image_text_label'].append(prompts[p_it])
        pred_dict['image_text_similarity'].append(np.max(similarity_image_text[i]))    
        pred_dict['meg_text_label'].append(prompts[p_mt])
        pred_dict['meg_text_similarity'].append(np.max(similarity_meg_text[i])) 
    # with open(os.path.join(savedir, 'preds.txt'), 'w') as f:
    #     f.writelines(pred_labels)
    pd.DataFrame(pred_dict).to_csv(os.path.join(savedir, 'preds.csv'))
    print('Compatibility image and meg', np.mean([it==mt for it, mt in zip(pred_dict['image_text_label'],pred_dict['meg_text_label'])]))
    print('chance is {}'.format(1/len(prompts)))
    print('save to ', os.path.join(savedir, 'preds.csv'))
    # calc_similarity(Zs, Ys)
    # import pdb; pdb.set_trace()

    # # MSE
    # squared_error = (Zs.unsqueeze(1) - text_features.unsqueeze(0))**2
    # squared_error = torch.sqrt(squared_error.mean(dim=-1))
    # squared_error = squared_error.cpu().numpy()
    # preds = np.argmax(squared_error, axis=1)
    # pred_labels= []
    # for i, p in enumerate(preds):
    #     print('{}th: {}'.format(i, prompts[p]), squared_error[i])
    #     pred_labels.append(prompts[p]+'\n')
    # with open(os.path.join(savedir, 'preds_mse.txt'), 'w') as f:
    #     f.writelines(pred_labels)


def calc_similarity(x, y):
    batch_size = len(x)
    gt_size = len(y)

    similarity = torch.empty(batch_size, gt_size).to('cuda')
    for i in range(batch_size):
        for j in range(gt_size):
            similarity[i, j] = (x[i] @ y[j]) / max((x[i].norm() * y[j].norm()), 1e-8)
    return similarity.cpu().numpy()




if __name__ == '__main__':
    prompt_root = '/home/yainoue/meg2image/codes/MEG-decoding/data/prompts'
    prompt_sub_dir = 'prompt5'
    prompt_dir = os.path.join(prompt_root, prompt_sub_dir)
    prompt_dict_file = os.path.join(prompt_dir, 'classification1.json')
    with open(prompt_dict_file, 'r') as f:
        prompt_dict = json.load(f)
    text_features, prompts =  get_language_model(prompt_dict, prompt_dir)
    # exit()
    from hydra import initialize, compose
    with initialize(version_base=None, config_path="../configs/"):
        args = compose(config_name='20230429_sbj01_eegnet_regression')
    savedir = os.path.join(args.save_root, 'classification', prompt_sub_dir)
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    
    prompts = [p.strip() for p in prompts]
    evaluate(args, text_features.cpu().numpy(), prompts, savedir, eval_sbj='1')