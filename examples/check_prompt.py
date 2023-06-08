#from importlib.resources import path
import os
import torch
import clip
import numpy as np
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from tqdm import tqdm
import pandas as pd
import time
from PIL import Image
from PIL import ImageFile  # 大きな画像もロード
import pickle
import cv2
import json
ImageFile.LOAD_TRUNCATED_IMAGES = True

def calc_similarity(x, y):
    batch_size = len(x)
    gt_size = len(y)

    similarity = torch.empty(batch_size, gt_size).to('cuda')
    for i in range(batch_size):
        for j in range(gt_size):
            similarity[i, j] = (x[i] @ y[j]) / max((x[i].norm() * y[j].norm()), 1e-8)
    return similarity.cpu().numpy()

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/16", device=device)
model = model.eval()


prompt_root = '/home/yainoue/meg2image/codes/MEG-decoding/data'
prompt_sub_dir = 'prompt2'
prompt_dir = os.path.join(prompt_root, prompt_sub_dir)
prompt_dict_file = os.path.join(prompt_dir, 'classification1.json')
with open(prompt_dict_file, 'r') as f:
    prompt_dict = json.load(f)

prompts = []
prefix = prompt_dict['prefix']
for i, t in prompt_dict.items():
    if i == 'prefix':
        continue
    prompts.append(t+'\n')
text = clip.tokenize([prefix + s.replace('\n','') for s in prompts]).to(device)
with torch.no_grad():
    text_features = model.encode_text(text)


images_dir = "/storage/dataset/ECoG/internal/GODv2-4/images_val/"
print('start getting image file name list')
list_images = os.listdir(images_dir)
print('end getting image file name list')
len(list_images)

image_names_list = []
image_feats_list = []

for cnt, img in enumerate(list_images):
    img_dir = images_dir + img
    image = preprocess(Image.open(img_dir)).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
    sims = calc_similarity(image_features, text_features)

    
    image_feats_list.append(image_features)
    image_names_list.append(img)
    
    label = prompts[np.argmax(probs)]
    print(img_dir)
    # print(f"{cnt}th Label {label} probs:", probs, 'logits', logits_per_image)
    print(f"{cnt}th Label {label} probs:", probs, 'sims', sims)


    