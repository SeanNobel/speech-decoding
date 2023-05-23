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
ImageFile.LOAD_TRUNCATED_IMAGES = True

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/16", device=device)
model = model.eval()

# images_dir = "./imagenet_fall2011_oneImagePerCat_21-2k_20230309/"
images_dir = "/storage/dataset/ECoG/internal/GODv2-4/images_trn/"
print('start getting image file name list')
list_images = os.listdir(images_dir)
print('end getting image file name list')
len(list_images)

image_names_list = []
image_feats_list = []

for img in tqdm(list_images):
    img_dir = images_dir + img
    image = preprocess(Image.open(img_dir)).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)

    
    image_features = image_features.to('cpu').detach().numpy().copy()
    
    image_feats_list.append(image_features)
    image_names_list.append(img)

image_names_np = np.array(image_names_list)
image_feats_np = np.array(image_feats_list)

image_feats_np = np.squeeze(image_feats_np)

name_feat_dict = {}
# for img_name  in image_names_list:
#     for feat_val in image_feats_list:
for img_name, feat_val in zip(image_names_list, image_feats_list):
    name_feat_dict[img_name] = np.squeeze(feat_val)

a_file = open("/home/yainoue/meg2image/codes/MEG-decoding/data/GOD/vit-b16/train_features.pkl", "wb")
pickle.dump(name_feat_dict, a_file)
a_file.close()

# device = "cuda" if torch.cuda.is_available() else "cpu"
# model, preprocess = clip.load("ViT-B/16", device=device)
# model = model.eval()

# images_dir = "./imagenet_fall2011_oneImagePerCat_21-2k_20230309/"
images_dir = "/storage/dataset/ECoG/internal/GODv2-4/images_val/"
print('start getting image file name list')
list_images = os.listdir(images_dir)
print('end getting image file name list')
len(list_images)

image_names_list = []
image_feats_list = []

for img in tqdm(list_images):
    img_dir = images_dir + img
    image = preprocess(Image.open(img_dir)).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)

    
    image_features = image_features.to('cpu').detach().numpy().copy()
    
    image_feats_list.append(image_features)
    image_names_list.append(img)

image_names_np = np.array(image_names_list)
image_feats_np = np.array(image_feats_list)

image_feats_np = np.squeeze(image_feats_np)

name_feat_dict = {}
# for img_name  in image_names_list:
#     for feat_val in image_feats_list:
for img_name, feat_val in zip(image_names_list, image_feats_list):
    name_feat_dict[img_name] = np.squeeze(feat_val)

a_file = open("/home/yainoue/meg2image/codes/MEG-decoding/data/GOD/vit-b16/val_features.pkl", "wb")
pickle.dump(name_feat_dict, a_file)
a_file.close()