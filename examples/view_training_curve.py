import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import os

loss_key = ['train_loss', 'test_loss']
acc_key = ['trainTop1acc', 'trainTop10acc', 'testTop1acc', 'testTop10acc']
lr_key = ['lrate']
temp_key = ['temp']

def get_data(pkl_file:str):
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    return data

def visualize(logs, savefile):
    fig, axes = plt.subplots(ncols=4, figsize=(32,8))
    epochs = np.arange(len(logs))
    ax=axes[0]
    for k in loss_key:
        ax.plot(epochs, logs[k].to_list(), label=k)
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')   
    ax.legend()
    ax=axes[1]
    for k in acc_key:
        ax.plot(epochs, logs[k].to_list(), label=k)
    ax.set_xlabel('epoch')
    ax.set_ylabel('accuracy')
    ax.legend()
    ax=axes[2]
    for k in lr_key:
        ax.plot(epochs, logs[k].to_list(), label=k)
    ax.set_xlabel('epoch')
    ax.set_ylabel('lr') 
    ax.legend()
    ax=axes[3]
    for k in temp_key:
        ax.plot(epochs, logs[k].to_list(), label=k)
    ax.set_xlabel('epoch')
    ax.set_ylabel('temp') 
    ax.legend()
    plt.savefig(savefile)
    print('save as ', savefile)

def parse_data(filepath):
    data = get_data(filepath)
    log_names = list(data.keys())
    savedir = os.path.dirname(filepath)
    for name in log_names:
        logs = data[name]
        columns = logs.keys()
        logs = pd.DataFrame(logs)
        savefile = os.path.join(savedir, f'{name}_learning_curve.png')
        visualize(logs, savefile)


if __name__ == '__main__':
    # filepath = '/home/yainoue/meg2image/results/20230413_sbj01/runs/2023-04-13 21:52:17.333087'#'home/yainoue/meg2image/results/test/2023-04-11 18:06:44.273986'
    filepath = '/home/yainoue/meg2image/results/20230412_mini/runs/2023-04-13 15:17:53.310665'
    parse_data(filepath)
