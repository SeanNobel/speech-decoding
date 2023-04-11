import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd

def get_data(pkl_file:str):
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    return data

def parse_data(data):
    log_names = list(data.keys())
    for name in log_names:
        logs = data[name]
        columns = logs.keys()
        logs = pd.DataFrame(logs)
        import pdb; pdb.set_trace()

if __name__ == '__main__':
    filepath = '/home/yainoue/meg2image/results/test/2023-04-11 18:06:44.273986'
    data = get_data(filepath)
    parse_data(data)
