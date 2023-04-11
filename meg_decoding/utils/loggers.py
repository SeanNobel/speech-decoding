import csv
import pandas as pd
from datetime import datetime
import os
import pickle


def get_timestamp():
    return datetime.today().isoformat(' ')


class Pickleogger():
    def __init__(self, logdir):
        if not os.path.isdir(logdir):
            os.makedirs(logdir)
        self.logfile = os.path.join(logdir, get_timestamp())
        print('logger file is ', self.logfile)
        self.logs = {}

    def create_logger(self, name:str, keys:list):
        self.logs[name] = {k:[] for k in keys}

    def log(self, data, name):
        if not(name in self.logs.keys()):
            self.create_logger(name, list(data.keys()))
        for k, v in data.items():
            self.logs[name][k].append(v)

        with open(self.logfile, 'wb') as f:
            pickle.dump(self.logs, f)
