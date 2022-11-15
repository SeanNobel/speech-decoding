import os, torch, random
import numpy as np

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.use_deterministic_algorithms(True)
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
g = torch.Generator()
g.manual_seed(0)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)