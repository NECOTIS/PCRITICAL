import os
import torch
import torch.nn as nn
import numpy as np
import networkx as nx
from quantities import ms
from ebdataset.vision import NMnist
from torch.utils import data
from tqdm import tqdm, trange
import joblib
from modules.topologies import SmallWorldTopology
from tqdm import tqdm
from modules.pcritical import PCritical
from modules.utils import OneToNLayer
import fire
from ebdataset.vision.transforms import Compose, ToDense, Flatten
from torch.multiprocessing import Pool, set_start_method

try:
    set_start_method("spawn")
except RuntimeError:
    pass

NMNIST_PATH = os.environ["NMNIST_DATASET_PATH"]
OUT_DIR = "nmnist_output"


def process_batch(args):
    i, (batch, labels), model, data_type, device = args
    batch, labels = batch.to(device), labels.to(device)
    model[1].batch_size = batch.shape[0]
    spikes = []
    for t in trange(batch.shape[-1]):
        spikes.append(model(batch[:, :, t]))
    spikes = torch.stack(spikes, dim=2)
    joblib.dump(
        (spikes.cpu().numpy(), labels.cpu().numpy()),
        os.path.join(OUT_DIR, "%s_batch_%i" % (data_type, i)),
        compress=3,
    )


def collate_fn(samples):
    max_duration = max([s[0].shape[-1] for s in samples])
    batch = torch.zeros(len(samples), 34 * 34 * 2, max_duration)
    labels = []
    for i, s in enumerate(samples):
        batch[i, :, : s[0].shape[-1]] = s[0]
        labels.append(s[1])
    return batch, torch.tensor(labels)


def main(seed=0x1B, pool_size=6, num_threads_per_cpu=2):
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    topology = SmallWorldTopology(
        SmallWorldTopology.Configuration(
            minicolumn_shape=(6, 6, 5),
            macrocolumn_shape=(4, 4, 3),
            minicolumn_spacing=1460,
            p_max=0.11,
            intracolumnar_sparseness=635,
            neuron_spacing=40,
            inhibitory_init_weight_range=(0.1, 0.3),
            excitatory_init_weight_range=(0.2, 0.5),
        )
    )

    dt = 1 * ms
    transforms = Compose([ToDense(dt=dt), Flatten(),])

    N_features = 34 * 34 * 2

    params = {
        "batch_size": 4,
        "collate_fn": collate_fn,
        "shuffle": True,
        "num_workers": 2,
    }

    pcritical_configs = {
        "alpha": 1e-2,
        "stochastic_alpha": False,
        "beta": 1e-3,
        "tau_v": 30 * ms,
        "tau_i": 5 * ms,
        "tau_v_pair": 5 * ms,
        "tau_i_pair": 0 * ms,
        "v_th": 1,
    }

    reservoir = PCritical(params["batch_size"], topology, **pcritical_configs).float()
    model = torch.nn.Sequential(
        OneToNLayer(1, N_features, topology.number_of_nodes()), reservoir
    )
    model = model.to(device)
    model[0].W_t = model[0].W_t.to_dense()

    torch.set_num_threads(num_threads_per_cpu)
    with Pool(pool_size) as p:
        training_set = NMnist(NMNIST_PATH, is_train=True, transforms=transforms)
        training_generator = data.DataLoader(training_set, **params)

        train_samples = map(
            lambda args: (*args, model, "TRAIN", device), enumerate(training_generator)
        )

        pbar = tqdm(total=len(training_generator))
        for _ in p.imap(process_batch, train_samples):
            pbar.update(1)
        pbar.close()

        test_set = NMnist(NMNIST_PATH, is_train=False, transforms=transforms)
        test_generator = data.DataLoader(test_set, **params)

        test_samples = map(
            lambda i, sample: (i, *sample, model, "TEST", device),
            enumerate(test_generator),
        )
        pbar = tqdm(total=len(test_generator))
        for _ in p.imap(process_batch, test_samples):
            pbar.update(1)
        pbar.close()


if __name__ == "__main__":
    fire.Fire(main)
