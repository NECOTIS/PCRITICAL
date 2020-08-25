import os
import sys
import random
import subprocess
import signal
import fire
import numpy as np
import torch
from torch.utils import data as torch_data
from quantities import ms, second
from tqdm import tqdm
from modules.topologies import SmallWorldTopology
from modules.pcritical import PCritical
from ebdataset.audio import NTidigits
from modules.utils import OneToNLayer


DATASET_PATH = os.environ["NTIDIGITS_DATASET_PATH"]
SOCWATCH_PATH = os.environ["SOCWATCH_DIR"]
n_features = 64
dt = 1 * ms


def rec_array_to_spike_train(sparse_spike_train):
    ts = sparse_spike_train.ts * second
    ts = (ts.rescale(dt.units) / dt).magnitude
    duration = np.ceil(np.max(ts)) + 1
    spike_train = torch.zeros((n_features, duration.astype(int)))
    spike_train[sparse_spike_train.addr, ts.astype(int)] = 1
    return spike_train


def collate_fn(samples):
    """Create a batch out of a list of tuple [(spike_train_tensor, str_label)]
    by zero-padding the spike trains"""
    max_duration = max([s[0].shape[-1] for s in samples])
    batch = torch.zeros(len(samples), n_features, max_duration)
    labels = []
    for i, s in enumerate(samples):
        batch[i, :, : s[0].shape[-1]] = s[0]
        labels.append(int(s[1].replace("z", "0").replace("o", "10")))
    return batch, torch.tensor(labels)


def run_power_ntidigits():
    topology = SmallWorldTopology(
        SmallWorldTopology.Configuration(
            minicolumn_shape=(4, 4, 4),
            macrocolumn_shape=(2, 2, 2),
            minicolumn_spacing=1460,
            p_max=0.11,
            spectral_radius_norm=False,
            intracolumnar_sparseness=635,
            neuron_spacing=40,
            inhibitory_init_weight_range=(0.1, 0.3),
            excitatory_init_weight_range=(0.2, 0.5),
        )
    )

    pcritical_configs = {
        "alpha": 1e-2,
        "stochastic_alpha": True,
        "beta": 1e-5,
        "tau_v": 30 * ms,
        "tau_i": 5 * ms,
        "tau_v_pair": 5 * ms,
        "tau_i_pair": 0 * ms,
        "v_th": 1,
    }

    model = torch.nn.Sequential(
        OneToNLayer(N=2, dim_input=n_features, dim_output=topology.number_of_nodes()),
        PCritical(1, topology, dt=dt, **pcritical_configs),
    )

    train_set = NTidigits(
        DATASET_PATH,
        train=True,
        transforms=rec_array_to_spike_train,
        only_single_digits=True,
    )
    data_loader_parameters = {
        "batch_size": len(train_set),  # Load all in memory before starting socwatch
        "num_workers": 1,
        "pin_memory": True,
        "timeout": 120,
        "collate_fn": collate_fn,
    }
    generator = torch_data.DataLoader(
        train_set, shuffle=False, **data_loader_parameters
    )
    data, _ = next(iter(generator))
    data = data.view(data.shape[0], 1, *data.shape[1:])

    duration = data.shape[-1]

    # First get idle power
    process = subprocess.Popen(
        [
            os.path.join(SOCWATCH_PATH, "socwatch"),
            "-m",
            "-t",
            "60",
            "-f",
            "cpu-cstate",
            "-f",
            "cpu-pstate",
            "-f",
            "pkg-pwr",
            "-o",
            "/opt/socwatch/results/ntidigits_idle",
        ],
        stdout=sys.stdout,
        stdin=sys.stdin,
        stderr=sys.stderr,
        shell=False,
    )
    process.wait()

    # Start SoC watch for dynamic power
    process = subprocess.Popen(
        [
            os.path.join(SOCWATCH_PATH, "socwatch"),
            "-m",
            "-f",
            "cpu-cstate",
            "-f",
            "cpu-pstate",
            "-f",
            "pkg-pwr",
            "-o",
            "/opt/socwatch/results/ntidigits_running",
        ],
        stdout=sys.stdout,
        stdin=sys.stdin,
        stderr=sys.stderr,
        shell=False,
    )
    for spike_train in data:
        for t in range(duration):
            model(spike_train[:, :, t])

    os.kill(process.pid, signal.SIGINT)  # Stop SoC watch
    process.wait()


def main(seed=0x1B,):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)

    run_power_ntidigits()


if __name__ == "__main__":
    fire.Fire(main)
