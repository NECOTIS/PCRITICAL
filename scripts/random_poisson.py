#!/usr/bin/env python
# coding: utf-8
import os
import random
import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from modules.pcritical import PCritical
from modules.utils import OneToNLayer
from modules.topologies import SmallWorldTopology


if __name__ == "__main__":
    seed = 0x1B
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    device = torch.device("cpu")

    topology = SmallWorldTopology(
        SmallWorldTopology.Configuration(
            minicolumn_shape=[4, 4, 4], macrocolumn_shape=[2, 2, 2], p_max=0.02,
        )
    )
    for u, v in topology.edges:
        topology.edges[u, v]["weight"] = np.clip(
            np.random.normal(loc=0.08, scale=0.2), -0.2, 0.6
        )

    N = topology.number_of_nodes()
    N_inputs = N // 3

    fig = plt.figure(figsize=(18, 10))
    ax = fig.add_subplot(1, 1, 1)
    duration = 1000
    freqs = [10, 15, 30, 50]  # in Hz
    mean_weights_output = []

    for freq in tqdm(freqs):

        reservoir = PCritical(1, topology, alpha=0.05).to(device)
        model = torch.nn.Sequential(OneToNLayer(1, N_inputs, N), reservoir)

        in_spikes = np.random.rand(N_inputs, duration) < freq / 1000.0  # mhz
        neuron_idx, spike_times = np.nonzero(in_spikes)

        inp = torch.zeros((1, N_inputs, duration), device=device)

        inp[0, torch.LongTensor(neuron_idx), torch.LongTensor(spike_times)] = 1

        recorded_spikes = []
        W_means = []
        sng = torch.sign(reservoir.W_rec)
        for t in range(duration):
            i, j = torch.nonzero(reservoir.W_rec > 0, as_tuple=True)
            W_means.append(torch.sum(reservoir.W_rec[i, j]) / i.shape[0])
            cur_sign = torch.sign(reservoir.W_rec)
            assert torch.allclose(sng[sng < 0], cur_sign[cur_sign < 0])
            S = model(inp[:, :, t])
            recorded_spikes.append(S)
        recorded_spikes = torch.stack(recorded_spikes, dim=2)

        recorded_weights = torch.stack(W_means, dim=0)

        weights_over_time = recorded_weights.cpu().detach().numpy()
        mean_weights_output.append(weights_over_time)
        plt.plot(np.arange(duration), weights_over_time, label="%i Hz" % freq)

    joblib.dump(mean_weights_output, "mean_weights_output_torch.joblib")
    plt.grid(True)
    ax.set_title(
        "Autoregulation of the averaged weight using the Paired CRITICAL plasticity rule\n with random poisson input (170 inputs for 512 neurons)",
        fontsize=22,
    )

    ax.set_ylabel("Average weight", fontsize=16)
    ax.set_xlabel("Time [ms]", fontsize=16)
    ax.legend(loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.set_xlim((0, duration))
    fig.savefig("weights.svg", bbox_inches="tight")

    plt.show()
