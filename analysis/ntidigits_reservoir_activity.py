import logging
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import seaborn as sns
from h5py import File
from scipy.ndimage.filters import gaussian_filter1d
from scipy.interpolate import InterpolatedUnivariateSpline
from .branching_factor import (
    branching_factor_from_topology,
    branching_factor_from_spike_count,
)
from modules.utils import display_spike_train


if __name__ == "__main__":
    # Plot various data inferred from recordings
    sns.set_style("whitegrid")
    plt.rcParams.update({"font.size": 30})
    reservoir_topology = None
    input_matrix = None
    input_spike_train = None
    reservoir_spike_train = None

    # The files can be generated with python -m scripts.ntidigits --debug
    with File("pcritical-tidigits-spike-recording.h5", "r") as f:
        reservoir_topology = f["reservoir_topology"][()]  # 512 x 512
        input_matrix = f["input_topology"][()]  # 64 x 512
        input_spike_train = f["input_spikes"][()]  # 50k x 64
        reservoir_spike_train = f["reservoir_spikes"][()]  # 50k x 512

    input_spike_train_mapped = np.clip(
        input_spike_train @ input_matrix, 0, 1
    )  # Mapped to reservoir neurons

    fig, ax = plt.subplots(3, 1, figsize=(32, 20))
    for i in range(3):
        ax[i].set_xlim([0, reservoir_spike_train.shape[0]])
    #display_spike_train(input_spike_train_mapped[:, :].T, ax[0])
    ax[0].plot(input_spike_train_mapped.sum(axis=-1))
    ax[0].set_title("(A) Input spike activity")
    ax[0].set_ylabel("Sum of spike counts")

    # Roll to remove delay of 1 from simulation
    input_less_reservoir_spike_train = reservoir_spike_train - np.roll(
        input_spike_train_mapped, 1, axis=0
    )
    excitatory_synapses = np.argwhere(reservoir_topology > 1e-7)
    excitatory_neurons = np.unique(excitatory_synapses[:, 0])

    ax[1].plot(input_less_reservoir_spike_train.sum(axis=-1))
    #display_spike_train(input_less_reservoir_spike_train[:, :].T, ax[1])
    ax[1].set_title("(B) Self-induced reservoir spike activity")
    ax[1].set_ylabel("Sum of spike counts")

    topo_branching_factor = branching_factor_from_topology(
        reservoir_topology.astype(bool),
        input_less_reservoir_spike_train.T.astype(np.float),
        excitatory_neurons,
    )
    topo_branching_factor_mean = np.nanmean(topo_branching_factor, axis=1)
    topo_branching_factor_sum = np.nansum(topo_branching_factor, axis=1)

    count_branching_factor = branching_factor_from_spike_count(
        input_less_reservoir_spike_train.T.astype(np.float), excitatory_neurons
    )
    # Gaussian filters
    sigma = 0.7
    topo_branching_factor_filtered = gaussian_filter1d(
        topo_branching_factor_mean, sigma=sigma
    )
    count_branching_factor_filtered = gaussian_filter1d(
        count_branching_factor, sigma=sigma
    )
    ax[2].plot(
        np.arange(count_branching_factor.shape[0]),
        count_branching_factor_filtered,
        label="Spike-count branching factor",
    )
    ax[2].plot(
        np.arange(topo_branching_factor.shape[0]),
        topo_branching_factor_filtered,
        label="Topology-aware branching factor",
    )
    ax[2].set_ylabel("Branching factor")
    ax[2].set_ylim([-0.1, 5.0])
    leg = ax[2].legend()

    for line in leg.get_lines():
        line.set_linewidth(5.0)

    ax[2].set_title(f"(C) Branching factor gaussian-filtered with $\\sigma={sigma:.2f}$")
    ax[2].set_xlabel("Time [ms]")

    for i in range(3):
        ax[i].yaxis.set_label_coords(-0.035, 0.5)

    plt.tight_layout()
    fig.savefig("spike_activity_over_time_with_branching_factor.eps", bbox_inches="tight")

    # Poincaré plot for binned t >= 2500
    fig, ax = plt.subplots(figsize=(16, 10))
    bin_size = 5
    S = np.count_nonzero(
        input_less_reservoir_spike_train[2500:, excitatory_neurons], axis=1
    )
    binned_S = S.reshape(len(S) // bin_size, bin_size).sum(axis=1)
    binned_S = binned_S / len(excitatory_neurons) / bin_size
    ax.scatter(
        binned_S[:-1], binned_S[1:], label=f"{bin_size}ms bins", color="orange", s=20
    )
    x = np.linspace(binned_S.min(), binned_S.max(), 1000)
    ax.plot(x, x, label="Model with $\sigma$ = 1", color="black")
    ax.set_xlabel("Normalized spike count at t")
    ax.set_ylabel("Normalized spike count at t+1")
    ax.yaxis.set_label_coords(-0.1, 0.5)
    ax.xaxis.set_label_coords(0.5, -0.1)


    ax.legend()
    plt.tight_layout()
    fig.savefig("branching_factor_poincarre.eps", bbox_inches="tight")

    # plt.show()
