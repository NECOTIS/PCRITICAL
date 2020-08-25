import os
from time import time
import random
import logging
import fire
import numpy as np
from modules import reporter
import torch
from quantities import ms
from tqdm import tqdm
from modules.topologies import SmallWorldTopology
from modules.pcritical import PCritical
from modules.utils import display_spike_train, OneToNLayer
import matplotlib.pyplot as plt
from scipy import signal
import scipy.fftpack
import seaborn as sns
import cv2


dt = 1 * ms
_logger = logging.getLogger(__name__)


def to_raster(spike_train):
    neuron_ids, time_of_spikes = np.nonzero(spike_train)
    number_of_neurons = np.max(neuron_ids) + 1
    raster = [0] * number_of_neurons
    for j in range(number_of_neurons):
        raster[j] = time_of_spikes[neuron_ids == j]
    return raster


def run_experiment(
    freq: int, display: bool = False, debug: bool = False,
):
    plt.ioff()
    sns.set()
    topology = SmallWorldTopology(
        SmallWorldTopology.Configuration(
            minicolumn_shape=(4, 4, 4),
            macrocolumn_shape=(1, 1, 1),
            # minicolumn_spacing=1460,
            p_max=0.17,
            intracolumnar_sparseness=635.0,
            neuron_spacing=40.0,
            inhibitory_init_weight_range=(0.1, 0.3),
            excitatory_init_weight_range=(0.2, 0.5),
        )
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pcritical_configs: dict = {
        "alpha": 1e-2,
        "stochastic_alpha": False,
        "beta": 1e-3,
        "tau_v": 30 * ms,
        "tau_i": 1 * ms,
        "tau_v_pair": 5 * ms,
        "tau_i_pair": 0 * ms,
        "v_th": 1,
    }

    n_neurons = topology.number_of_nodes()
    n_input = 8
    model = torch.nn.Sequential(
        OneToNLayer(N=1, dim_input=n_input, dim_output=n_neurons),
        PCritical(1, topology, dt=dt, **pcritical_configs),
    ).to(device)

    freq = freq / 1000
    duration = 5000
    input_spike_train = torch.from_numpy(
        np.random.poisson(lam=freq, size=(1, n_input, duration))
    ).float()

    # Set up figure
    fig = plt.figure(constrained_layout=True, figsize=(16, 10))
    gs = fig.add_gridspec(3, 2)

    reservoir_spikes_ax = fig.add_subplot(gs[1, :])
    reservoir_spikes_ax.set_title(f"P-CRITICAL enabled reservoir spike train")
    reservoir_spikes_ax.set_xlim(0, duration)
    reservoir_spikes_ax.set_ylim(-0.5, n_neurons + 0.5)

    in_spikes_ax = fig.add_subplot(gs[0, :], sharex=reservoir_spikes_ax)
    in_spikes_ax.set_title(
        f"Input spike train of {n_input} neurons poisson-sampled at {freq * 1000} Hz"
    )
    in_spikes_ax.set_ylim(-0.5, n_input + 0.5)
    for label in in_spikes_ax.get_xticklabels():
        label.set_visible(False)

    weight_hist_ax = fig.add_subplot(gs[2, 0])
    weight_hist_ax.set_ylim([0.0, 0.3])
    weight_hist_ax.set_title("Excitatory Weights Probability Density Function")

    relative_weights_ax = fig.add_subplot(gs[2, 1])
    relative_weights_ax.set_ylim([0.0, 300.0])
    relative_weights_ax.set_xlim([-1, 1])
    relative_weights_ax.set_title(
        "Relative Excitatory Weights Adaptation (current - initial) / initial"
    )

    # Set up video recorder
    out = cv2.VideoWriter(
        f"{freq*1000:.0f}hz_spike_analysis.avi",
        cv2.VideoWriter_fourcc(*"MP42"),
        1000.0,
        (800 * 2, 1000),
    )

    # Iterate over time
    excitatory_mask = model[1].W_rec.numpy() > 0
    initial_excitatory_weights = model[1].W_rec[excitatory_mask].numpy()

    blit = True

    if blit:
        fig.canvas.draw()
        backgrounds = [
            fig.canvas.copy_from_bbox(weight_hist_ax.bbox),
            fig.canvas.copy_from_bbox(relative_weights_ax.bbox),
        ]

    for t in tqdm(range(duration)):
        input_spikes = input_spike_train[:, :, t]
        reservoir_spikes = model(input_spikes).numpy()
        input_spikes = input_spikes.numpy()
        adj_matrix = model[1].W_rec.numpy()
        excitatory_weights = adj_matrix[excitatory_mask]
        # pair_spikes = model[1].S_paired.numpy()

        # Input spikes plot
        neurons_that_spikes = np.flatnonzero(input_spikes[0])
        if len(neurons_that_spikes) > 0:
            raster = [
                [] if i not in neurons_that_spikes else [t] for i in range(n_input)
            ]
            input_spikes_ax_data = in_spikes_ax.eventplot(raster)
        else:
            input_spikes_ax_data = []

        # Reservoir spikes plot
        neurons_that_spikes = np.flatnonzero(reservoir_spikes[0])
        if len(neurons_that_spikes) > 0:
            raster = [
                [] if i not in neurons_that_spikes else [t] for i in range(n_neurons)
            ]
            reservoir_spikes_ax_data = reservoir_spikes_ax.eventplot(raster)
        else:
            reservoir_spikes_ax_data = []

        # Weight hist plot
        _, _, weight_hist_ax_data = weight_hist_ax.hist(
            excitatory_weights,
            bins=np.arange(0.0, 1.0, 0.01),
            weights=np.ones_like(excitatory_weights) / len(excitatory_weights),
            color=sns.color_palette()[0],
        )

        # Relative hist plot
        _, _, relative_weights_ax_data = relative_weights_ax.hist(
            np.divide(
                excitatory_weights - initial_excitatory_weights,
                initial_excitatory_weights,
            ),
            bins=np.arange(-1.0, 1.0, 0.0125),
            color=sns.color_palette()[1],
        )

        # Compute frame to video
        if blit:
            # reservoir_spikes_ax.draw_artist(reservoir_spikes_ax.patch)
            for i in reservoir_spikes_ax_data:
                reservoir_spikes_ax.draw_artist(i)
            # in_spikes_ax.draw_artist(in_spikes_ax.patch)
            for i in input_spikes_ax_data:
                in_spikes_ax.draw_artist(i)

            for background in backgrounds:
                fig.canvas.restore_region(background)

            for i in weight_hist_ax_data:
                weight_hist_ax.draw_artist(i)
            for i in relative_weights_ax_data:
                relative_weights_ax.draw_artist(i)
            fig.canvas.update()
        else:
            fig.canvas.draw()

        fig.canvas.flush_events()

        img = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        img = img[:, :, ::-1]  # argb to bgra
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)  # Removing alpha channel

        if display:
            cv2.imshow("Analysis of the weights", img)
            cv2.waitKey(1)

        out.write(img)

        # Clear data from plot
        for d in weight_hist_ax_data:
            d.remove()
        weight_hist_ax_data.clear()
        for d in relative_weights_ax_data:
            d.remove()
        relative_weights_ax_data.clear()

    for _ in range(15):  # Hold the last frame for 15 fps
        out.write(img)

    out.release()


def main(
    freq: int, display: bool = False, seed: int = 0x1B, debug: bool = False,
):
    # Set-up reproducibility
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Init logging
    reporter.init(
        "pcritical-poisson-swipe", backend=reporter.Backend.Logging, debug=debug,
    )
    reporter.log_parameter("seed", seed)

    # Start
    run_experiment(freq=freq, display=display, debug=debug)


if __name__ == "__main__":
    fire.Fire(main)
