import os
import numpy as np
from h5py import File
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
import seaborn as sns
import joblib

if __name__ == "__main__":
    sns.set_style("whitegrid")
    plt.rcParams.update({"font.size": 22})

    loihi = joblib.load("mean_weights_output_loihi.joblib")
    torch = joblib.load("mean_weights_output_torch.joblib")

    freqs = [10, 15, 30, 50]
    duration = 1000
    colors = cm.rainbow(np.linspace(0, 1, len(freqs)))
    fig, ax1 = plt.subplots(figsize=(16, 10))
    ax2 = ax1.twinx()

    for i, freq in enumerate(freqs):
        ax2.plot(
            np.arange(duration), torch[i], "--", color=colors[i], label="%i Hz" % freq, dashes=(5, 8),
        )
        ax1.plot(
            np.arange(duration), loihi[i], "-", color=colors[i], label="%i Hz" % freq
        )

    # plt.grid(True)
    # ax.set_title('Autoregulation of the averaged weight using the Paired CRITICAL plasticity rule\n with random poisson input (170 inputs for 512 neurons)', fontsize=22)

    ax1.set_ylabel("Average weight on Loihi", fontsize=22)
    ax2.set_ylabel("Average weight on PyTorch", fontsize=22)
    ax1.set_xlabel("Time [ms]", fontsize=22)

    # ax2.legend(loc='upper right')
    ax1.spines["top"].set_visible(False)
    ax1.spines["bottom"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["left"].set_visible(False)
    ax1.set_xlim((0, duration))

    lines = [Line2D([0], [0], color=c, linewidth=1, linestyle="-") for c in colors]
    lines += [
        Line2D([0], [0], color="black", linewidth=1, linestyle="-"),
        Line2D([0], [0], color="black", linewidth=1, linestyle="--"),
    ]
    labels = list(map(lambda f: f"{f} Hz", freqs))
    labels += [
        "Loihi",
        "PyTorch",
    ]
    leg = ax1.legend(lines, labels, loc="upper center", ncol=3)
    for line in leg.get_lines():
        line.set_linewidth(5.0)

    fig.tight_layout()
    fig.savefig("pcritical-weight-adaptation-for-poisson.eps")
    plt.show()
