import os
import numpy as np
from h5py import File
import matplotlib.pyplot as plt
import seaborn as sns

import json
from collections import defaultdict

if __name__ == "__main__":
    dir = "pcritical_ntidigits_output"
    files = os.listdir(dir)

    results = [json.load(open(os.path.join(dir, i), "r")) for i in files]

    sns.set_style("whitegrid")
    plt.rcParams.update({"font.size": 22})
    fig, ax = plt.subplots(figsize=(16, 10))
    accuracies = defaultdict(list)
    for result in results:
        plasticity, spectral_norm = result["plasticity"], result["spectral_radius_norm"]
        if plasticity == str(True):
            exp_name = "P-CRITICAL"
        elif spectral_norm == str(True):
            exp_name = "$\\rho$ normalization"
        else:
            exp_name = "Reservoir"

        accuracies[exp_name].append(result["test_accuracies"])

    for exp_name, accuracy in accuracies.items():
        accuracy = np.array(accuracy)

        mean = np.mean(accuracy, axis=0)
        std = np.std(accuracy, axis=0)
        ax.errorbar(
            np.arange(1, 1 + accuracy.shape[1]),
            mean,
            yerr=std,
            label=exp_name,
            capsize=5,
            elinewidth=1,
            fmt="-o",
        )

    ax.set_xlabel("Epoch")
    ax.set_xticks(np.arange(1, 21))
    ax.set_ylabel("Accuracy")

    # Sort the legend
    handles, labels = ax.get_legend_handles_labels()
    order = {
        "P-CRITICAL": 0,
        "$\\rho$ normalization": 1,
        "Reservoir": 2,
    }
    handles, labels = list(
        zip(*sorted(zip(handles, labels), key=lambda i: order[i[1]]))
    )
    ax.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1),
        ncol=3,
        fancybox=True,
    )
    plt.tight_layout()

    filename = "ntidigits-results.png"
    fig.savefig(filename)
    plt.show()
