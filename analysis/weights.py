import numpy as np
from h5py import File
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from tqdm import tqdm
import fire


def main(
    output_file: str = "weights_over_time",
    mean: bool = False,
    adjacency: bool = False,
    hist: bool = False,
    relative: bool = False,
    display: bool = False,
    style: str = None,
    timestepsize: int = 10,
):
    assert np.any(
        [mean, adjacency, hist, relative]
    ), "Please select at least one type of plot"
    if style is None:
        sns.set()
    else:
        sns.set_style(style)

    ax_cnt = np.sum([mean, adjacency, hist, relative])
    fig, axs = plt.subplots(1, ax_cnt, figsize=(8 * ax_cnt, 10))
    axs = np.asarray(axs).flatten()
    out = cv2.VideoWriter(
        f"{output_file}.avi", cv2.VideoWriter_fourcc(*"MP42"), 5.0, (800 * ax_cnt, 1000)
    )
    plt.ioff()  # Make sure interactive mode is off

    with File("pcritical-tidigits-weight-recording.h5", "r") as f:
        reservoir_weights = f["reservoir_weights"]
        duration, n, n = reservoir_weights.shape

        ax_cntr = 0

        if adjacency:
            adj_ax = axs[ax_cntr]
            ax_cntr += 1
            adj_ax.grid(False)
            adj_ax.set_yticklabels([])
            adj_ax.set_xticklabels([])
        if mean:
            mean_ax = axs[ax_cntr]
            ax_cntr += 1
            means = np.array([])
            stds = np.array([])
        if hist:
            hist_ax = axs[ax_cntr]
            ax_cntr += 1
        if relative:
            relative_ax = axs[ax_cntr]
            ax_cntr += 1

        excitatory_mask = None
        first_iter = True
        plot_xs = np.arange(0, duration + timestepsize, timestepsize)

        for t in tqdm(range(duration)):
            if t % timestepsize != 0:
                continue

            plt.cla()

            weights = reservoir_weights[t][()]

            if first_iter:
                excitatory_mask = weights > 0
                if adjacency:
                    imshow = adj_ax.imshow(weights, cmap="seismic", vmin=-0.4, vmax=0.4)
                    fig.colorbar(
                        imshow, ax=axs[0], shrink=0.8, orientation="horizontal"
                    )
                if relative:
                    initial_excitatory_weights = weights[weights > 0]

            if adjacency:
                imshow.set_data(weights)
                adj_ax.set_title(f"Adjacency matrix of the reservoir at t={t} ms")

            excitatory_weights = weights[excitatory_mask]

            if mean:
                means = np.append(means, excitatory_weights.mean())
                stds = np.append(stds, excitatory_weights.std())
                if first_iter:
                    (mean_ax_data,) = mean_ax.plot(
                        plot_xs[: len(means)], means, color="black"
                    )
                    mean_ax.fill_between(
                        plot_xs[: len(means)],
                        means - stds,
                        means + stds,
                        color="gray",
                        alpha=0.3,
                    )
                else:
                    mean_ax_data.set_data([plot_xs[: len(means)], means])
                    mean_ax.collections.clear()
                    mean_ax.fill_between(
                        plot_xs[: len(means)],
                        means - stds,
                        means + stds,
                        color="gray",
                        alpha=0.3,
                    )

                mean_ax.set_xlim([0, duration])
                mean_ax.set_ylim([0.0, 0.4])
                # axs[0].set_xticks(nb_of_frames)

                mean_ax.set_title(f"Average of the excitatory weights over time")
                # axs[1].set_title(f"Average of the weights over time")
                mean_ax.set_xlabel("Time [ms]")

            if hist:
                hist_ax.clear()
                hist_ax.hist(
                    excitatory_weights,
                    bins=np.arange(0.0, 1.0, 0.01),
                    weights=np.ones_like(excitatory_weights) / len(excitatory_weights),
                )
                hist_ax.set_title(f"Excitatory Weights Probability Density Function")
                hist_ax.set_ylim([0.0, 0.2])

            if relative:
                relative_ax.clear()
                relative_ax.hist(
                    np.divide(
                        excitatory_weights - initial_excitatory_weights,
                        initial_excitatory_weights,
                    ),
                    bins=np.arange(-1.0, 1.0, 0.0125),
                )
                relative_ax.set_ylim([0.0, 700.0])
                relative_ax.set_title(
                    f"Relative Excitatory Weights Adaptation (current - initial) / initial"
                )

            fig.tight_layout()
            fig.canvas.draw()

            img = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            img = img[:, :, ::-1]  # argb to bgra
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)  # Removing alpha channel
            if display:
                cv2.imshow("Analysis of the weights", img)
                cv2.waitKey(1)

            out.write(img)
            first_iter = False

        fig.savefig(f"{output_file}.eps", bbox_inches="tight")

        for _ in range(15):  # Hold the last frame for 15 fps
            out.write(img)

        plt.close(fig)
        out.release()


if __name__ == "__main__":
    fire.Fire(main)
