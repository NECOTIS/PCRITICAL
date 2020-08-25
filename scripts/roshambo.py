import os
import logging
import random

from quantities import us, ms

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils import data as torch_data
from tqdm import tqdm
import neptune

from ebdataset.vision import INIRoshambo
from ebdataset.vision.transforms import Compose, MaxTime, ScaleDown, ToDense, Flatten
from modules.topologies import SmallWorldTopology
from modules.pcritical import PCritical
from modules.readout import LinearReadout, LinearWithBN

_logger = logging.getLogger(__name__)


def display_spike_train(subplot, torch_train):
    neuron_ids, time_of_spikes = np.nonzero(torch_train.cpu().numpy())
    number_of_neurons = np.max(neuron_ids) + 1
    raster = [0] * number_of_neurons
    for j in range(number_of_neurons):
        raster[j] = time_of_spikes[neuron_ids == j]
    subplot.eventplot(raster)


def run_roshambo():
    seed = 0x1B
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    neptune.set_property("seed", seed)
    neptune.append_tag("ROSHAMBO")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _logger.info("Using device type %s", str(device))

    reduction_factor = 5  # Reduce dimension axis by this factor
    neptune.set_property("reduction_factor", reduction_factor)

    width = 240 // reduction_factor
    height = 180 // reduction_factor
    n_features = width * height * 2
    batch_size = 5
    neptune.set_property("batch_size", batch_size)

    dt = 1 * ms
    neptune.set_property("dt", dt)

    bin_size = 50 * ms
    neptune.set_property("bin_size", bin_size)

    bin_steps = rescale(bin_size, dt, int)
    duration_per_sample = 500 * ms
    neptune.set_property("duration_per_sample", duration_per_sample)

    number_of_steps = rescale(duration_per_sample, dt, int)

    topology = SmallWorldTopology(
        SmallWorldTopology.Configuration(
            minicolumn_shape=(7, 7, 7),
            macrocolumn_shape=(3, 3, 3),
            minicolumn_spacing=300,
            p_max=0.025,
            sparse_init=True,
        )
    )
    n_neurons = topology.number_of_nodes()
    nb_of_bins = 1 + number_of_steps // bin_steps
    linear_readout = LinearWithBN(n_neurons * nb_of_bins, 3).to(device)
    loss_fn = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(linear_readout.parameters(), lr=0.001)
    neptune.set_property("adam.lr", 0.001)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
    neptune.set_property("steplr.gamma", 0.1)
    neptune.set_property("steplr.step_size", 2)

    p_critical_configs = {
        "alpha": 0.0025,
        "beta": 0.00025,
        "tau_v": 50 * ms,
        "tau_i": 5 * ms,
        "v_th": 1.0,
    }

    for k, v in p_critical_configs.items():
        neptune.set_property(k, v)

    model = PCritical(
        n_features, batch_size, topology, dt=dt, **p_critical_configs,
    ).to(device)

    all_transforms = Compose(
        [
            ScaleDown(240, 180, factor=reduction_factor),
            ToDense(width, height, duration_per_sample, dt=dt),
            Flatten(),
        ]
    )

    label_dict = {
        "scissors": 0,
        "paper": 1,
        "rock": 2,
    }

    data = INIRoshambo(
        os.getenv("ROSHAMBO_DATASET_LOCATION_500ms_subsamples"),
        transforms=all_transforms,
    )
    train_data, val_data = split_per_user(data, train_ratio=0.85)
    _logger.info(
        "Keeping %i samples for training and %i for validation",
        len(train_data),
        len(val_data),
    )

    def labels_to_tensor(labels):
        return torch.tensor([label_dict[l] for l in labels])

    def run_batch(X, y):
        current_batch_size = len(y)
        model.batch_size = current_batch_size
        bins = torch.zeros(current_batch_size, n_neurons, nb_of_bins, device=device)
        for t in range(number_of_steps):
            out_spikes = model.forward(X[:, :, t])
            bins[:, :, t // bin_steps] += out_spikes
        return bins

    for iter_nb in range(10):
        train_generator = torch_data.DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            timeout=120,
        )
        for i, (X, labels) in enumerate(tqdm(train_generator)):
            if i >= 20:
                break

            neptune.log_metric("iteration", i)
            X, y = X.to(device), labels_to_tensor(labels).to(device)

            # fig, axs = plt.subplots()
            # display_spike_train(axs, X[0])
            # plt.show()
            # print(X.shape)
            # exit(0)

            bins = run_batch(X, y)

            # fig, axs = plt.subplots()
            # activity = bins[0].sum(dim=0)
            # axs.plot(np.arange(nb_of_bins), activity.cpu().numpy())
            # plt.show()

            optimizer.zero_grad()
            out = linear_readout(bins.view(len(y), -1))
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()
            loss_val = loss.cpu().detach().item()
            _logger.info("Loss: %.3f", loss_val)
            neptune.log_metric("loss", loss_val)

        total_accurate = 0
        total_elems = 0
        val_generator = torch_data.DataLoader(
            val_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            timeout=120,
        )
        for i, (X, labels) in enumerate(tqdm(val_generator)):
            if i >= 10:
                break
            X, y = X.to(device), labels_to_tensor(labels).to(device)
            bins = run_batch(X, y)
            out = linear_readout(bins.view(len(y), -1))
            preds = torch.argmax(out, dim=1)
            total_accurate += torch.sum(preds == y).cpu().float().item()
            total_elems += len(y)
            _logger.info("Current accuracy: %.4f", total_accurate / total_elems)
            neptune.log_metric("current_accuracy", total_accurate / total_elems)

        scheduler.step()

        _logger.info(
            "Final accuracy at iter %i: %.4f", iter_nb, total_accurate / total_elems
        )
        neptune.log_metric("final_accuracy", total_accurate / total_elems)


def rescale(quantity, dt, type):
    return type((quantity.rescale(dt.units) / dt).magnitude)


def split_per_user(data, train_ratio):
    """Split into train - validation while keeping users in either"""

    def get_roshambo_user(sample_name):
        parts = sample_name[: sample_name.find(".aedat")].split("_")
        if parts[1] in set(
            ["glove", "back", "front"]
        ):  # Order is not always consistent
            return parts[2]
        return parts[1]

    nb_train = int(len(data) * train_ratio)
    o_samples = list(map(get_roshambo_user, data.samples))
    _, indices, counts = np.unique(o_samples, return_inverse=True, return_counts=True)
    training_users = np.nonzero(np.cumsum(counts) <= nb_train)[0]
    training_mask = np.isin(indices, training_users)
    train_data = torch_data.Subset(data, indices=np.arange(len(data))[training_mask])
    val_data = torch_data.Subset(data, indices=np.arange(len(data))[~training_mask])

    return train_data, val_data


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        filename="roshambo.log",
    )
    neptune.init("abc/abc", backend=neptune.OfflineBackend())
    # neptune.init("tihbe/pcritical")
    with neptune.create_experiment(
        "roshambo", upload_stdout=False, upload_stderr=False
    ):
        run_roshambo()
