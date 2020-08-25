import numpy as np
from tqdm import trange


def branching_factor_from_topology(
    adjacency_matrix: np.array, spike_train: np.array, mask: np.array = None
):
    """
    Calculate the local branching factor by assuming that a spike in the
    next time frame (t + 1) is part of the sequence.
    :param adjacency_matrix: (np.array) Connectivity matrix of network topology (n x n)
    :param spike_train: (np.array) bool array of spikes (n x duration)
    :param mask: (np.array) Average only for neurons in this mask (n)
    :return: (np.array) local branching factor (duration, n)
    """
    duration = spike_train.shape[1]
    branching_factors = (
        np.zeros((duration, adjacency_matrix.shape[0])) * np.nan
    )  # No presynaptic spikes is NaN
    for t in trange(1, duration):
        (neurons_that_spiked,) = np.nonzero(spike_train[:, t - 1])

        for neuron_id in neurons_that_spiked:
            # Count the number of spikes that happened at t+1 to the connected neurons
            branching_factors[t - 1, neuron_id] = np.count_nonzero(
                spike_train[adjacency_matrix[neuron_id], t]
            )

    if mask is not None:
        return branching_factors[:, mask]
    return branching_factors


def branching_factor_from_spike_count(spike_train: np.array, mask: np.array = None):
    """
    Calculate the global branching factor by taking sum(spikes) at t+1 divided by sum(spikes) at t
    :param spike_train: (np.array) bool array of spikes (n x duration)
    :param mask: (np.array) Average only for neurons in this mask (n)
    :return: (np.array) average branching factor (duration)
    """
    st = spike_train if mask is None else spike_train[mask]
    spike_count = np.count_nonzero(st, axis=0)
    spike_count_at_t = spike_count[:-1]
    spike_count_at_t_plus_1 = spike_count[1:]
    branching_factor = np.zeros(spike_count_at_t.shape)
    np.divide(
        spike_count_at_t_plus_1,
        spike_count_at_t,
        out=branching_factor,
        where=spike_count_at_t > 0,
    )
    return branching_factor
