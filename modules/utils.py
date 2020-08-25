import logging
from select import select
import sys
from typing import List, Tuple
import torch
import torch.nn as nn
from h5py import File
import networkx as netx
import scipy.sparse
import numpy as np
import matplotlib.pyplot as plt

_logger = logging.getLogger(__name__)


class OneToNLayer(nn.Module):
    """
    Connect input neurons to N output neurons with fixed weights
    Internal operators are sparse for efficient computing
    """

    def __init__(self, N, dim_input, dim_output, weight=100.0):
        super().__init__()

        pre = np.arange(dim_input * N) % dim_input
        post = (
            np.random.permutation(max(dim_input, dim_output) * N)[: dim_input * N]
            % dim_output
        )
        i = torch.LongTensor(np.vstack((pre, post)))
        v = torch.ones(dim_input * N) * weight

        # Save the transpose version of W for sparse-dense multiplication
        self.W_t = torch.sparse.FloatTensor(
            i, v, torch.Size((dim_input, dim_output))
        ).t()

    def forward(self, x):
        return self.W_t.mm(x.t()).t()

    def _apply(self, fn):
        super()._apply(fn)
        self.W_t = fn(self.W_t)
        return self


def unbatchifier(x, callback, **nargs):
    """
    Transform a mini-batch of data x of shape (n, ..) to n single samples of shape (1, ...)
    to be processed by the callback fct with nargs as optional named arguments
    :param x: mini-batch of data
    :param callback: processing function taking single sample tensors
    :param nargs: optional arguments for callback
    :return: mini-batch of processed items callback(x)
    """
    return torch.stack([callback(i.unsqueeze(0), **nargs)[0] for i in x], dim=0)


def display_spike_train(spike_train, subplot=plt):
    """Display a spike train of shape (nb_neurons, duration) to subplot
    """
    neuron_ids, time_of_spikes = np.nonzero(spike_train)
    number_of_neurons = np.max(neuron_ids) + 1
    raster = [0] * number_of_neurons
    for j in range(number_of_neurons):
        raster[j] = time_of_spikes[neuron_ids == j]
    subplot.eventplot(raster)


def graph_to_tensor(graph: netx.DiGraph) -> torch.Tensor:
    n = graph.number_of_nodes()
    adj_matrix = netx.adjacency_matrix(graph, weight="weight")
    i, j, w = scipy.sparse.find(adj_matrix)
    return torch.sparse.FloatTensor(
        torch.LongTensor(np.vstack((i, j))), torch.FloatTensor(w), torch.Size((n, n))
    )


def tensor_to_graph(tensor: torch.Tensor) -> netx.DiGraph:
    tensor = tensor.coalesce()
    n, n = tensor.size()
    graph = netx.DiGraph()
    graph.add_nodes_from(range(n))
    indices = tensor.indices().numpy().T
    values = tensor.values().numpy()
    graph.add_weighted_edges_from(np.c_[indices, values])
    return graph


def input_with_timeout(prompt, timeout):
    """Same as input(prompt), with a timeout (seconds) that returns None"""
    print(prompt)
    out = select([sys.stdin], [], [], timeout)[0]
    if sys.stdin in out:
        return sys.stdin.readline().strip()
    else:
        return None


class SpikeRecorder:
    """
    Record spikes from lsm-like models for up to nb_of_timesteps
    """

    def __init__(
        self,
        file_output_name,
        input_topology: torch.Tensor,
        reservoir_topology: netx.DiGraph,
        nb_of_timesteps: int,
        step: int = 1,
    ):
        self.spike_recorder = File(file_output_name, "w-")
        self.spike_recorder["input_topology"] = input_topology.cpu().to_dense().numpy()
        self.spike_recorder["reservoir_topology"] = (
            graph_to_tensor(reservoir_topology).cpu().to_dense().numpy()
        )

        self.reservoir_spike_recorder = self.spike_recorder.create_dataset(
            "reservoir_spikes",
            (nb_of_timesteps, reservoir_topology.number_of_nodes()),
            dtype="b",
        )
        self.input_spike_recorder = self.spike_recorder.create_dataset(
            "input_spikes", (nb_of_timesteps, input_topology.shape[0]), dtype="b"
        )
        self.timestep_counter = 0
        self.nb_of_timesteps = nb_of_timesteps
        self.step = step

    def __call__(self, input_spikes, reservoir_spikes):
        if self.timestep_counter >= self.nb_of_timesteps:
            return False
        if self.timestep_counter % self.step == 0:
            self.input_spike_recorder[
                self.timestep_counter
            ] = input_spikes.cpu().numpy()
            self.reservoir_spike_recorder[
                self.timestep_counter
            ] = reservoir_spikes.cpu().numpy()

        self.timestep_counter += 1
        if self.timestep_counter == self.nb_of_timesteps:
            self.spike_recorder.close()

        return True


class StateRecorder:
    """
    Record state tensors at given interval
    """

    def __init__(
        self,
        file_output_name: str,
        nb_of_timesteps: int,
        *args: List[Tuple[str, int]],
        step: int = 1,
    ):
        """
        Initialize state recorder
        :param file_output_name: string of output file name
        :param nb_of_timesteps: record this amount of iters
        :param *args: tuples of (state_name, state_size)
        :param step: record every this amount iters
        """
        self.nb_of_timesteps = nb_of_timesteps
        self.step = step
        self.file_hndlr = File(file_output_name, "w-")
        self.state_hndlrs = [
            self.file_hndlr.create_dataset(
                state_name,
                (nb_of_timesteps // step, *np.array(state_size).flatten()),
                dtype="f",
            )
            for state_name, state_size in args
        ]
        self.timestep_counter = 0

    def __call__(self, *args) -> bool:
        if self.timestep_counter >= self.nb_of_timesteps:
            return False

        if self.timestep_counter % self.step == 0:
            assert len(args) == len(
                self.state_hndlrs
            ), "State Recorder length do not match with the number of state recorded"
            for recording, state_hndlr in zip(args, self.state_hndlrs):
                state_hndlr[
                    self.timestep_counter // self.step
                ] = recording.cpu().numpy()

        self.timestep_counter += 1
        if self.timestep_counter == self.nb_of_timesteps:
            self.file_hndlr.close()
        return True
