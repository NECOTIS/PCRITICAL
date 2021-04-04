from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import numpy as np


LinearWithBN = lambda d_in, d_out: nn.Sequential(
    nn.BatchNorm1d(d_in), nn.Linear(d_in, d_out)
)


class ReservoirReadout(nn.Module, ABC):
    @abstractmethod
    def number_of_features(self) -> int:
        return -1


class TimeBinningLayer(ReservoirReadout):
    def __init__(self, bin_size, max_duration, nb_of_neurons):
        super().__init__()
        self.bin_size = bin_size
        self.nb_of_bins = max_duration // bin_size + 1
        self.bins = None
        self.nb_of_neurons = nb_of_neurons
        self._dummy = torch.empty(1)  # Keep track of _apply functions

    def number_of_features(self) -> int:
        return self.nb_of_bins * self.nb_of_neurons

    def forward(self, spikes, time, **_):
        if self.bins is None:
            self.bins = torch.zeros(
                spikes.shape[0],
                self.nb_of_neurons,
                self.nb_of_bins,
                device=self._dummy.device,
                dtype=self._dummy.dtype,
            )
        self.bins[:, :, time // self.bin_size] += spikes
        return self.bins.view(spikes.shape[0], -1)

    def reset(self):
        self.bins = None

    def _apply(self, fn):
        super()._apply(fn)
        self._dummy = fn(self._dummy)
        return self


class ExponentialFilterLayer(ReservoirReadout):
    def __init__(self, tau, nb_of_neurons):
        super().__init__()
        self.tau = tau
        self.filter_output = None
        self.nb_of_neurons = nb_of_neurons
        self._dummy = torch.empty(1)  # Keep track of _apply functions

    def number_of_features(self) -> int:
        return self.nb_of_neurons

    def forward(self, spikes, time, duration, **_):
        if self.filter_output is None:
            self.filter_output = torch.zeros(
                spikes.shape[0],
                self.nb_of_neurons,
                device=self._dummy.device,
                dtype=self._dummy.dtype,
            )

        self.filter_output += spikes * np.exp(-(duration - time) / self.tau)
        return self.filter_output

    def reset(self):
        self.filter_output = None

    def _apply(self, fn):
        super()._apply(fn)
        self._dummy = fn(self._dummy)
        return self


class ReverseExponentialFilterLayer(ReservoirReadout):
    def __init__(self, tau, nb_of_neurons):
        super().__init__()
        self.tau = tau
        self.filter_output = None
        self._dummy = torch.empty(1)  # Keep track of _apply functions
        self.nb_of_neurons = nb_of_neurons

    def number_of_features(self) -> int:
        return self.nb_of_neurons

    def forward(self, spikes, time, **_):
        if self.filter_output is None:
            self.filter_output = torch.zeros(
                spikes.shape[0],
                spikes.shape[1],
                device=self._dummy.device,
                dtype=self._dummy.dtype,
            )

        self.filter_output += spikes * np.exp(-time / self.tau)
        return self.filter_output

    def reset(self):
        self.filter_output = None

    def _apply(self, fn):
        super()._apply(fn)
        self._dummy = fn(self._dummy)
        return self
