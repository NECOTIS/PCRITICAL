from dataclasses import dataclass, asdict
from typing import Union, Tuple
import logging
import numpy as np
import networkx as nx
from modules import reporter
from scipy.spatial.distance import cdist
from scipy.sparse import bsr_matrix, vstack
from sklearn.metrics import pairwise_distances_chunked


_logger = logging.getLogger(__name__)


class SmallWorldTopology(nx.DiGraph):
    """Create a small-world type topology by creating a i1*i2*i3 cube of neurons separated by distance
    neuron_spacing in a 3d shape of j1*j2*j3 cubes distanced by minicolumn_spacing. i1, i2, i3 is the
    minicolumn_shape while j1, j2, j3 is the macrocolumn_shape. Connectivity is distance based with prob-
    ability of p_max * e^(-dist / intracolumnar_sparseness).
    """

    @dataclass(frozen=True)
    class Configuration:
        minicolumn_shape: Tuple[int, int, int] = (2, 2, 2)
        macrocolumn_shape: Tuple[int, int, int] = (4, 4, 4)
        neuron_spacing: float = 10.0
        minicolumn_spacing: float = 100.0
        p_max: float = 0.056
        intracolumnar_sparseness: float = 3 * 125

        # Construct the topology using sparse matrices of size mem_available
        sparse_init: bool = False
        mem_available: int = 8 * 1024

        init_weights: bool = True  # If true, init the weights with following parameters:
        inhibitory_prob: float = 0.2  # Ratio of inhibitory neurons [0, 1]
        inhibitory_init_weight_range: Tuple[float, float] = (0.01, 0.2)
        excitatory_init_weight_range: Tuple[float, float] = (0.1, 0.7)

        spectral_radius_norm: bool = False

    def __init__(self, configs: Union[Configuration, dict, nx.DiGraph]):
        if type(configs) == nx.DiGraph:  # Assume we're creating a copy
            super().__init__(configs)
            return
        elif type(configs) == dict:
            configs = SmallWorldTopology.Configuration(**configs)

        super().__init__()
        self.__dict__.update(asdict(configs))

        assert (
            len(self.minicolumn_shape) == 3
        ), "Minicolumn shape must be of dimension 3 (3D)"
        assert (
            len(self.macrocolumn_shape) == 3
        ), "Macrocolumn shape must be of dimension 3 (3D)"

        # Initial neuron positions (all separated by neuron_spacing)
        i, j, k = np.multiply(self.macrocolumn_shape, self.minicolumn_shape)
        grid = np.mgrid[:i, :j, :k].reshape(3, -1)
        x, y, z = grid * self.neuron_spacing

        # Adding minicolumnSpacing (from random to small world topology)
        if self.minicolumn_spacing > 0:
            for d in range(3):  # For each dimension
                grid[d] //= self.minicolumn_shape[d]
            x += grid[0] * self.minicolumn_spacing
            y += grid[1] * self.minicolumn_spacing
            z += grid[2] * self.minicolumn_spacing

        positions = map(lambda p: {"position": p}, zip(x, y, z))
        self.add_nodes_from(zip(range(len(x)), positions))

        # Distance-based random connectivity
        positions = np.stack(np.asarray(self.nodes.data("position"))[:, 1])

        if (
            self.sparse_init
        ):  # Slower but iterative (for adjacency matrices that don't fit in memory)
            distances = pairwise_distances_chunked(
                positions,
                metric="euclidean",
                n_jobs=-1,
                reduce_func=lambda chunk, start: bsr_matrix(
                    np.random.random(chunk.shape)
                    < self.p_max * np.exp(-chunk / self.intracolumnar_sparseness)
                ),
                working_memory=self.mem_available,
            )
            adjacency_matrix = vstack(list(distances))
            adjacency_matrix.setdiag(0)  # Avoid self-connections
            self.add_edges_from(zip(*adjacency_matrix.nonzero()))
        else:
            distances = cdist(positions, positions, "euclidean")
            probabilities = self.p_max * np.exp(
                -distances / self.intracolumnar_sparseness
            )
            np.fill_diagonal(probabilities, 0)  # Avoid self-connections
            rand_matrix = np.random.random(probabilities.shape)
            i, j = np.nonzero(rand_matrix < probabilities)
            self.add_edges_from(zip(i, j))

        n_neurons = self.number_of_nodes()
        self.inhibitory_neurons = set(
            np.random.permutation(n_neurons)[: int(n_neurons * self.inhibitory_prob)]
        )

        for u, v in self.edges:
            if u in self.inhibitory_neurons:
                self.edges[u, v]["weight"] = -np.random.uniform(
                    *self.inhibitory_init_weight_range
                )
            else:
                self.edges[u, v]["weight"] = np.random.uniform(
                    *self.excitatory_init_weight_range
                )

        if self.spectral_radius_norm:
            spectral_radius = lambda matrix: np.max(np.abs(np.linalg.eigvals(matrix)))
            adj = nx.adjacency_matrix(self, weight="weight").todense()
            scale = 1.0 / spectral_radius(np.abs(adj))

            for i, (u, v) in enumerate(self.edges):
                self.edges[u, v]["weight"] = self.edges[u, v]["weight"] * scale

        if _logger.isEnabledFor(logging.INFO):
            # Some extra info about the topology
            out_degrees = np.array(self.out_degree())[:, 1]
            reporter.log_metrics(
                {
                    "number-of-neurons": n_neurons,
                    "number-of-synapses": self.number_of_edges(),
                    "excitatory-ratio": 100.0
                    * (1.0 - len(self.inhibitory_neurons) / n_neurons),
                    "avg-out-degree": np.mean(out_degrees),
                    "nb-out-degree-0": len(out_degrees) - np.count_nonzero(out_degrees),
                    "nb-isolates": nx.number_of_isolates(self),
                }
            )

            # if not self.sparse_init:
            #    algebraic_connectivity = nx.algebraic_connectivity(self.to_undirected())

            # sigma = nx.sigma(self.to_undirected(), seed=np.random.randint(0, 2**32-1))
            # _logger.info("Small-world coefficient (sigma): %.5", sigma)

            # omega = nx.omega(self.to_undirected(), seed=np.random.randint(0, 2**32-1))
            # _logger.info("Small-world coefficient (omega): %.5", omega)

            # rich_club_coefficient = nx.rich_club_coefficient(self.to_undirected(), seed=np.random.randint(0, 2**32-1))
            # avg_rich_club_coeff = np.mean(list(rich_club_coefficient.values()))
            # _logger.info("Rich club coefficient: %.5f", avg_rich_club_coeff)
