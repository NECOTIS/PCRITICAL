P-CRITICAL
==========

Adaptation of the CRITICAL plasticity rule, available at <https://github.com/NECOTIS/CRITICAL>, for neuromorphic devices. This is the code for the preprint available at <https://arxiv.org/pdf/2009.05593.pdf>.

# Installation

The required python packages are listed in requirements.txt, and can be installed with: 
```bash
pip install -r requirements.txt
```

# Usage

Most experiments can be executed using the command line interface:
```
cd pcritical
python -m scripts.random_poisson
```

Other datasets and experiments can be created using primarily the PCritical torch module:
```python
import torch
import numpy as np
import matplotlib.pyplot as plt
from networkx import DiGraph
from torch.nn import Sequential
from modules.pcritical import PCritical
from modules.utils import OneToNLayer, display_spike_train


# Reservoir can be constructed using networkx's Directed Graphs
topology = DiGraph()
topology.add_nodes_from(range(32))
for i in range(31):
    topology.add_edge(i, i+1, weight=np.random.random())

# Input and reservoir layers of a liquid state machine:
NFeatures = 10
lsm = Sequential(
    OneToNLayer(1, NFeatures, 32),  # Input layer of 10 neurons randomly connected to 10 out of 32 neurons (one-to-one)
    PCritical(1, topology=topology)
)

duration = 120
input_spike_train = torch.tensor(np.random.poisson(
    lam=0.5, size=(1, NFeatures, duration)
).clip(0, 1)).float()

output_spikes = [lsm(input_spike_train[:, :, t]) for t in range(duration)]
output_spikes = torch.stack(output_spikes, dim=2)

display_spike_train(output_spikes[0].numpy())
plt.show()

```

# Acknowledgements

© Copyright (August 2020) Ismael Balafrej, prof. Jean Rouat. University of Sherbrooke. [NEuro COmputational & Intelligent Signal Processing Research Group (NECOTIS)](http://www.gel.usherbrooke.ca/necotis/)

<img src="images/necotis.png" width="250" /> <img src="images/UdeS.jpg" width="250" />
