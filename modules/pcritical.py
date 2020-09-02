import torch
import torch.nn as nn
import networkx as netx
import quantities as units
import numpy as np
from .utils import graph_to_tensor


class PCritical(nn.Module):
    def __init__(
        self,
        batch_size,
        topology: netx.DiGraph,
        alpha=0.025,
        beta=0.00025,
        tau_v=50 * units.ms,
        tau_i=1 * units.ms,
        v_th=1.0,
        refractory_period=2 * units.ms,
        stochastic_alpha=False,
        dt=1 * units.ms,
        tau_v_pair=None,
        tau_i_pair=None,
        dtype=torch.float32,
    ):
        super().__init__()
        self.N = topology.number_of_nodes()
        self.alpha = alpha
        self.beta = beta
        self.plasticity = True
        self.stochastic_alpha = stochastic_alpha

        # Store the sparse adj matrix as dense to facilitate operations
        self.W_rec = graph_to_tensor(topology).type(dtype).to_dense()
        self.sign_mask = torch.sign(self.W_rec) > 0

        # Calculate constants from tau_v and tau_i
        tau_v = tau_v.rescale(dt.units)
        tau_i = tau_i.rescale(dt.units)

        # Potential scale constant (at each time step)
        self.inv_tau_i = np.exp(-(dt / tau_i).magnitude).item()
        self.inv_tau_v = np.exp(-(dt / tau_v).magnitude).item()

        if tau_v_pair is None:
            tau_v_pair = tau_v
        if tau_i_pair is None:
            tau_i_pair = tau_i

        self.inv_tau_v_pair = (
            0.0
            if tau_v_pair == 0 * units.ms
            else np.exp(-(dt / tau_v_pair).magnitude).item()
        )
        self.inv_tau_i_pair = (
            0.0
            if tau_i_pair == 0 * units.ms
            else np.exp(-(dt / tau_i_pair).magnitude).item()
        )

        # Neuron states
        self.refrac = torch.tensor(
            [(refractory_period.rescale(dt.units) / dt).magnitude.astype(int).item()],
            dtype=torch.int32,
        )
        self._batch_size = batch_size
        self._set_mem_state(dtype, None)
        self.v_th = v_th
        self.t = torch.Tensor([0.0])
        self.exp_tau = (tau_v / dt).magnitude.item()

    def forward(self, inp):
        """This method takes an input spike train and pass it through the PCritical reservoir (for one time step)

        Arguments:
            inp {Torch.Tensor} -- Input membrane current

        Returns:
            [Torch.Tensor] -- Return the spikes that occurred in the reservoir before integration of the input
        """

        # Decrement refrac counter
        self.refrac_neurons = torch.where(
            self.refrac_neurons > 0, self.refrac_neurons - 1, self.refrac_neurons
        )

        # Spikes
        S = (self.mem_pot - self.v_th).ceil_().clamp_(0, 1)
        S_paired = (self.mem_pot_paired - self.v_th - self.alpha).ceil_().clamp_(0, 1)
        self.S_paired = S_paired  # Store in obj for possible debugging

        # Plasticity
        if self.plasticity:  # and (inp.sum() > 0 or S.sum() > 0)
            S_paired_batch = (
                self.S_paired.max(dim=0, keepdim=True)
                .values.view(self.N, 1)
                .expand_as(self.W_rec)
            )

            factor = (
                self.alpha
                * torch.rand(
                    (self.N, 1), dtype=self.mem_pot.dtype, device=self.mem_pot.device
                )
                if self.stochastic_alpha
                else self.alpha
            )
            self.st = torch.where(S > 0, torch.ones_like(self.st) * self.t, self.st)
            max_st = self.st.max(dim=0, keepdim=False).values
            a, b = torch.meshgrid(max_st, max_st)
            # self.sp += S.matmul(self.sign_mask.t().float())

            update_mask = (S_paired_batch > 0) & self.sign_mask

            updated_weights = self.W_rec + self.beta
            updated_weights[update_mask] -= factor * torch.exp(
                (a - b)[update_mask].abs() / self.exp_tau
            )
            updated_weights = updated_weights.clamp_(0.0, 1.0)
            self.W_rec = torch.where(self.sign_mask, updated_weights, self.W_rec)

            # expo = torch.exp((a-b).abs() / self.exp_tau)
            # update_value = self.beta - factor * S_paired_batch * expo
            # new_weights = (
            #     (update_value + self.W_rec).clamp_(0.0, 1.0)
            # )

            # # Only update excitatory weights
            # self.W_rec = torch.where(self.sign_mask, new_weights, self.W_rec)

        # Input + recurrent propagation of the spikes
        active_neurons_mask = self.refrac_neurons == 0

        self.mem_cur = torch.where(
            active_neurons_mask, inp + S.matmul(self.W_rec) + self.mem_cur, self.mem_cur
        )
        self.mem_cur_paired = torch.where(
            active_neurons_mask,
            S.matmul(self.W_rec.t()) + self.mem_cur_paired,
            self.mem_cur_paired,
        )

        # Current integration to pot
        self.mem_pot = torch.where(
            active_neurons_mask, self.mem_cur + self.mem_pot, self.mem_pot
        )

        self.mem_pot_paired = torch.where(
            active_neurons_mask,
            self.mem_cur_paired + self.mem_pot_paired,
            self.mem_pot_paired,
        )

        # Leaks
        self.mem_pot *= self.inv_tau_v
        self.mem_cur *= self.inv_tau_i
        self.mem_pot_paired *= self.inv_tau_v_pair
        self.mem_cur_paired *= self.inv_tau_i_pair
        # self.sp *= self.inv_tau_v

        # Reset
        self.mem_pot = torch.where(S > 0, torch.zeros_like(self.mem_pot), self.mem_pot)
        self.mem_pot_paired = torch.where(
            S_paired > 0, torch.zeros_like(self.mem_pot_paired), self.mem_pot_paired
        )

        self.refrac_neurons = torch.where(S > 0, self.refrac, self.refrac_neurons)
        self.t += 1

        return S

    def reset_neuron_states(self):
        """Reset the membrane potential and current for the reservoir (can be used in-between samples)
        """
        self.mem_pot[:] = 0
        self.mem_cur[:] = 0
        self.mem_pot_paired[:] = 0
        self.mem_cur_paired[:] = 0
        self.refrac_neurons[:] = 0
        # self.sp = torch.zeros_like(self.sp)

    def _set_mem_state(self, dtype, device):
        self.mem_pot = torch.zeros(
            (self._batch_size, self.N), dtype=dtype, device=device
        )
        self.mem_cur = torch.zeros(
            (self._batch_size, self.N), dtype=dtype, device=device
        )
        self.mem_pot_paired = torch.zeros(
            (self._batch_size, self.N), dtype=dtype, device=device
        )
        self.mem_cur_paired = torch.zeros(
            (self._batch_size, self.N), dtype=dtype, device=device
        )
        self.refrac_neurons = torch.zeros(
            (self._batch_size, self.N), dtype=torch.int32, device=device
        )
        # self.sp = torch.zeros((self.N, self.N), dtype=dtype, device=device)
        self.st = torch.zeros(self.N, dtype=dtype, device=device)

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, batch_size):
        if self._batch_size != batch_size:
            self._batch_size = batch_size
            self._set_mem_state(self.mem_pot.dtype, self.mem_pot.device)

        else:
            self.reset_neuron_states()  # For consistency with a batch size change

    def _apply(self, fn):
        super(PCritical, self)._apply(fn)
        self.W_rec = fn(self.W_rec)
        self.mem_pot = fn(self.mem_pot)
        self.mem_cur = fn(self.mem_cur)
        self.mem_pot_paired = fn(self.mem_pot_paired)
        self.mem_cur_paired = fn(self.mem_cur_paired)
        self.sign_mask = fn(self.sign_mask)
        self.refrac_neurons = fn(self.refrac_neurons)
        self.refrac = fn(self.refrac)
        # self.sp = fn(self.sp)
        self.st = fn(self.st)
        self.t = fn(self.t)
        return self
