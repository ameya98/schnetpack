from typing import Callable, Dict, Tuple
import torch
from torch import nn
import e3nn

import schnetpack.properties as structure
from schnetpack.nn import Dense, scatter_add
from schnetpack.nn.activations import shifted_softplus

import schnetpack.nn as snn

__all__ = ["E3SchNet", "E3SchNetInteraction"]


def irreps_mul_to_axis(
    input_irreps: e3nn.o3.Irreps, num_channels: int
) -> e3nn.o3.Irreps:
    """Returns the corresponding irreps for the output of mul_to_axis()."""
    return e3nn.o3.Irreps(
        [(mul // num_channels, (ir.l, ir.p)) for mul, ir in input_irreps]
    )


def irreps_axis_to_mul(
    input_irreps: e3nn.o3.Irreps, num_channels: int
) -> e3nn.o3.Irreps:
    """Returns the corresponding irreps for the output of axis_to_mul()."""
    return e3nn.o3.Irreps(
        [(mul * num_channels, (ir.l, ir.p)) for mul, ir in input_irreps]
    )


def mul_to_axis(
    input: torch.Tensor, input_irreps: e3nn.o3.Irreps, num_channels: int
) -> Tuple[torch.Tensor, e3nn.o3.Irreps]:
    """Reshapes a tensor of shape (num_atoms, num_channels * irreps.dim) to (num_atoms, num_channels, irreps.dim)."""

    assert (
        input.shape[-1] == input_irreps.dim
    ), f"Expected {input_irreps.dim} features, got {input.shape[1]}"
    assert all(mul % num_channels == 0 for mul, _ in input_irreps)

    def get_slices_for_channel(channel):
        """Returns a slice for each irrep in input_irreps for the given channel."""
        for irrep_slice in input_irreps.slices():
            irrep_dim = (irrep_slice.stop - irrep_slice.start) // num_channels
            yield slice(
                irrep_slice.start + channel * irrep_dim,
                irrep_slice.start + (channel + 1) * irrep_dim,
            )

    num_atoms = input.shape[0]
    output_irreps = irreps_mul_to_axis(input_irreps, num_channels)
    output = torch.zeros_like(input).reshape((num_atoms, num_channels, output_irreps.dim))
    for channel in range(num_channels):
        start = 0
        for irrep_slice in get_slices_for_channel(channel):
            output[
                :, channel, start : start + irrep_slice.stop - irrep_slice.start
            ] = input[:, irrep_slice]
            start += irrep_slice.stop - irrep_slice.start
    return output


def axis_to_mul(
    input: torch.Tensor, input_irreps: e3nn.o3.Irreps
) -> Tuple[torch.Tensor, e3nn.o3.Irreps]:
    """Reshapes a tensor of shape (num_atoms, num_channels, irreps.dim) to (num_atoms, num_channels * irreps.dim)."""

    assert (
        input.shape[-1] == input_irreps.dim
    ), f"Expected {input_irreps.dim} features, got {input.shape[1]}"

    def get_slices_for_channel(channel):
        """Returns a slice for each irrep in input_irreps for the given channel."""
        for irrep_slice in output_irreps.slices():
            irrep_dim = (irrep_slice.stop - irrep_slice.start) // num_channels
            yield slice(
                irrep_slice.start + channel * irrep_dim,
                irrep_slice.start + (channel + 1) * irrep_dim,
            )

    num_atoms, num_channels = input.shape[0], input.shape[1]
    output_irreps = irreps_axis_to_mul(input_irreps, num_channels)
    output = torch.zeros_like(input).reshape((num_atoms, output_irreps.dim))
    for channel in range(num_channels):
        start = 0
        for irrep_slice in get_slices_for_channel(channel):
            output[:, irrep_slice] = input[
                :, channel, start : start + irrep_slice.stop - irrep_slice.start
            ]
            start += irrep_slice.stop - irrep_slice.start
    return output


class E3SchNetInteraction(nn.Module):
    r"""E(3)-equivariant SchNet interaction block for modeling interactions of atomistic systems."""

    def __init__(
        self,
        n_atom_basis: int,
        n_rbf: int,
        n_filters: int,
        max_ell: int,
        activation: Callable = shifted_softplus,
    ):
        """
        Args:
            n_atom_basis: number of features to describe atomic environments.
            n_rbf (int): number of radial basis functions.
            n_filters: number of filters used in continuous-filter convolution.
            activation: if None, no activation function is used.
        """
        super(E3SchNetInteraction, self).__init__()
        self.n_atom_basis = n_atom_basis
        self.n_filters = n_filters
        self.max_ell = max_ell

        input_irreps = self.n_atom_basis * e3nn.o3.Irreps.spherical_harmonics(
            self.max_ell
        )
        input_irreps = input_irreps.sort().irreps.simplify()

        irreps_after_in2f = self.n_filters * e3nn.o3.Irreps.spherical_harmonics(
            self.max_ell
        )
        self.irreps_after_in2f = irreps_after_in2f.sort().irreps.simplify()
        self.in2f = e3nn.o3.Linear(
            irreps_in=input_irreps, irreps_out=self.irreps_after_in2f
        )

        self.irreps_after_mul_to_axis = irreps_mul_to_axis(
            self.irreps_after_in2f, self.n_filters
        )

        self.Yr_irreps = e3nn.o3.Irreps.spherical_harmonics(self.max_ell)
        self.tensor_product_x_Yr = e3nn.o3.FullTensorProduct(
            self.irreps_after_mul_to_axis, self.Yr_irreps
        )
        self.irreps_after_tensor_product_x_Yr = self.tensor_product_x_Yr.irreps_out

        self.irreps_after_axis_to_mul = irreps_axis_to_mul(
            self.irreps_after_tensor_product_x_Yr, self.n_filters
        )

        self.W_irreps = e3nn.o3.Irreps(f"{self.irreps_after_axis_to_mul.num_irreps}x0e")
        self.filter_network = nn.Sequential(
            Dense(n_rbf, n_filters, activation=activation),
            Dense(n_filters, self.W_irreps.dim),
        )

        self.continuous_filter_convolution = e3nn.o3.ElementwiseTensorProduct(
            self.irreps_after_axis_to_mul, self.W_irreps
        )
        self.irreps_after_continuous_filter_convolution = (
            self.continuous_filter_convolution.irreps_out
        )

        output_irreps = input_irreps
        self.f2out_1 = e3nn.o3.Linear(
            irreps_in=self.irreps_after_continuous_filter_convolution,
            irreps_out=output_irreps,
        )
        self.f2out_act = e3nn.nn.Activation(
            irreps_in=output_irreps,
            acts=[activation if ir.l == 0 else None for _, ir in output_irreps],
        )
        self.f2out_2 = e3nn.o3.Linear(irreps_in=output_irreps, irreps_out=output_irreps)

    def forward(
        self,
        x: torch.Tensor,
        idx_i: torch.Tensor,
        idx_j: torch.Tensor,
        f_ij: torch.Tensor,
        rcut_ij: torch.Tensor,
        r_ij: torch.Tensor,
    ):
        """Compute interaction output.

        Args:
            x: input values
            idx_i: index of center atom i
            idx_j: index of neighbors j
            f_ij: d_ij passed through the embedding function
            rcut_ij: d_ij passed through the cutoff function
            r_ij: relative position of neighbor j to atom i
        Returns:
            atom features after interaction
        """
        # Embed the inputs.
        x = self.in2f(x)

        # Previously x_j.shape == (num_edges, n_filters * x_irreps.dim)
        # We want x_j.shape == (num_edges, n_filters, x_irreps.dim)
        x_j = x[idx_j]
        x_j = mul_to_axis(x_j, self.irreps_after_in2f, self.n_filters)

        # Compute the spherical harmonics of relative positions.
        # r_ij: (n_edges, 3)
        # Yr_ij: (n_edges, (max_ell + 1) ** 2)
        Yr_ij = e3nn.o3.spherical_harmonics(self.Yr_irreps, r_ij, normalize=True)
        # Reshape Yr_ij to (num_edges, 1, x_irreps.dim).
        Yr_ij = Yr_ij.reshape((Yr_ij.shape[0], 1, Yr_ij.shape[1]))

        # Apply e3nn.o3.FullTensorProduct to get new x_j of shape (num_edges, n_filters, new_x_irreps).
        x_j = self.tensor_product_x_Yr(x_j, Yr_ij)

        # Reshape x_j back to (num_edges, n_filters * x_irreps.dim).
        x_j = axis_to_mul(x_j, self.irreps_after_tensor_product_x_Yr)

        # Compute filter.
        Wij = self.filter_network(f_ij)
        Wij = Wij * rcut_ij[:, None]

        # Continuous-filter convolution.
        x_ij = self.continuous_filter_convolution(x_j, Wij)
        x = scatter_add(x_ij, idx_i, dim_size=x.shape[0])

        # Apply final linear and activation layer.
        x = self.f2out_1(x)
        x = self.f2out_act(x)
        x = self.f2out_2(x)
        return x


class E3SchNet(nn.Module):
    """E(3)-equivariant SchNet architecture for learning representations of atomistic systems

    Reduces to standard SchNet when max_ell = 0.

    References:

    .. [#schnet1] Schütt, Arbabzadah, Chmiela, Müller, Tkatchenko:
       Quantum-chemical insights from deep tensor neural networks.
       Nature Communications, 8, 13890. 2017.
    .. [#schnet_transfer] Schütt, Kindermans, Sauceda, Chmiela, Tkatchenko, Müller:
       SchNet: A continuous-filter convolutional neural network for modeling quantum
       interactions.
       In Advances in Neural Information Processing Systems, pp. 992-1002. 2017.
    .. [#schnet3] Schütt, Sauceda, Kindermans, Tkatchenko, Müller:
       SchNet - a deep learning architecture for molceules and materials.
       The Journal of Chemical Physics 148 (24), 241722. 2018.

    """

    def __init__(
        self,
        n_atom_basis: int,
        n_interactions: int,
        radial_basis: nn.Module,
        cutoff_fn: Callable,
        n_filters: int = None,
        shared_interactions: bool = False,
        max_z: int = 100,
        activation: Callable = shifted_softplus,
        max_ell: int = 3,
    ):
        """
        Args:
            n_atom_basis: number of features to describe atomic environments.
                This determines the size of each embedding vector; i.e. embeddings_dim.
            n_interactions: number of interaction blocks.
            radial_basis: layer for expanding interatomic distances in a basis set
            cutoff_fn: cutoff function
            n_filters: number of filters used in continuous-filter convolution
            shared_interactions: if True, share the weights across
                interaction blocks and filter-generating networks.
            max_z: maximal nuclear charge
            activation: activation function
        """
        super().__init__()
        self.n_atom_basis = n_atom_basis
        self.size = (self.n_atom_basis,)
        self.n_filters = n_filters or self.n_atom_basis
        self.radial_basis = radial_basis
        self.cutoff_fn = cutoff_fn
        self.cutoff = cutoff_fn.cutoff
        self.max_ell = max_ell

        latent_irreps = self.n_atom_basis * e3nn.o3.Irreps.spherical_harmonics(
            self.max_ell
        )
        latent_irreps = latent_irreps.sort().irreps.simplify()
        self.latent_irreps = latent_irreps

        # layers
        self.embedding = nn.Embedding(max_z, self.n_atom_basis, padding_idx=0)
        self.post_embedding = e3nn.o3.Linear(
            irreps_in=f"{self.n_atom_basis}x0e", irreps_out=self.latent_irreps
        )
        self.interactions = snn.replicate_module(
            lambda: E3SchNetInteraction(
                n_atom_basis=self.n_atom_basis,
                n_rbf=self.radial_basis.n_rbf,
                n_filters=self.n_filters,
                activation=activation,
                max_ell=self.max_ell,
            ),
            n_interactions,
            shared_interactions,
        )

    def forward(self, inputs: Dict[str, torch.Tensor]):
        atomic_numbers = inputs[structure.Z]
        r_ij = inputs[structure.Rij]
        idx_i = inputs[structure.idx_i]
        idx_j = inputs[structure.idx_j]

        # Compute atom embeddings.
        # Initially, the atom embeddings are just scalars.
        x = self.embedding(atomic_numbers)
        x = self.post_embedding(x)

        # Compute radial basis functions to cut off interactions
        d_ij = torch.norm(r_ij, dim=1)
        f_ij = self.radial_basis(d_ij)
        rcut_ij = self.cutoff_fn(d_ij)
        r_ij = r_ij * rcut_ij[:, None]

        assert r_ij.shape == (r_ij.shape[0], 3), r_ij.shape

        # Compute interaction block to update atomic embeddings
        for interaction in self.interactions:
            v = interaction(x, idx_i, idx_j, f_ij, rcut_ij, r_ij)
            x = x + v

        # Extract only the scalars.
        inputs["scalar_representation"] = x[:, self.latent_irreps.slices()[0]]
        return inputs
