"""Post-training changes of basis for shared-basis surrogate models."""

from typing import Any
import torch
from src.model import FCModel


def representative_coefficients(
    model: FCModel, params: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return signal and derivative coefficients before the shared decoder."""
    with torch.no_grad():
        network_input, tangent_T1, tangent_T2 = (
            model.input_transform.forward_with_tangents(params)
        )
        return model.subnet_signal.forward_with_tangents(
            network_input, tangent_T1, tangent_T2
        )


def orthogonalize_shared_basis(model: FCModel, params: torch.Tensor) -> None:
    """Make the trained decoder orthonormal and order it by target variance.

    Rewrite the signal network's final layer with the inverse change of basis.
    At full rank the signal and both derivatives remain unchanged apart from
    floating-point rounding. The ordering uses the supplied representative
    physical parameters and allows meaningful truncation afterward.
    """
    if model.derivative_mode != "forward":
        raise ValueError("Basis reordering requires derivative_mode='forward'")
    if params.ndim != 2 or params.shape[0] == 0 or params.shape[1] != 3:
        raise ValueError("params must have shape (n_samples, 3) with n_samples > 0")

    decoder = model.U_signal[0]
    if not isinstance(decoder, torch.nn.Linear):
        raise TypeError("Expected the first signal decoder module to be Linear")
    last_layer = model.subnet_signal.net[-1]
    if not isinstance(last_layer, torch.nn.Linear) or last_layer.bias is None:
        raise TypeError("Expected subnet_signal's last layer to be a biased Linear")

    weight = decoder.weight.detach()
    if weight.shape[0] < weight.shape[1]:
        raise ValueError("The decoder rank must not exceed its output dimension")
    orthonormal_raw, upper_triangular = torch.linalg.qr(weight, mode="reduced")

    signal, dT1, dT2 = representative_coefficients(model, params)
    rank = upper_triangular.shape[0]
    covariance = torch.zeros((rank, rank), dtype=weight.dtype, device=weight.device)
    for coefficients in (signal, dT1, dT2):
        rotated = coefficients @ upper_triangular.T
        rotated = rotated / torch.linalg.vector_norm(
            rotated, dim=1, keepdim=True
        ).clamp_min(1e-4)
        covariance.addmm_(rotated.T, rotated)
    covariance = 0.5 * (covariance + covariance.T)

    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    rotation = eigenvectors[:, torch.argsort(eigenvalues, descending=True)]

    # c -> c @ change_of_basis for values and both tangents. The final linear
    # layer applies the same transform to each because tangents omit only bias.
    change_of_basis = upper_triangular.T @ rotation
    with torch.no_grad():
        last_layer.weight.copy_(change_of_basis.T @ last_layer.weight)
        last_layer.bias.copy_(last_layer.bias @ change_of_basis)
        decoder.weight.copy_(orthonormal_raw @ rotation)


def truncate_shared_basis(
    model: FCModel,
    model_config: dict[str, Any],
    data_config: dict[str, Any],
    rank: int,
) -> FCModel:
    """Build a smaller model from the first columns of an ordered decoder.

    Call after ``orthogonalize_shared_basis``. Truncation changes predictions;
    the full-rank basis rewrite itself does not.
    """
    if model.derivative_mode != "forward":
        raise ValueError("Basis truncation requires derivative_mode='forward'")
    full_rank = model.rank
    if not 0 < rank <= full_rank:
        raise ValueError(f"rank must be in (0, {full_rank}], got {rank}")

    smaller_config = dict(model_config)
    smaller_config["NB_signal"] = rank
    smaller = FCModel(model.NTR, smaller_config, data_config)
    state = {key: value.clone() for key, value in model.state_dict().items()}
    state["U_signal.0.weight"] = state["U_signal.0.weight"][:, :rank]
    last_layer_key = f"subnet_signal.net.{len(model.subnet_signal.net) - 1}"
    state[f"{last_layer_key}.weight"] = state[f"{last_layer_key}.weight"][:rank]
    state[f"{last_layer_key}.bias"] = state[f"{last_layer_key}.bias"][:rank]
    smaller.load_state_dict(state)
    smaller.to(next(model.parameters()).device)
    smaller.eval()
    return smaller


def worst_relative_output_difference(
    reference: FCModel, transformed: FCModel, params: torch.Tensor
) -> float:
    """Measure the largest relative difference across the three outputs."""
    with torch.no_grad():
        ref_outputs = reference(params)
        transformed_outputs = transformed(params)
    worst = 0.0
    for ref, changed in zip(ref_outputs, transformed_outputs, strict=True):
        scale = torch.linalg.vector_norm(ref).clamp_min(1e-8)
        worst = max(worst, (torch.linalg.vector_norm(ref - changed) / scale).item())
    return worst
