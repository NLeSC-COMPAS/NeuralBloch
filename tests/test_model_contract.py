import math
from collections.abc import Callable
from pathlib import Path
from typing import Any
import h5py
import numpy as np
import pytest
import torch
from data.data_loading import load_data_from_h5_file
from data.data_loading import make_loader
from src.loss import nmae_loss
from src.metrics import nrmse
from src.model import FCModel
from src.utils import complex_to_real_imag
from src.utils import sample_parameters


SAMPLING_CONFIG = {
    "data": {
        "T1_lower": 0.1,
        "T1_upper": 5.0,
        "T2_lower": 0.01,
        "T2_upper": 2.0,
        "B1_lower": 0.8,
        "B1_upper": 1.2,
    }
}


def test_model_logs_physical_parameters_and_returns_pairs(
    direct_config: dict[str, Any], make_model: Callable[..., FCModel]
) -> None:
    model = make_model(5, direct_config).eval()
    captured_inputs: list[torch.Tensor] = []

    def capture_input(
        _module: torch.nn.Module,
        inputs: tuple[torch.Tensor, ...],
    ) -> None:
        captured_inputs.append(inputs[0].detach().clone())

    handle = model.subnet_signal.register_forward_pre_hook(capture_input)
    x = torch.tensor([[0.8, 0.07, 1.0], [1.2, 0.1, 0.9]])

    try:
        with torch.no_grad():
            outputs = model(x)
    finally:
        handle.remove()

    expected_input = torch.cat((torch.log(x[..., :2]), x[..., 2:]), dim=-1)
    assert len(captured_inputs) == 1
    torch.testing.assert_close(captured_inputs[0], expected_input)

    assert model.NTR == 5
    for decoder in (model.U_signal, model.U_dT1, model.U_dT2):
        assert isinstance(decoder, torch.nn.Sequential)
        assert isinstance(decoder[-1], torch.nn.Unflatten)

    for output in outputs:
        assert output.shape == (2, 5, 2)
        complex_output = torch.view_as_complex(output.contiguous())
        assert complex_output.shape == (2, 5)


def test_forward_signal_matches_full_forward(
    direct_config: dict[str, Any], make_model: Callable[..., FCModel]
) -> None:
    model = make_model(5, direct_config).eval()
    x = torch.tensor([[0.8, 0.07, 1.0], [1.2, 0.1, 0.9]])

    with torch.no_grad():
        signal_only = model.forward_signal(x)
        full = model(x)

    assert signal_only.shape == (2, 5, 2)
    # Bitwise: forward_signal shares input_transform and the signal head
    # with forward, so the two paths may never diverge.
    assert torch.equal(signal_only, full[0])


def test_forward_coeffs_reproduce_the_signal_through_the_basis(
    derived_config: dict[str, Any], make_model: Callable[..., FCModel]
) -> None:
    # The low-rank MR-STAT operators apply the basis themselves, in image
    # space, so they consume the coefficients and the decoder matrix
    # separately. This pins the factorization they rely on: coefficients
    # times the decoder, unflattened, must be the signal.
    model = make_model(5, derived_config).eval()
    x = torch.tensor([[0.8, 0.07, 1.0], [1.2, 0.1, 0.9]])

    with torch.no_grad():
        coefficients = model.forward_coeffs(x)
        signal = model.forward_signal(x)

    assert coefficients.shape == (2, derived_config["NB_signal"])
    decoder = model.U_signal[0].weight
    assert decoder.shape == (2 * model.NTR, derived_config["NB_signal"])
    reconstructed = (coefficients @ decoder.T).unflatten(-1, (model.NTR, 2))
    torch.testing.assert_close(reconstructed, signal)


def test_forward_coeffs_reproduce_signal_in_direct_mode(
    direct_config: dict[str, Any], make_model: Callable[..., FCModel]
) -> None:
    # In direct mode, the derivative heads have their own bases, but the signal
    # still factors into coefficients and its signal decoder.
    model = make_model(5, direct_config).eval()
    x = torch.tensor([[0.8, 0.07, 1.0], [1.2, 0.1, 0.9]])

    with torch.no_grad():
        coefficients = model.forward_coeffs(x)
        signal = model.forward_signal(x)

    reconstructed = (coefficients @ model.U_signal[0].weight.T).unflatten(
        -1, (model.NTR, 2)
    )
    torch.testing.assert_close(reconstructed, signal)


def test_decoder_rows_interleave_real_and_imaginary_parts(
    derived_config: dict[str, Any], make_model: Callable[..., FCModel]
) -> None:
    # The Julia side rebuilds a complex (NTR, rank) basis from this real
    # (2*NTR, rank) matrix. Row 2t is the real part at time t and row 2t+1
    # the imaginary part -- NOT a [all real; all imag] block layout. Getting
    # this wrong is silent: the shapes still match and the numbers are junk.
    model = make_model(5, derived_config).eval()
    decoder = model.U_signal[0].weight.detach()
    basis = torch.complex(decoder[0::2, :], decoder[1::2, :])
    assert basis.shape == (model.NTR, derived_config["NB_signal"])

    x = torch.tensor([[0.8, 0.07, 1.0], [1.2, 0.1, 0.9]])
    with torch.no_grad():
        coefficients = model.forward_coeffs(x)
        signal = model.forward_signal(x)
    expected = torch.view_as_complex(signal.contiguous())
    torch.testing.assert_close(coefficients.to(basis.dtype) @ basis.T, expected)


def test_forward_sensitivity_matches_pytorch_jvp_in_physical_units(
    derived_config: dict[str, Any], make_model: Callable[..., FCModel]
) -> None:
    data_config = SAMPLING_CONFIG["data"]
    x = torch.tensor([[0.8, 0.07, 1.0], [1.2, 0.1, 0.9]])

    for activation in ("silu", "gelu", "tanh", "softplus"):
        config = {
            **derived_config,
            "activation": activation,
            "normalize_inputs": True,
        }
        model = make_model(5, config, data_config).eval()
        signal, dT1, dT2 = model(x)

        tangent_T1 = torch.zeros_like(x)
        tangent_T1[:, 0] = 1.0
        tangent_T2 = torch.zeros_like(x)
        tangent_T2[:, 1] = 1.0
        primal_T1, expected_dT1 = torch.func.jvp(
            model.forward_signal, (x,), (tangent_T1,)
        )
        primal_T2, expected_dT2 = torch.func.jvp(
            model.forward_signal, (x,), (tangent_T2,)
        )

        torch.testing.assert_close(signal, primal_T1)
        torch.testing.assert_close(signal, primal_T2)
        torch.testing.assert_close(dT1, expected_dT1, rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(dT2, expected_dT2, rtol=2e-5, atol=2e-6)


def test_derived_mode_uses_one_basis_and_no_derivative_networks(
    derived_config: dict[str, Any], make_model: Callable[..., FCModel]
) -> None:
    model = make_model(5, derived_config)

    assert model.U_dT1 is None
    assert model.U_dT2 is None
    assert model.subnet_dT1 is None
    assert model.subnet_dT2 is None


def test_forward_sensitivity_loss_backpropagates_to_all_parameters(
    derived_config: dict[str, Any], make_model: Callable[..., FCModel]
) -> None:
    model = make_model(5, derived_config)
    x = torch.tensor([[0.8, 0.07, 1.0], [1.2, 0.1, 0.9]])

    loss = sum(output.square().mean() for output in model(x))
    loss.backward()

    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.all(torch.isfinite(parameter.grad))


def test_input_transform_adds_no_checkpoint_state(
    direct_config: dict[str, Any], make_model: Callable[..., FCModel]
) -> None:
    model = make_model(5, direct_config)
    assert len(list(model.input_transform.parameters())) == 0
    assert not any(key.startswith("input_transform") for key in model.state_dict())


def test_model_output_contract_supports_multiple_batch_sizes(
    direct_config: dict[str, Any], make_model: Callable[..., FCModel]
) -> None:
    model = make_model(5, direct_config).eval()

    for batch_size in (1, 3, 8):
        x = torch.tensor([[0.8, 0.07, 1.0]]).repeat(batch_size, 1)
        with torch.no_grad():
            outputs = model(x)
        for output in outputs:
            assert output.shape == (batch_size, 5, 2)


def test_complex_conversion_pairs_real_and_imaginary_values() -> None:
    values = np.array(
        [[1 + 2j, 3 + 4j], [5 + 6j, 7 + 8j]],
        dtype=np.complex64,
    )

    paired = complex_to_real_imag(values)

    assert paired.dtype == np.float32
    assert paired.shape == (2, 2, 2)
    np.testing.assert_array_equal(paired[..., 0], values.real)
    np.testing.assert_array_equal(paired[..., 1], values.imag)
    restored = torch.view_as_complex(torch.from_numpy(paired).contiguous())
    np.testing.assert_array_equal(restored.numpy(), values)


def test_h5_loader_preserves_physical_parameters_and_target_shape(
    tmp_path: Path,
) -> None:
    T1 = np.array([0.8, 1.2], dtype=np.float32)
    T2 = np.array([0.07, 0.1], dtype=np.float32)
    B1 = np.array([0.9, 1.1], dtype=np.float32)
    signal = np.array(
        [[1 + 2j, 3 + 4j], [5 + 6j, 7 + 8j]],
        dtype=np.complex64,
    )

    path = tmp_path / "data.h5"
    with h5py.File(path, "w") as file:
        file.create_dataset("T1", data=T1)
        file.create_dataset("T2", data=T2)
        file.create_dataset("B1", data=B1)
        file.create_dataset("signal", data=signal)
        file.create_dataset("deriv_T1", data=signal)
        file.create_dataset("deriv_T2", data=signal)

    params, loaded_signal, dT1, dT2, n_samples, NTR = load_data_from_h5_file(str(path))

    np.testing.assert_array_equal(params, np.stack((T1, T2, B1), axis=1))
    assert n_samples == 2
    assert NTR == 2

    paired = complex_to_real_imag(loaded_signal)
    loader = make_loader(
        params,
        paired,
        complex_to_real_imag(dT1),
        complex_to_real_imag(dT2),
        batchsize=2,
        device="cpu",
        shuffle=False,
    )
    x, sig, dT1_batch, dT2_batch = next(iter(loader))
    assert x.shape == (2, 3)
    assert sig.shape == (2, 2, 2)
    assert dT1_batch.shape == (2, 2, 2)
    assert dT2_batch.shape == (2, 2, 2)


def test_parameter_sampling_is_physical_and_log_uniform() -> None:
    torch.manual_seed(42)
    samples = sample_parameters(SAMPLING_CONFIG, 20_000)

    assert samples.shape == (20_000, 3)
    for column, lower, upper in (
        (0, 0.1, 5.0),
        (1, 0.01, 2.0),
        (2, 0.8, 1.2),
    ):
        assert torch.all(samples[:, column] >= lower)
        assert torch.all(samples[:, column] <= upper)

    expected_log_T1_mean = (math.log(0.1) + math.log(5.0)) / 2
    expected_log_T2_mean = (math.log(0.01) + math.log(2.0)) / 2
    assert torch.log(samples[:, 0]).mean().item() == pytest.approx(
        expected_log_T1_mean, abs=0.03
    )
    assert torch.log(samples[:, 1]).mean().item() == pytest.approx(
        expected_log_T2_mean, abs=0.03
    )


def test_metrics_reduce_over_all_non_batch_dimensions() -> None:
    true = np.array(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[2.0, 1.0], [4.0, 3.0]],
        ],
        dtype=np.float32,
    )
    pred = true + np.array(
        [
            [[0.5, -0.5], [1.0, -1.0]],
            [[-0.25, 0.25], [-0.5, 0.5]],
        ],
        dtype=np.float32,
    )

    flat_true = true.reshape(2, -1)
    flat_pred = pred.reshape(2, -1)
    np.testing.assert_allclose(nrmse(pred, true), nrmse(flat_pred, flat_true))

    torch_true = torch.from_numpy(true)
    torch_pred = torch.from_numpy(pred)
    flat_loss = nmae_loss(
        torch_pred.reshape(2, -1),
        torch_true.reshape(2, -1),
    )
    torch.testing.assert_close(nmae_loss(torch_pred, torch_true), flat_loss)


def test_metrics_handle_perfect_predictions_and_zero_targets() -> None:
    target = torch.zeros((2, 3, 2))
    assert nmae_loss(target, target).item() == 0.0
    assert torch.isfinite(nmae_loss(torch.ones_like(target), target))

    target_np = np.zeros((2, 3, 2), dtype=np.float32)
    assert np.all(nrmse(target_np, target_np) == 0.0)
    assert np.all(np.isfinite(nrmse(np.ones_like(target_np), target_np)))
