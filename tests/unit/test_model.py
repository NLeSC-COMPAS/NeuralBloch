from collections.abc import Callable
from typing import Any
import torch
from src.model import FCModel


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
    derived_config: dict[str, Any],
    data_config: dict[str, Any],
    make_model: Callable[..., FCModel],
) -> None:
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
