import math
from typing import Any
import numpy as np
import pytest
import torch
from neuralbloch.utils import complex_to_real_imag
from neuralbloch.utils import model_from_checkpoint
from neuralbloch.utils import sample_parameters


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


def test_parameter_sampling_is_physical_and_log_uniform(
    data_config: dict[str, Any],
) -> None:
    torch.manual_seed(42)
    samples = sample_parameters({"data": data_config}, 20_000)

    assert samples.shape == (20_000, 3)
    ranges = data_config["parameter_ranges"]
    for column, name in enumerate(("T1_s", "T2_s", "B1")):
        assert torch.all(samples[:, column] >= ranges[name]["lower"])
        assert torch.all(samples[:, column] <= ranges[name]["upper"])

    for column, name in enumerate(("T1_s", "T2_s")):
        lower = math.log(ranges[name]["lower"])
        upper = math.log(ranges[name]["upper"])
        assert torch.log(samples[:, column]).mean().item() == pytest.approx(
            (lower + upper) / 2, abs=0.03
        )


@pytest.mark.parametrize(
    ("model_config", "data_config"),
    [
        (
            {"derivative_mode": "direct", "NB_signal": 3},
            {"T1_lower": 0.1, "T1_upper": 6.0},
        ),
        # Only the model blocks are old
        (
            {"derivative_mode": "direct", "signal": {"rank": 3}},
            {"parameter_ranges": {}},
        ),
    ],
    ids=["old", "partly-new"],
)
def test_a_checkpoint_with_the_old_config_keys_is_refused(
    model_config: dict[str, Any], data_config: dict[str, Any]
) -> None:
    checkpoint = {
        "NTR": 5,
        "model_config": model_config,
        "data_config": data_config,
        "model_state": {},
    }
    with pytest.raises(ValueError, match="Train the model again"):
        model_from_checkpoint(checkpoint, "cpu")
