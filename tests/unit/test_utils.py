import math
from typing import Any
import numpy as np
import pytest
import torch
from src.utils import complex_to_real_imag
from src.utils import sample_parameters


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
    for column, name in enumerate(("T1", "T2", "B1")):
        assert torch.all(samples[:, column] >= data_config[f"{name}_lower"])
        assert torch.all(samples[:, column] <= data_config[f"{name}_upper"])

    for column, name in enumerate(("T1", "T2")):
        lower = math.log(data_config[f"{name}_lower"])
        upper = math.log(data_config[f"{name}_upper"])
        assert torch.log(samples[:, column]).mean().item() == pytest.approx(
            (lower + upper) / 2, abs=0.03
        )
