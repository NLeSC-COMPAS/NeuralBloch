import pytest
import torch
from data.data_loading import TensorBatchLoader


def test_a_loader_without_samples_is_refused() -> None:
    # Its epochs have no step, and their mean loss is 0 / 0
    tensors = (torch.zeros(0, 3), *(torch.zeros(0, 5, 2) for _ in range(3)))

    with pytest.raises(ValueError, match="at least one sample"):
        TensorBatchLoader(tensors, 4, shuffle=False)
