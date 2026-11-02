from collections.abc import Callable
import pytest
import torch
from neuralbloch.loss import Loss
from neuralbloch.loss import hybrid_relative_loss
from neuralbloch.loss import nmae_loss
from neuralbloch.loss import relative_l2_loss
from neuralbloch.loss import relative_l2_squared_loss


LossFunction = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

# The functions of Loss by their name (training.LOSS_FUNCTION)
FUNCTIONS: dict[str, LossFunction] = {
    "nmae": nmae_loss,
    "relative_l2": relative_l2_loss,
    "relative_l2_squared": relative_l2_squared_loss,
    "hybrid_relative": hybrid_relative_loss,
}

each_function = pytest.mark.parametrize(
    "function", FUNCTIONS.values(), ids=FUNCTIONS.keys()
)


@pytest.fixture
def target() -> torch.Tensor:
    """The targets of 4 samples, NTR = 5, each of another size."""
    torch.manual_seed(0)
    return torch.randn(4, 5, 2) * torch.tensor([1.0, 10.0, 0.1, 3.0])[:, None, None]


@pytest.fixture
def pred(target: torch.Tensor) -> torch.Tensor:
    """A prediction of target, with errors that are not proportional to it."""
    return target + 0.3 * torch.randn_like(target) * target.abs().mean()


@each_function
def test_the_target_gives_0_and_a_zero_prediction_gives_1(
    function: LossFunction, target: torch.Tensor
) -> None:
    assert function(target, target).item() == 0.0
    assert function(torch.zeros_like(target), target).item() == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("function", "expected"),
    [
        (nmae_loss, 2 / 3.5),
        (relative_l2_loss, 4 / 5),
        (relative_l2_squared_loss, 16 / 25),
        # The mean of nmae and relative_l2
        (hybrid_relative_loss, (2 / 3.5 + 4 / 5) / 2),
    ],
    ids=FUNCTIONS.keys(),
)
def test_the_loss_of_one_sample_has_the_value_computed_by_hand(
    function: LossFunction, expected: float
) -> None:
    # The error [0, 4] is not proportional to the target [3, 4], so the four
    # functions give four values: the mean absolute value of the error over
    # that of the target, the norm of the error over that of the target, the
    # square of this, and the mean of the first two
    loss = function(torch.tensor([[3.0, 0.0]]), torch.tensor([[3.0, 4.0]]))

    assert loss.item() == pytest.approx(expected)


@each_function
def test_each_sample_is_normalized_by_its_own_target(
    function: LossFunction, target: torch.Tensor, pred: torch.Tensor
) -> None:
    # A power of 2, so that the products are exact
    scale = torch.tensor([1024.0, 1.0, 1.0, 1.0])[:, None, None]

    assert function(pred * scale, target * scale) == function(pred, target)


@each_function
def test_each_sample_is_normalized_over_all_its_axes(
    function: LossFunction, target: torch.Tensor, pred: torch.Tensor
) -> None:
    # The targets are (N, NTR, 2). A normalization of each readout or of each
    # channel gives another loss, when the two channels have other sizes
    scale = torch.tensor([1.0, 0.01])
    target, pred = target * scale, pred * scale

    torch.testing.assert_close(
        function(pred, target),
        function(pred.flatten(start_dim=1), target.flatten(start_dim=1)),
    )


@each_function
def test_the_loss_of_a_batch_is_the_mean_of_the_losses_of_its_samples(
    function: LossFunction, target: torch.Tensor, pred: torch.Tensor
) -> None:
    losses = [function(pred[i : i + 1], target[i : i + 1]) for i in range(4)]

    torch.testing.assert_close(function(pred, target), torch.stack(losses).mean())


@each_function
def test_a_zero_target_gives_a_finite_loss(function: LossFunction) -> None:
    target = torch.zeros(2, 5, 2)

    assert torch.isfinite(function(torch.ones_like(target), target))


@pytest.mark.parametrize("name", FUNCTIONS)
def test_loss_gives_the_terms_of_its_function_and_their_mean(name: str) -> None:
    # The signal, dT1 and dT2, each with errors of another size
    torch.manual_seed(1)
    targets = torch.randn(3, 4, 5, 2)
    errors = torch.tensor([0.1, 0.5, 2.0])[:, None, None, None]
    preds = targets + errors * torch.randn_like(targets)

    total, *terms = Loss(name)(*preds, *targets)

    expected = [FUNCTIONS[name](p, t) for p, t in zip(preds, targets, strict=True)]
    torch.testing.assert_close(terms, expected)
    torch.testing.assert_close(total, torch.stack(expected).mean())
