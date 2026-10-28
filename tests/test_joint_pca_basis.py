import torch
from data.data_loading import TensorBatchLoader
from scripts.train import initialize_joint_pca_basis
from src.model import FCModel


def test_initial_basis_matches_best_rank_limited_svd_subspace() -> None:
    torch.manual_seed(42)
    n_samples = 12
    n_tr = 4
    rank = 2
    config = {
        "activation": "gelu",
        "derivative_mode": "derived",
        "normalize_inputs": False,
        "NB_signal": rank,
        "hidden_size_signal": 8,
        "num_layers_signal": 2,
    }
    model = FCModel(n_tr, config)
    parameters = torch.ones(n_samples, 3)
    targets = tuple(torch.randn(n_samples, n_tr, 2) for _ in range(3))
    loader = TensorBatchLoader((parameters, *targets), 4, shuffle=False)

    initialize_joint_pca_basis(model, loader)

    basis = model.U_signal[0].weight.detach()
    normalized_rows = torch.cat(
        tuple(
            rows.flatten(start_dim=1)
            / torch.linalg.vector_norm(
                rows.flatten(start_dim=1), dim=1, keepdim=True
            ).clamp_min(1e-4)
            for rows in targets
        )
    )
    _, _, right_vectors = torch.linalg.svd(normalized_rows, full_matrices=False)
    optimal_basis = right_vectors[:rank].T

    torch.testing.assert_close(basis.T @ basis, torch.eye(rank), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(
        basis @ basis.T,
        optimal_basis @ optimal_basis.T,
        atol=1e-5,
        rtol=1e-5,
    )
    assert model.U_signal[0].weight.requires_grad
