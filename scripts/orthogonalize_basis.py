"""Save an ordered shared-basis checkpoint and optional smaller-rank copies.

Run explicitly after training. The input checkpoint remains untouched.
"""

import argparse
from pathlib import Path
from typing import Any
import h5py
import numpy as np
import torch
from src.basis import orthogonalize_shared_basis
from src.basis import truncate_shared_basis
from src.basis import worst_relative_output_difference
from src.model import FCModel
from src.utils import get_device


def load_checkpoint(path: Path, device: str) -> tuple[FCModel, dict[str, Any]]:
    """Load a shared-basis model and its checkpoint dictionary."""
    checkpoint: dict[str, Any] = torch.load(path, map_location=device)
    model = FCModel(
        checkpoint["NTR"], checkpoint["model_config"], checkpoint["data_config"]
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, checkpoint


def postprocess_basis(
    checkpoint_path: Path,
    fit_dataset_path: Path,
    output_path: Path,
    ranks: list[int],
    device: str,
) -> list[Path]:
    """Write an ordered full-rank checkpoint and requested truncated copies."""
    if checkpoint_path.resolve() == output_path.resolve():
        raise ValueError("--output must differ from --checkpoint")
    if any(rank <= 0 for rank in ranks) or len(set(ranks)) != len(ranks):
        raise ValueError("--truncate ranks must be positive and unique")
    rank_paths = [
        output_path.with_name(f"{output_path.stem}_r{rank}.pt") for rank in ranks
    ]
    if any(path.resolve() == checkpoint_path.resolve() for path in rank_paths):
        raise ValueError("A truncated output path would overwrite the input checkpoint")

    model, checkpoint = load_checkpoint(checkpoint_path, device)
    if model.derivative_mode != "forward":
        raise ValueError("Basis postprocessing requires derivative_mode='forward'")
    full_rank = checkpoint["model_config"]["NB_signal"]
    if any(rank > full_rank for rank in ranks):
        raise ValueError(f"--truncate ranks must not exceed full rank {full_rank}")

    with h5py.File(fit_dataset_path, "r") as dataset:
        params_np = np.stack(
            (dataset["T1"][:], dataset["T2"][:], dataset["B1"][:]), axis=1
        )
    params = torch.as_tensor(params_np, dtype=torch.float32, device=device)
    reference, _ = load_checkpoint(checkpoint_path, device)
    orthogonalize_shared_basis(model, params)
    worst_difference = worst_relative_output_difference(reference, model, params)
    print(f"Full-rank relative output difference: {worst_difference:.2e}")
    if worst_difference > 1e-4:
        raise RuntimeError("Ordered model diverged from the original beyond tolerance")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint["model_state"] = model.state_dict()
    torch.save(checkpoint, output_path)
    print(f"Saved ordered checkpoint to {output_path}")
    written = [output_path]

    for rank, rank_path in zip(ranks, rank_paths, strict=True):
        smaller = truncate_shared_basis(
            model, checkpoint["model_config"], checkpoint["data_config"], rank
        )
        rank_checkpoint = dict(checkpoint)
        rank_checkpoint["model_config"] = {
            **checkpoint["model_config"],
            "NB_signal": rank,
        }
        rank_checkpoint["model_state"] = smaller.state_dict()
        torch.save(rank_checkpoint, rank_path)
        written.append(rank_path)
        print(f"Saved rank-{rank} checkpoint to {rank_path}")

    return written


def parse_args() -> argparse.Namespace:
    """Parse the explicit post-training command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/model.pt"))
    parser.add_argument(
        "--fit-dataset",
        type=Path,
        default=Path("data/datasets/dataset_for_validation.h5"),
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--truncate",
        nargs="+",
        type=int,
        default=[],
        metavar="RANK",
        help="Also save checkpoints using the first RANK ordered basis vectors",
    )
    return parser.parse_args()


def main() -> None:
    """Run basis postprocessing from the command line."""
    args = parse_args()
    device = get_device({"experiment": {"DEVICE": "cuda"}})
    postprocess_basis(
        args.checkpoint, args.fit_dataset, args.output, args.truncate, device
    )


if __name__ == "__main__":
    main()
