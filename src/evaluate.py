"""Evaluation metrics for MeshLex validation experiment."""
import torch
import numpy as np
from torch_geometric.loader import DataLoader
from collections import Counter


@torch.no_grad()
def evaluate_reconstruction(model, dataset, device="cuda", batch_size=256):
    """Compute reconstruction CD and codebook utilization on a dataset.

    Returns:
        dict with keys:
          - mean_cd: mean Chamfer Distance (x 10^3)
          - std_cd: std of per-batch CD
          - utilization: fraction of codebook used
          - code_histogram: Counter of code usage
          - n_unique_codes: number of unique codes used
          - total_codes: codebook size K
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_cds = []
    all_indices = []

    for batch in loader:
        batch = batch.to(device)
        result = model(
            batch.x, batch.edge_index, batch.batch,
            batch.n_vertices, batch.gt_vertices,
        )

        all_cds.append(result["recon_loss"].item())
        all_indices.append(result["indices"].cpu())

    all_idx = torch.cat(all_indices)
    code_counts = Counter(all_idx.numpy().tolist())

    utilization = len(code_counts) / model.codebook.K
    mean_cd = np.mean(all_cds) * 1000  # x 10^3

    return {
        "mean_cd": mean_cd,
        "std_cd": np.std(all_cds) * 1000,
        "utilization": utilization,
        "code_histogram": code_counts,
        "n_unique_codes": len(code_counts),
        "total_codes": model.codebook.K,
    }


def compute_go_nogo(same_cat_cd: float, cross_cat_cd: float, utilization: float = None):
    """Apply Go/No-Go decision matrix with utilization gate.

    Args:
        same_cat_cd: mean CD on same-category test set
        cross_cat_cd: mean CD on cross-category test set
        utilization: fraction of codebook used (0.0 to 1.0)

    Returns:
        dict with ratio, decision, next_step, utilization
    """
    if same_cat_cd < 1e-10:
        return {"ratio": float("inf"), "decision": "ERROR", "next_step": "CD is zero — check data"}

    ratio = cross_cat_cd / same_cat_cd

    # Utilization gate — collapse detection
    if utilization is not None and utilization < 0.10:
        return {
            "ratio": ratio,
            "decision": "COLLAPSE - HALT",
            "next_step": "Codebook collapsed (<10% utilization). Debug VQ training before proceeding.",
            "same_cat_cd": same_cat_cd,
            "cross_cat_cd": cross_cat_cd,
            "utilization": utilization,
        }

    if ratio < 1.2 and (utilization is None or utilization > 0.30):
        decision = "STRONG GO"
        next_step = "Proceed to full MeshLex experiment design"
    elif ratio < 1.2:
        decision = "CONDITIONAL GO"
        next_step = "CD ratio good but utilization suboptimal. Consider decoder enhancement (Phase B)."
    elif ratio < 2.0:
        decision = "WEAK GO"
        next_step = "Adjust story to 'transferable vocabulary', continue"
    elif ratio < 3.0:
        decision = "HOLD"
        next_step = "Analyze failure, consider category-adaptive codebook"
    else:
        decision = "NO-GO"
        next_step = "Core hypothesis falsified. Pivot direction."

    return {
        "ratio": ratio,
        "decision": decision,
        "next_step": next_step,
        "same_cat_cd": same_cat_cd,
        "cross_cat_cd": cross_cat_cd,
        "utilization": utilization,
    }
