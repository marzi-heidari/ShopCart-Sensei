# sequence_eval.py

"""
Evaluation script for sequence-based recommendation models.
Supports Recall@K, NDCG@K, MAP@K and exports results to JSON/CSV.
Integrates with CLI for modular usage.
"""

import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from typing import Literal
import argparse
import json
import csv
import logging
from pathlib import Path

# === Logging ===
logging.basicConfig(
    format='[%(asctime)s] %(levelname)s - %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

# === Metrics ===
def recall_at_k(preds: torch.Tensor, targets: torch.Tensor, k: int) -> float:
    hits = (preds[:, :k] == targets.unsqueeze(1)).any(dim=1).float()
    return hits.mean().item()

def ndcg_at_k(preds: torch.Tensor, targets: torch.Tensor, k: int) -> float:
    dcg = 0.0
    for i in range(len(preds)):
        top_k = preds[i, :k]
        if targets[i].item() in top_k:
            rank = (top_k == targets[i].item()).nonzero(as_tuple=True)[0].item() + 1
            dcg += 1.0 / np.log2(rank + 1)
    return dcg / len(preds)

def average_precision(preds: torch.Tensor, target: torch.Tensor, k: int) -> float:
    top_k = preds[:k]
    if target.item() in top_k:
        rank = (top_k == target.item()).nonzero(as_tuple=True)[0].item() + 1
        return 1.0 / rank
    return 0.0

def map_at_k(preds: torch.Tensor, targets: torch.Tensor, k: int) -> float:
    ap_sum = sum(average_precision(preds[i], targets[i], k) for i in range(len(preds)))
    return ap_sum / len(preds)


# === Evaluation Function ===
def evaluate(model: torch.nn.Module, dataloader: DataLoader, device: torch.device, k: int = 10) -> dict:
    """
    Evaluates the model using Recall@K, NDCG@K, and MAP@K metrics.

    Args:
        model: Trained model.
        dataloader: DataLoader for test set.
        device: torch.device.
        k: Top-k cutoff.

    Returns:
        dict: Evaluation metrics.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            topk = torch.topk(logits, k, dim=-1).indices  # [B, T, k]
            mask = y != 0
            preds_flat = topk[mask]
            targets_flat = y[mask]
            all_preds.append(preds_flat)
            all_targets.append(targets_flat)

    if not all_preds:
        logging.warning("No valid targets found for evaluation.")
        return {}

    preds_tensor = torch.cat(all_preds, dim=0)
    targets_tensor = torch.cat(all_targets, dim=0)

    metrics = {
        f"recall@{k}": round(recall_at_k(preds_tensor, targets_tensor, k), 4),
        f"ndcg@{k}": round(ndcg_at_k(preds_tensor, targets_tensor, k), 4),
        f"map@{k}": round(map_at_k(preds_tensor, targets_tensor, k), 4)
    }
    logging.info(f"Evaluation Results @ Top-{k}:")
    for name, value in metrics.items():
        logging.info(f"{name.upper()}: {value:.4f}")
    return metrics


# === CLI Interface ===
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--model-name", type=str, choices=["sasrec", "bert4rec"], required=True)
    parser.add_argument("--save-metrics", type=str, help="Optional path to save metrics (JSON or CSV)")
    parser.add_argument("--top-k", type=int, default=10, help="Top-k cutoff")
    args = parser.parse_args()

    from train_sequence_models import SequenceDataset, collate_fn, get_model, DATA_PATH, BATCH_SIZE

    df = pd.read_parquet(DATA_PATH)
    dataset = SequenceDataset(df)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_model(args.model_name, num_items=len(dataset.item_set))
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)

    metrics = evaluate(model, dataloader, device=device, k=args.top_k)

    if args.save_metrics:
        out_path = Path(args.save_metrics)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.suffix == ".json":
            with open(out_path, "w") as f:
                json.dump(metrics, f, indent=2)
        elif out_path.suffix == ".csv":
            with open(out_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["metric", "value"])
                for k, v in metrics.items():
                    writer.writerow([k, v])
        else:
            raise ValueError("Unsupported file format. Use .json or .csv")
        logging.info(f"Metrics saved to {out_path}")


if __name__ == "__main__":
    main()
