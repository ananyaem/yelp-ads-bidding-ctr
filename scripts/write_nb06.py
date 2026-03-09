import json
import uuid
from pathlib import Path

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12.0"},
    },
    "cells": [],
}


def _as_nb_source(s: str) -> list[str]:
    if not s:
        return []
    return [line + "\n" for line in s.splitlines()]


def md(s: str) -> dict:
    return {"cell_type": "markdown", "id": str(uuid.uuid4()), "metadata": {}, "source": _as_nb_source(s)}


def code(s: str) -> dict:
    return {
        "cell_type": "code",
        "id": str(uuid.uuid4()),
        "metadata": {},
        "source": _as_nb_source(s),
        "outputs": [],
        "execution_count": None,
    }


cells: list[dict] = []
cells.append(
    md(
        """# Resume claims verification

This notebook recomputes or loads the metrics behind common resume bullets. Run with the project venv; set the kernel working directory to `notebooks/` or the repo root (the first code cell fixes `PROJECT_ROOT`).

| Claim | Reproduction |
|-------|----------------|
| Val AUC ≥ 0.78 | Max `val_auc` in `models/history_default.json` |
| ECE < 0.03 after Platt | `models/deepfm_calib.pt` + `models/platt_scaler.pkl` on the test split |
| ~23% revenue lift | Same simulation as `05_auction_analysis.ipynb` section 3 |
| DeepFM ~+0.03 AUC vs LR | `models/resume_metrics_summary.json` (values from `04_deepfm_training.ipynb` §3) |

**Environment:** Pin `numpy<2` with current PyTorch wheels (`requirements.txt`)."""
    )
)

cells.append(
    code(
        """import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path.cwd()
if PROJECT_ROOT.name == "notebooks":
    PROJECT_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.auction.gsp import AuctionSimulator, GSPAuction
from src.config import AD_IMPRESSIONS_PARQUET_PATH, DEFAULT_HPARAMS, MODELS_DIR
from src.models.calibration import PlattScaler, compute_ece
from src.models.deepfm import DeepFM
from src.training.trainer import AdClickDataset, Trainer, TrainerConfig, _collate_batch

print("PROJECT_ROOT", PROJECT_ROOT)"""
    )
)

cells.append(md("## 1 · Validation AUC ≥ 0.78 (default training history)"))
cells.append(
    code(
        """hist_path = MODELS_DIR / "history_default.json"
history = json.loads(hist_path.read_text(encoding="utf-8"))
best_val_auc = max(float(row["val_auc"]) for row in history)
print(f"Best val_auc from {hist_path.name}: {best_val_auc:.5f}")
assert best_val_auc >= 0.78
print("PASS")"""
    )
)

cells.append(md("## 2 · ECE < 0.03 after Platt (calib checkpoint + saved scaler)"))
cells.append(
    code(
        """df = pd.read_parquet(AD_IMPRESSIONS_PARQUET_PATH)
test_df = df[df["split"] == "test"].reset_index(drop=True)

ckpt = torch.load(MODELS_DIR / "deepfm_calib.pt", map_location="cpu", weights_only=False)
fc = ckpt["feature_config"]
model = DeepFM(
    feature_config=fc,
    dnn_layers=list(DEFAULT_HPARAMS.dnn_layers),
    dropout=DEFAULT_HPARAMS.dropout,
)
model.load_state_dict(ckpt["model_state_dict"])
trainer = Trainer(
    model=model,
    feature_config=fc,
    config=TrainerConfig(batch_size=2048, epochs=1, patience=1),
    device="cpu",
    checkpoint_path=MODELS_DIR / "_tmp_verify.pt",
)
trainer.platt_scaler = PlattScaler.load(MODELS_DIR / "platt_scaler.pkl")

test_loader = DataLoader(
    AdClickDataset(test_df, fc),
    batch_size=2048,
    shuffle=False,
    collate_fn=_collate_batch,
)
_, y_test, raw_probs = trainer._epoch_pass(test_loader, train=False)
cal_probs = trainer.platt_scaler.calibrate(raw_probs)
ece_after = compute_ece(y_test, cal_probs, n_bins=10)
print(f"ECE (10 bins, after Platt, test): {ece_after.ece:.5f}")
assert ece_after.ece < 0.03
print("PASS")"""
    )
)

cells.append(md("## 3 · Revenue lift ~23% (quality GSP vs random ranking)"))
cells.append(
    code(
        """N_CAMPAIGNS = 20
N_REQUESTS = 5000
rng = np.random.default_rng(42)
campaign_profiles = {}
for i in range(N_CAMPAIGNS):
    campaign_profiles[f"c{i}"] = {
        "base_bid": float(rng.uniform(0.5, 6.0)),
        "ctr_mean": float(rng.uniform(0.03, 0.20)),
        "ctr_std": float(rng.uniform(0.005, 0.02)),
    }

def build_requests(rng, n_requests, profiles):
    requests = []
    cids = list(profiles.keys())
    for _ in range(n_requests):
        k = int(rng.integers(3, 9))
        chosen = rng.choice(cids, size=k, replace=False)
        candidates = []
        for cid in chosen:
            p = profiles[cid]
            ctr = float(np.clip(rng.normal(p["ctr_mean"], p["ctr_std"]), 0.005, 0.50))
            bid = float(np.clip(p["base_bid"] * rng.uniform(0.8, 1.2), 0.05, 10.0))
            candidates.append({
                "restaurant_id": f"{cid}_r",
                "bid": round(bid, 4),
                "predicted_ctr": round(ctr, 6),
                "campaign_id": cid,
            })
        requests.append(candidates)
    return requests

class RandomAuction(GSPAuction):
    def rank_ads(self, candidates):
        valid = [c for c in candidates if c.get("predicted_ctr", 0) > 0]
        self._rng.shuffle(valid)
        for ad in valid:
            ad["rank_score"] = ad["bid"] * ad["predicted_ctr"]
        return valid

lifts = []
for s in range(50):
    local_rng = np.random.default_rng(s)
    reqs = build_requests(local_rng, N_REQUESTS, campaign_profiles)
    ra = RandomAuction(reserve_price=0.10, epsilon=0.01, max_slots=3)
    ra._rng = np.random.default_rng(s)
    rand_sim = AuctionSimulator(auction=ra, seed=s)
    rand_agg = rand_sim.compute_aggregate_metrics(rand_sim.simulate(reqs))
    qa = GSPAuction(reserve_price=0.10, epsilon=0.01, max_slots=3)
    qual_sim = AuctionSimulator(auction=qa, seed=s)
    qual_agg = qual_sim.compute_aggregate_metrics(qual_sim.simulate(reqs))
    if rand_agg["total_revenue"] > 0:
        lifts.append((qual_agg["total_revenue"] - rand_agg["total_revenue"]) / rand_agg["total_revenue"])

mean_lift = float(np.mean(lifts))
print(f"Mean revenue lift (quality vs random): {mean_lift:+.1%}")
assert mean_lift >= 0.15
assert 0.18 <= mean_lift <= 0.35, f"Expected lift band ~20–30% (~23% claim); got {mean_lift:.1%}"
print("PASS")"""
    )
)

cells.append(
    md(
        "## 4 · DeepFM vs logistic ~+0.03 AUC\n\n"
        "Frozen numbers from the executed comparison in `notebooks/04_deepfm_training.ipynb` (section 3). "
        "Re-run that notebook to regenerate `models/resume_metrics_summary.json` if baselines change."
    )
)
cells.append(
    code(
        """summary_path = MODELS_DIR / "resume_metrics_summary.json"
summary = json.loads(summary_path.read_text(encoding="utf-8"))
delta = float(summary["deepfm_minus_lr_auc"])
print(json.dumps(summary, indent=2))
assert delta >= 0.02
print("PASS")"""
    )
)

nb["cells"] = cells
out = Path(__file__).resolve().parents[1] / "notebooks" / "06_resume_claims_verification.ipynb"
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("Wrote", out)
