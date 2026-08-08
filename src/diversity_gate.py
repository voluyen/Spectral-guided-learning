"""Step-diversity non-degeneracy + feasibility gate reporter (read-only, CPU).

Separate from the paper-reproduction `capture` stage: it recomputes nothing, it just reads the
spectral-strengths parquet that capture already wrote and reports the distributions of the two
step-diversity signals plus a PASS / DEGENERATE verdict, so the diversity experiment's gate is
inspectable on its own.

    python src/diversity_gate.py --config configs/capture-config.yaml

- Div  = Sigma-weighted residual energy fraction (energy outside the top-k* consensus subspace).
- Nov  = inverse-participation-ratio novelty (effective number of SVD directions a step spreads
         across); the signal that stays graded when T >> d.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def _flatten(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame:
        return np.array([])
    rows = [np.asarray(row, dtype=float) for row in frame[column] if len(row)]
    return np.concatenate(rows) if rows else np.array([])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/capture-config.yaml")
    parser.add_argument(
        "--std-threshold",
        type=float,
        default=0.005,
        help="minimum Div std below which the metric is judged degenerate",
    )
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    frame = pd.read_parquet(config["strengths_path"])
    div = _flatten(frame, "step_diversity")
    nov = _flatten(frame, "step_novelty")

    if div.size == 0:
        print("no step_diversity in parquet -- rerun `capture` with the current code first")
        return

    dq = np.quantile(div, [0.1, 0.5, 0.9])
    print(f"samples={len(frame)}  steps={div.size}")
    print(
        f"Div (residual frac): mean={div.mean():.4f} std={div.std():.4f} "
        f"p10={dq[0]:.4f} median={dq[1]:.4f} p90={dq[2]:.4f}"
    )
    if nov.size:
        nq = np.quantile(nov, [0.1, 0.5, 0.9])
        print(
            f"IPR novelty (eff #modes): mean={nov.mean():.2f} std={nov.std():.2f} "
            f"p10={nq[0]:.2f} median={nq[1]:.2f} p90={nq[2]:.2f} max={nov.max():.2f}"
        )

    degenerate = div.std() < args.std_threshold
    verdict = "DEGENERATE -- reformulate metric" if degenerate else "PASS"
    print(
        f"\nNON-DEGENERACY GATE: {verdict}  "
        f"(Div std {div.std():.4f} vs threshold {args.std_threshold})"
    )


if __name__ == "__main__":
    main()
