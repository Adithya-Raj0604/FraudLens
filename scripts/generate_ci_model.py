"""
Generate a lightweight XGBoost model + threshold for test runs where the real
trained artifacts aren't available.

models/xgboost_fraud_realistic.json, models/threshold_realistic.txt, and
Dataset/features.parquet are all gitignored (see .gitignore) — a fresh clone
or CI checkout has none of them. This script fits a small model on a synthetic
dataset that encodes the same PaySim fraud signature the real pipeline learns
from (large balance depletion + elevated transaction velocity => higher fraud
probability), so /predict and /explain behave sensibly enough for the API and
UI test suites without needing the 470MB source dataset.

Run: python scripts/generate_ci_model.py
"""

from pathlib import Path

import numpy as np
import xgboost as xgb

rng = np.random.default_rng(42)

N_SAMPLES = 4000
FRAUD_RATE = 0.3


def make_row(fraud: bool) -> list[float]:
    step = rng.integers(1, 744)
    type_encoded = rng.choice([1, 2])  # TRANSFER=1, CASH_OUT=2
    # Amount is drawn from the same wide range for both classes so the model
    # can't just key off transaction size — it has to learn the balance
    # pattern instead, same as the real PaySim fraud signature.
    amount = rng.uniform(50, 50000)

    if fraud:
        old_org = amount + rng.uniform(0, amount * 0.05)  # spends ~everything it has
        new_org = rng.uniform(0, amount * 0.02)  # origin balance drained to ~0
        old_dest = rng.uniform(0, 2000)
        new_dest = old_dest + rng.uniform(-50, 50)  # destination NOT credited (red flag)
        vel_24 = rng.integers(1, 20)
    else:
        old_org = amount / rng.uniform(0.05, 0.4)  # amount is a modest fraction of balance
        new_org = old_org - amount  # balance properly reduced
        old_dest = rng.uniform(1000, 50000)
        new_dest = old_dest + amount  # destination properly credited
        vel_24 = rng.integers(1, 6)

    vel_1 = min(vel_24, rng.integers(1, 4))
    vel_3 = min(vel_24, vel_1 + rng.integers(0, 3))
    vel_cum = vel_24 + rng.integers(0, 20)
    log_amount = float(np.log1p(amount))

    return [
        step, type_encoded, amount, log_amount,
        old_org, new_org, old_dest, new_dest,
        vel_cum, vel_1, vel_3, vel_24,
    ]


def main() -> None:
    X, y = [], []
    for _ in range(N_SAMPLES):
        fraud = rng.random() < FRAUD_RATE
        X.append(make_row(fraud))
        y.append(int(fraud))

    X = np.array(X, dtype="float32")
    y = np.array(y, dtype="int32")

    model = xgb.XGBClassifier(n_estimators=50, max_depth=4, eval_metric="logloss")
    model.fit(X, y)

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    model.save_model(str(models_dir / "xgboost_fraud_realistic.json"))
    (models_dir / "threshold_realistic.txt").write_text("0.5")

    print("Wrote models/xgboost_fraud_realistic.json + threshold_realistic.txt (synthetic test fixture)")


if __name__ == "__main__":
    main()
