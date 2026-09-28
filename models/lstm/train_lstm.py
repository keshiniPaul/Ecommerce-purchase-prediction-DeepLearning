import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from tensorflow import keras
from sklearn.utils.class_weight import compute_class_weight

from common.config import SEED, PROCESSED_DIR
from common.utils import set_seed, Timer, count_params
from common.metrics import compute_all_metrics, metrics_to_dataframe
from common.evaluation import (
    plot_learning_curves,
    plot_roc_curve,
    plot_confusion_matrix,
)

from models.lstm.sequence_dataset import build_sequences, MAX_LEN
from models.lstm.lstm_model import build_lstm_model

set_seed(SEED)

# ----- 1. Load shared processed data -----
model_events = pd.read_parquet(f"{PROCESSED_DIR}/model_events.parquet")
labels = pd.read_csv(f"{PROCESSED_DIR}/labels.csv")

with open(f"{PROCESSED_DIR}/split_ids.json") as f:
    split = json.load(f)

print(model_events.head())
print(labels["target"].value_counts())
print({k: len(v) for k, v in split.items()})

# ----- 2. Leak check: train / val / test must not overlap -----
train_ids = set(split["train"])
val_ids = set(split["val"])
test_ids = set(split["test"])
assert train_ids.isdisjoint(val_ids), "train and val overlap!"
assert train_ids.isdisjoint(test_ids), "train and test overlap!"
assert val_ids.isdisjoint(test_ids), "val and test overlap!"
print("OK: train / val / test sessions are separate")
print("Sizes:", len(train_ids), len(val_ids), len(test_ids))

# ----- 3. Sequences (test NOT used in training) -----
X_train, y_train = build_sequences(model_events, labels, split["train"])
X_val, y_val = build_sequences(model_events, labels, split["val"])
X_test, y_test = build_sequences(model_events, labels, split["test"])

# ----- 4. Class weights from TRAIN only -----
classes = np.array([0, 1])
weights = compute_class_weight("balanced", classes=classes, y=y_train)
class_weight = {0: float(weights[0]), 1: float(weights[1])}
print("class_weight:", class_weight)

# ----- 5. Unique folder for THIS run -----
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir = Path("models/lstm/runs") / run_id
ckpt_dir = run_dir / "checkpoints"
out_dir = run_dir / "outputs"
ckpt_dir.mkdir(parents=True, exist_ok=True)
out_dir.mkdir(parents=True, exist_ok=True)
print("Run ID:", run_id)
print("Saving to:", run_dir.resolve())

# ----- 6. Model + callbacks (paths use this run) -----
model = build_lstm_model(max_len=MAX_LEN)
model.summary()

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_auc", mode="max", patience=3, restore_best_weights=True
    ),
    keras.callbacks.ModelCheckpoint(
        str(ckpt_dir / "best_lstm.keras"),
        monitor="val_auc", mode="max", save_best_only=True,
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6
    ),
]

# ----- 7. Train on train only; validate on val only -----
with Timer() as t_train:
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=15,
        batch_size=256,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )
hist = {k: list(map(float, v)) for k, v in history.history.items()}

# ----- 8. Test once; save under this run_id -----
with Timer() as t_infer:
    y_prob = model.predict(X_test, batch_size=512).ravel()
y_pred = (y_prob >= 0.5).astype(int)

metrics = compute_all_metrics(y_test, y_pred, y_prob)
row = metrics_to_dataframe(metrics, model_name="LSTM")
row["params"] = count_params(model)
row["train_time_sec"] = t_train.elapsed
row["infer_time_sec"] = t_infer.elapsed
row["run_id"] = run_id
row.to_csv(out_dir / "lstm_test_metrics.csv", index=False)

plot_learning_curves(hist, "LSTM", str(out_dir))
plot_roc_curve(y_test, y_prob, "LSTM", str(out_dir))
plot_confusion_matrix(metrics["confusion_matrix"], "LSTM", str(out_dir))

# ----- 9. Record hyperparameters for this run -----
run_meta = {
    "run_id": run_id,
    "max_len": int(MAX_LEN),
    "embed_dim": 16,
    "lstm_units": 32,
    "dropout": 0.3,
    "batch_size": 256,
    "epochs_max": 15,
    "class_weight": "balanced",
    "n_train": int(len(y_train)),
    "n_val": int(len(y_val)),
    "n_test": int(len(y_test)),
    "accuracy": float(metrics["accuracy"]),
    "precision": float(metrics["precision"]),
    "recall": float(metrics["recall"]),
    "f1": float(metrics["f1"]),
    "roc_auc": float(metrics["roc_auc"]),
}
with open(run_dir / "run_config.json", "w") as f:
    json.dump(run_meta, f, indent=2)

print(metrics)
print("Saved run →", run_dir)
