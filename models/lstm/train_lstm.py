from tensorflow import keras
from sklearn.utils.class_weight import compute_class_weight
import json
import numpy as np
import pandas as pd
from pathlib import Path

from common.config import SEED, PROCESSED_DIR
from common.utils import set_seed, Timer, count_params
from common.metrics import compute_all_metrics, metrics_to_dataframe
from common.evaluation import (
    plot_learning_curves, plot_roc_curve, plot_confusion_matrix
)

from models.lstm.sequence_dataset import build_sequence, MAX_LEN
from models.lstm.lstm_model import build_lstm_model

set_seed(SEED)

model_events = pd.read_parquet(f'{PROCESSED_DIR}/model_events.parquet ')
labels = pd.read_csv(f'{PROCESSED_DIR}/labels.csv')
with open(f'{PROCESSED_DIR}/split-ids.json') as f:
    split = jason.load(f)

    print(model_events.head())
    print(labels['target'].value_counts())
    print({k: len(V) for k, v in split.items()})


X_train, y_train = build_sequences(model_events, labels, split['train'])
X_val,   y_val = build_sequences(model_events, labels, split['val'])
X_test,  y_test = build_sequences(model_events, labels, split['test'])


classes = np.array([0, 1])
weights = compute_class_weight('balanced', classes=classes, y=y_train)
class_weight = {0: float(weights[0]), 1: float(weights[1])}
print('class_weight:', class_weight)


Path('models/lstm/checkpoints').mkdir(parents=True, exist_ok=True)
Path('models/lstm/outputs').mkdir(parents=True, exist_ok=True)
model = build_lstm_model(max_len=MAX_LEN)
model.summary()
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_auc', mode='max', patience=3, restore_best_weights=True),
    keras.callbacks.ModelCheckpoint(
        'models/lstm/checkpoints/best_lstm.keras',
        monitor='val_auc', mode='max', save_best_only=True),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6),
]
with Timer() as t_train:
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=15, batch_size=256,
        class_weight=class_weight,
        callbacks=callbacks, verbose=1,
    )
hist = {k: list(map(float, v)) for k, v in history.history.items()}
