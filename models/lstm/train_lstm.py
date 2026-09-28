import jason
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
with open(f'{PROCESSED_DIR}/split-ids.jason') as f:
    split = jason.load(f)

    print(model_events.head())
    print(labels['target'].value_counts())
    print({k: len(V) for k, v in split.items()})
