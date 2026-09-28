from tensorflow import keras
from tensorflow.keras import layers


def build_lstm_model(max_len=20, vocab_size=3, embed_dim=16, lstm_units=32, dropout=0.3):
    inp = keras.Input(shape=(max_len,), name='event_seq')
    x = layers.Embedding(vocab_size, embed_dim,
                         mask_zero=True, name='event_embedding')(inp)

    x = layers.LSTM(lstm_units, name='lstm')(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(16, activation='relu')(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(1, activation='sigmoid', name='purchase_prob')(x)

    model = keras.Model(inp, out, name='SessionLSTM')
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss='binary_crossentropy',
        metrics=[
            keras.metrics.AUC(name='auc'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
        ],
    )

    return model
