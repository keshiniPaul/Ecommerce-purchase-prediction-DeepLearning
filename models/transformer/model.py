"""Transformer Encoder model for session purchase prediction."""

from tensorflow.keras import layers, Model, optimizers

from models.transformer.config import (
    MAX_SEQUENCE_LENGTH,
    MODEL_DIM,
    NUM_HEADS,
    FF_DIM,
    NUM_TRANSFORMER_BLOCKS,
    DROPOUT_RATE,
    LEARNING_RATE,
)

from models.transformer.layers import (
    SessionEmbedding,
    TransformerEncoderBlock,
    MaskedGlobalAveragePooling1D,
)


def build_transformer_model(item_vocab_size):
    """
    Build and compile the Transformer Encoder classifier.
    """

    event_input = layers.Input(
        shape=(MAX_SEQUENCE_LENGTH,),
        dtype="int32",
        name="event_input",
    )

    item_input = layers.Input(
        shape=(MAX_SEQUENCE_LENGTH,),
        dtype="int32",
        name="item_input",
    )

    embedding_layer = SessionEmbedding(
        item_vocab_size=item_vocab_size,
        max_length=MAX_SEQUENCE_LENGTH,
        model_dim=MODEL_DIM,
        name="session_embedding",
    )

    x = embedding_layer(
        [event_input, item_input]
    )

    mask = embedding_layer.compute_mask(
        [event_input, item_input]
    )

    for block_index in range(
        NUM_TRANSFORMER_BLOCKS
    ):
        x = TransformerEncoderBlock(
            model_dim=MODEL_DIM,
            num_heads=NUM_HEADS,
            ff_dim=FF_DIM,
            dropout_rate=DROPOUT_RATE,
            name=f"transformer_block_{block_index + 1}",
        )(
            x,
            mask=mask,
        )

    x = MaskedGlobalAveragePooling1D(
        name="masked_global_average_pooling"
    )(
        x,
        mask=mask,
    )

    x = layers.Dropout(
        DROPOUT_RATE,
        name="classification_dropout",
    )(x)

    x = layers.Dense(
        64,
        activation="relu",
        name="classification_dense",
    )(x)

    x = layers.Dropout(
        DROPOUT_RATE,
        name="classification_dropout_2",
    )(x)

    output = layers.Dense(
        1,
        activation="sigmoid",
        name="purchase_probability",
    )(x)

    model = Model(
        inputs={
            "event_input": event_input,
            "item_input": item_input,
        },
        outputs=output,
        name="retailrocket_transformer",
    )

    model.compile(
        optimizer=optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
        ],
    )

    return model