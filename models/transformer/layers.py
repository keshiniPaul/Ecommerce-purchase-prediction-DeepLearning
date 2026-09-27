"""Custom layers for the Transformer Encoder."""

import tensorflow as tf
from tensorflow.keras import layers


class SessionEmbedding(layers.Layer):
    """
    Combines event, item and positional embeddings.
    """

    def __init__(
        self,
        item_vocab_size,
        max_length,
        model_dim,
        num_event_tokens=3,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.item_vocab_size = item_vocab_size
        self.max_length = max_length
        self.model_dim = model_dim
        self.num_event_tokens = num_event_tokens

        self.item_embedding = layers.Embedding(
            input_dim=item_vocab_size,
            output_dim=model_dim,
            mask_zero=True,
            name="item_embedding",
        )

        self.event_embedding = layers.Embedding(
            input_dim=num_event_tokens,
            output_dim=model_dim,
            mask_zero=True,
            name="event_embedding",
        )

        self.position_embedding = layers.Embedding(
            input_dim=max_length,
            output_dim=model_dim,
            name="position_embedding",
        )

    def call(self, inputs):
        event_ids, item_ids = inputs

        sequence_length = tf.shape(event_ids)[1]

        positions = tf.range(
            start=0,
            limit=sequence_length,
            delta=1,
        )

        event_embeddings = self.event_embedding(
            event_ids
        )

        item_embeddings = self.item_embedding(
            item_ids
        )

        position_embeddings = (
            self.position_embedding(positions)
        )

        return (
            event_embeddings
            + item_embeddings
            + position_embeddings
        )

    def compute_mask(self, inputs, mask=None):
        event_ids, item_ids = inputs

        return tf.logical_and(
            tf.not_equal(event_ids, 0),
            tf.not_equal(item_ids, 0),
        )

    def get_config(self):
        config = super().get_config()

        config.update(
            {
                "item_vocab_size": self.item_vocab_size,
                "max_length": self.max_length,
                "model_dim": self.model_dim,
                "num_event_tokens": self.num_event_tokens,
            }
        )

        return config


class TransformerEncoderBlock(layers.Layer):
    """
    Standard Transformer Encoder block.
    """

    def __init__(
        self,
        model_dim,
        num_heads,
        ff_dim,
        dropout_rate=0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate

        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=model_dim // num_heads,
        )

        self.ffn = tf.keras.Sequential(
            [
                layers.Dense(
                    ff_dim,
                    activation="relu",
                ),
                layers.Dense(model_dim),
            ]
        )

        self.layernorm1 = layers.LayerNormalization(
            epsilon=1e-6
        )

        self.layernorm2 = layers.LayerNormalization(
            epsilon=1e-6
        )

        self.dropout1 = layers.Dropout(
            dropout_rate
        )

        self.dropout2 = layers.Dropout(
            dropout_rate
        )

    def call(
        self,
        inputs,
        training=False,
        mask=None,
    ):
        attention_mask = None

        if mask is not None:
            attention_mask = tf.cast(
                mask[:, tf.newaxis, :],
                dtype=tf.bool,
            )

        attention_output = self.attention(
            query=inputs,
            value=inputs,
            key=inputs,
            attention_mask=attention_mask,
            training=training,
        )

        attention_output = self.dropout1(
            attention_output,
            training=training,
        )

        out1 = self.layernorm1(
            inputs + attention_output
        )

        ffn_output = self.ffn(out1)

        ffn_output = self.dropout2(
            ffn_output,
            training=training,
        )

        return self.layernorm2(
            out1 + ffn_output
        )

    def get_config(self):
        config = super().get_config()

        config.update(
            {
                "model_dim": self.model_dim,
                "num_heads": self.num_heads,
                "ff_dim": self.ff_dim,
                "dropout_rate": self.dropout_rate,
            }
        )

        return config


class MaskedGlobalAveragePooling1D(layers.Layer):
    """
    Average pooling that ignores padded positions.
    """

    def call(self, inputs, mask=None):
        if mask is None:
            return tf.reduce_mean(
                inputs,
                axis=1,
            )

        mask = tf.cast(
            mask,
            inputs.dtype,
        )

        mask = tf.expand_dims(
            mask,
            axis=-1,
        )

        masked_inputs = inputs * mask

        summed = tf.reduce_sum(
            masked_inputs,
            axis=1,
        )

        count = tf.reduce_sum(
            mask,
            axis=1,
        )

        count = tf.maximum(
            count,
            tf.keras.backend.epsilon(),
        )

        return summed / count