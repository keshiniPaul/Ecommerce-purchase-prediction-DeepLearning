"""Custom layers for the Transformer Encoder."""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class SessionEmbedding(layers.Layer):
    """
    Combines event, item, and positional embeddings.

    Padding token 0 is ignored using a boolean mask.
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

        # Item embedding
        self.item_embedding = layers.Embedding(
            input_dim=item_vocab_size,
            output_dim=model_dim,
            mask_zero=True,
            name="item_embedding",
        )

        # Event embedding
        self.event_embedding = layers.Embedding(
            input_dim=num_event_tokens,
            output_dim=model_dim,
            mask_zero=True,
            name="event_embedding",
        )

        # Positional embedding
        self.position_embedding = layers.Embedding(
            input_dim=max_length,
            output_dim=model_dim,
            name="position_embedding",
        )

        # This layer supports masking.
        self.supports_masking = True

    def call(self, inputs):
        """
        Create combined event + item + positional embeddings.

        inputs:
            event_ids: (batch_size, sequence_length)
            item_ids:  (batch_size, sequence_length)

        returns:
            Tensor of shape:
            (batch_size, sequence_length, model_dim)
        """
        event_ids, item_ids = inputs

        # Keras-compatible symbolic shape operation
        sequence_length = keras.ops.shape(event_ids)[1]

        # Position IDs:
        # [0, 1, 2, ..., sequence_length - 1]
        positions = keras.ops.arange(
            0,
            sequence_length,
            1,
            dtype="int32",
        )

        event_embeddings = self.event_embedding(event_ids)

        item_embeddings = self.item_embedding(item_ids)

        position_embeddings = self.position_embedding(
            positions
        )

        return (
            event_embeddings
            + item_embeddings
            + position_embeddings
        )

    def compute_mask(self, inputs, mask=None):
        """
        Generate a boolean mask.

        A position is valid only when both its event ID
        and item ID are not padding (0).
        """
        event_ids, item_ids = inputs

        event_mask = keras.ops.not_equal(
            event_ids,
            0,
        )

        item_mask = keras.ops.not_equal(
            item_ids,
            0,
        )

        return keras.ops.logical_and(
            event_mask,
            item_mask,
        )

    def get_config(self):
        """
        Allows the custom layer to be serialized/saved.
        """
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
    Standard Transformer Encoder block containing:

    1. Multi-head self-attention
    2. Residual connection
    3. Layer normalization
    4. Feed-forward neural network
    5. Residual connection
    6. Layer normalization
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

        # Multi-head self-attention
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=model_dim // num_heads,
            name="multi_head_attention",
        )

        # Feed-forward network
        self.ffn = keras.Sequential(
            [
                layers.Dense(
                    ff_dim,
                    activation="relu",
                ),
                layers.Dense(
                    model_dim,
                ),
            ],
            name="feed_forward_network",
        )

        # Layer normalization
        self.layernorm1 = layers.LayerNormalization(
            epsilon=1e-6,
            name="layer_norm_1",
        )

        self.layernorm2 = layers.LayerNormalization(
            epsilon=1e-6,
            name="layer_norm_2",
        )

        # Dropout
        self.dropout1 = layers.Dropout(
            dropout_rate,
            name="attention_dropout",
        )

        self.dropout2 = layers.Dropout(
            dropout_rate,
            name="ffn_dropout",
        )

        self.supports_masking = True

    def call(
        self,
        inputs,
        training=False,
        mask=None,
    ):
        """
        Forward pass through the Transformer encoder block.

        inputs:
            (batch_size, sequence_length, model_dim)

        mask:
            (batch_size, sequence_length)
        """

        attention_mask = None

        if mask is not None:
            # Convert padding mask:
            #
            # (batch_size, sequence_length)
            #
            # into:
            #
            # (batch_size, 1, sequence_length)
            #
            # so MultiHeadAttention can broadcast it
            # across query positions.
            attention_mask = keras.ops.expand_dims(
                mask,
                axis=1,
            )

        # Multi-head self-attention
        attention_output = self.attention(
            query=inputs,
            value=inputs,
            key=inputs,
            attention_mask=attention_mask,
            training=training,
        )

        # Dropout
        attention_output = self.dropout1(
            attention_output,
            training=training,
        )

        # First residual connection + normalization
        out1 = self.layernorm1(
            inputs + attention_output
        )

        # Feed-forward network
        ffn_output = self.ffn(
            out1,
            training=training,
        )

        # Dropout
        ffn_output = self.dropout2(
            ffn_output,
            training=training,
        )

        # Second residual connection + normalization
        output = self.layernorm2(
            out1 + ffn_output
        )

        return output

    def compute_mask(self, inputs, mask=None):
        """
        Preserve the padding mask for subsequent layers.
        """
        return mask

    def get_config(self):
        """
        Allows the custom layer to be serialized/saved.
        """
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
    Global average pooling that ignores padded positions.

    Input:
        (batch_size, sequence_length, model_dim)

    Output:
        (batch_size, model_dim)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.supports_masking = True

    def call(
        self,
        inputs,
        mask=None,
    ):
        # If there is no mask, perform ordinary
        # global average pooling.
        if mask is None:
            return keras.ops.mean(
                inputs,
                axis=1,
            )

        # Convert boolean mask to same dtype as inputs.
        mask = keras.ops.cast(
            mask,
            inputs.dtype,
        )

        # Change:
        #
        # (batch_size, sequence_length)
        #
        # to:
        #
        # (batch_size, sequence_length, 1)
        mask = keras.ops.expand_dims(
            mask,
            axis=-1,
        )

        # Zero out padded positions.
        masked_inputs = inputs * mask

        # Sum valid sequence representations.
        summed = keras.ops.sum(
            masked_inputs,
            axis=1,
        )

        # Count valid positions.
        count = keras.ops.sum(
            mask,
            axis=1,
        )

        # Prevent division by zero.
        count = keras.ops.maximum(
            count,
            keras.backend.epsilon(),
        )

        # Calculate average of valid positions only.
        return summed / count

    def compute_mask(self, inputs, mask=None):
        """
        Pooling removes the sequence dimension,
        so the output no longer needs a sequence mask.
        """
        return None

    def get_config(self):
        return super().get_config()