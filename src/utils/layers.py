import tensorflow as tf


class FM_pro(tf.keras.layers.Layer):
    def __init__(self, n_vectors, **kwargs):
        self.n_vectors = n_vectors
        self.diag_dim = (self.n_vectors-1)*self.n_vectors//2
        super(FM_pro, self).__init__(**kwargs)

    def call(self, inputs):
        # inputs is a (B, n, e) size
        dot_prods = tf.matmul(inputs, tf.transpose(inputs, perm=[0, 2, 1]))  # (B, n, n)
        ones = tf.ones(dot_prods.shape[1:])
        mask_a = tf.linalg.band_part(ones, 0, -1)  # Upper triangular matrix of 0s and 1s
        mask_b = tf.linalg.band_part(ones, 0, 0)  # Diagonal matrix of 0s and 1s
        mask = tf.cast(mask_a - mask_b, dtype=tf.bool)  # Make a bool mask
        mask_giant = tf.broadcast_to(mask, shape=tf.shape(dot_prods))

        all_dots = tf.reshape(dot_prods[mask_giant], [tf.shape(dot_prods)[0], self.diag_dim])
        mean_vals = tf.math.reduce_mean(all_dots, axis=1, keepdims=True)
        std_vals = tf.math.reduce_std(all_dots, axis=1, keepdims=True)
        result = tf.concat([all_dots, mean_vals, std_vals], axis=1)
        return result

    def get_config(self):
        config = super(FM_pro, self).get_config()
        config.update({"n_vectors": self.n_vectors})
        return config


def scaled_dot_product_attention(q, k, v, mask):
    """
    Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Parameters:
    -----------
    q: tf.Tensor
        query shape (batch, seq_len_q, depth)
    k: tf.Tensor
        key shape (batch, seq_len_k, depth)
    v: tf.Tensor
        value shape (batch, seq_len_v, depth_v)
    mask: tf.Tensor
        Float tensor with shape broadcastable
        to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
      output, attention_weights
    """
    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(
        scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)
    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)
    return output, attention_weights


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, attention='dot', **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.d_model = d_model
        self.attention = attention

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)
        if self.attention == 'dot':
            pass
        elif self.attention == 'additive':
            self.activation = tf.keras.layers.Activation('tanh')

    def build(self, input_shape):
        if self.attention == 'additive':
            self.normalize = tf.keras.layers.Dense(input_shape[1], name='w_att_k')

    def split_heads(self, x, batch_size):
        """
        Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        if self.attention == 'dot':
            scaled_attention, attention_weights = scaled_dot_product_attention(
                q, k, v, mask)
        elif self.attention == 'additive':
            scaled_attention_logits = self.normalize(self.activation(tf.add(q, k)))
            if mask is not None:
                scaled_attention_logits += (mask * -1e9)
                # softmax is normalized on the last axis (seq_len_k) so that the scores
            # add up to 1.
            attention_weights = tf.nn.softmax(
                scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

            # (..., seq_len_q, depth_v) # scaled_attention [batch, heads, seq_length, depth]
            scaled_attention = tf.matmul(attention_weights, v)

        # (batch_size, seq_len_q, num_heads, depth)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        concat_attention = tf.reshape(
            scaled_attention,
            (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights

    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'attention': self.attention
        })
        return config


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1,
                 normalization='layer', attention='dot', **kwargs):
        super(EncoderLayer, self).__init__(**kwargs)

        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.rate = rate
        self.normalization = normalization
        self.attention = attention

        self.mha = MultiHeadAttention(self.d_model, self.num_heads, attention=self.attention)
        self.ffn = point_wise_feed_forward_network(self.d_model, self.dff)

        if self.normalization == 'layer':
            self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
            self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        elif self.normalization == 'batch':
            self.norm1 = tf.keras.layers.BatchNormalization(axis=2)
            self.norm2 = tf.keras.layers.BatchNormalization(axis=2)

        self.dropout1 = tf.keras.layers.Dropout(self.rate)
        self.dropout2 = tf.keras.layers.Dropout(self.rate)

    def call(self, x, training, mask):
        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.norm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.norm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)
        return out2

    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'dff': self.dff,
            'rate': self.rate,
            'normalization': self.normalization,
            'attention': self.attention
        })
        return config
