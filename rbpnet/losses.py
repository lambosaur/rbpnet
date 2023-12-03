# %%
import tensorflow as tf
import tensorflow_probability as tfp
import gin

# %%
# Register external functions with gin
gin.config.external_configurable(tf.math.log1p, module="tf.math")


# %%
@gin.configurable(denylist=["true_counts", "logits"])
@tf.function
def multinomial_loss(true_counts, logits, post_fn=None):
    # return tf.reduce_sum(logits)

    """
    Compute the multinomial negative log-likelihood along the sequence (axis=1).

    Args:
      true_counts: observed count values (batch, seqlen, channels)
      logits: predicted logit values (batch, seqlen, channels)
    """

    # expected shape: (batch_size, input_length)
    tf.debugging.assert_rank(logits, 2)
    tf.debugging.assert_rank(true_counts, 2)


    true_counts = tf.cast(true_counts, tf.float32)
    logits = tf.cast(logits, tf.float32)

    # Replace NaN values with zeros
    true_counts = tf.where(tf.math.is_finite(true_counts), true_counts, 0.0)
    logits = tf.where(tf.math.is_finite(logits), logits, 0.0)

    total_counts = tf.reduce_sum(true_counts, axis=-1)

    # Assertions for total_counts
    tf.debugging.assert_rank(total_counts, 1)

    dist = tfp.distributions.Multinomial(total_count=total_counts, logits=logits)

    # Replace NaN values with zeros in the log_prob calculation
    log_prob = tf.where(tf.math.is_finite(dist.log_prob(true_counts)), dist.log_prob(true_counts), 0.0)

    loss = -tf.reduce_mean(log_prob)

    if post_fn is not None:
        loss = post_fn(loss)

    return loss
