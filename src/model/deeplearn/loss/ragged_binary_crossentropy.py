import tensorflow as tf


def vl3d_ragged_binary_crossentropy():
    """
    Version of :meth:`tf.keras.backend.binary_crossentropy` that works with
    ragged tensors.
    """
    def _vl3d_ragged_binary_crossentropy(y_true, y_pred):
        # Ragged version (explicit ragged tensors)
        if isinstance(y_true, tf.RaggedTensor):
            return tf.keras.backend.mean(tf.ragged.map_flat_values(
                tf.keras.backend.binary_crossentropy,
                y_true,
                y_pred
            ))
        # Shadow version (implicit ragged tensors, regulars due to padding)
        else:
            true_mask = tf.not_equal(y_true, -1)
            y_true = tf.boolean_mask(y_true, true_mask, axis=0)
            y_pred = tf.boolean_mask(y_pred, true_mask, axis=0)
            return tf.keras.backend.mean(
                tf.keras.backend.binary_crossentropy(y_true, y_pred)
            )
    return _vl3d_ragged_binary_crossentropy