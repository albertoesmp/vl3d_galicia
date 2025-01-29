import tensorflow as tf


def vl3d_ragged_categorical_crossentropy():
    """
    Version of :meth:`tf.keras.backend.categorical_crossentropy` that works with
    ragged tensors.
    """
    def _vl3d_ragged_categorical_crossentropy(y_true, y_pred):
        # Ragged version (explicit ragged tensors)
        if isinstance(y_true, tf.RaggedTensor):
            y_true = y_true.to_tensor(default_value=0)
            if isinstance(y_pred, tf.RaggedTensor):
                y_pred = y_pred.to_tensor(default_value=0)
            true_mask = tf.reduce_any(tf.not_equal(y_true, 0), axis=-1)
            y_true = tf.boolean_mask(y_true, true_mask, axis=0)
            y_pred = tf.boolean_mask(y_pred, true_mask, axis=0)
            return tf.keras.backend.mean(
                tf.keras.backend.categorical_crossentropy(y_true, y_pred)
            )
        # Shadow version (implicit ragged tensors, regulars due to padding)
        else:
            true_mask = tf.reduce_any(tf.not_equal(y_true, -1), axis=-1)
            y_true = tf.boolean_mask(y_true, true_mask, axis=0)
            y_pred = tf.boolean_mask(y_pred, true_mask, axis=0)
            return tf.keras.backend.mean(
                tf.keras.backend.categorical_crossentropy(y_true, y_pred)
            )
    return _vl3d_ragged_categorical_crossentropy
