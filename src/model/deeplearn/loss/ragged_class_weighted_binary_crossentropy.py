import tensorflow as tf


def vl3d_ragged_class_weighted_binary_crossentropy(class_weight):
    """
    Version of :meth:`class_weighted_binary_crossentropy.vl3d_class_weighted_binary_crossentropy`
    that works with ragged tensors.
    """
    def _vl3d_ragged_class_weighted_binary_crossentropy(y_true, y_pred):
        def class_weighted_binary_crossentropy(y_true, y_pred):
            # Baseline binary cross entropy
            y_true = tf.cast(y_true, dtype=tf.float32)
            bce = tf.keras.backend.binary_crossentropy(y_true, y_pred)
            # Compute vector of class weights
            cw = y_true * class_weight[1] + (1.0 - y_true) * class_weight[0]
            # Compute weighted binary cross entropy
            return bce * cw
        # Ragged version (explicit ragged tensors)
        if isinstance(y_true, tf.RaggedTensor):
            return tf.keras.backend.mean(tf.ragged.map_flat_values(
                class_weighted_binary_crossentropy, y_true, y_pred
            ))
        # Shadow version (implicit ragged tensors, regulars due to padding)
        else:
            true_mask = tf.not_equal(y_true, -1)
            y_true = tf.boolean_mask(y_true, true_mask, axis=0)
            y_pred = tf.boolean_mask(y_pred, true_mask, axis=0)
            return tf.keras.backend.mean(
                class_weighted_binary_crossentropy(y_true, y_pred)
            )
    return _vl3d_ragged_class_weighted_binary_crossentropy