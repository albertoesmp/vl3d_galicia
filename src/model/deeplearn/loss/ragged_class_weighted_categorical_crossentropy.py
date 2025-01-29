import tensorflow as tf

def vl3d_ragged_class_weighted_categorical_crossentropy(class_weight):
    """
    Version of :meth:`class_weighted_categorical_crossentropy.vl3d_class_weighted_categorical_crossentropy`
    that works with ragged tensors.
    """
    def _vl3d_ragged_class_weighted_categorical_crossentropy(y_true, y_pred):
        def class_weighted_categorical_crossentropy(y_true_pred):
            # Baseline categorical cross entropy
            y_true, y_pred = y_true_pred
            if isinstance(y_true, tf.RaggedTensor):
                y_true = y_true.to_tensor()
            if isinstance(y_pred, tf.RaggedTensor):
                y_pred = y_pred.to_tensor()
            y_true = tf.cast(y_true, dtype=tf.float32)
            cce = tf.keras.backend.categorical_crossentropy(y_true, y_pred)
            # Compute class weights
            cw = tf.linalg.matvec(y_true, class_weight)
            # Compute weighted categorical cross entropy
            if len(cce.shape) > 1:
                return tf.RaggedTensor.from_tensor(cce * cw)
            return cce * cw
        # Ragged version (explicit ragged tensors)
        if isinstance(y_true, tf.RaggedTensor):
            return tf.keras.backend.mean(tf.map_fn(
                fn=class_weighted_categorical_crossentropy,
                elems=(y_true, y_pred),
                fn_output_signature=tf.RaggedTensorSpec(
                    shape=y_true.shape[1:-1],
                    dtype=tf.dtypes.float32
                )
            ))
        # Shadow version (implicit ragged tensors, regulars due to padding)
        else:
            y_true = tf.cast(y_true, dtype=tf.float32)
            true_mask = tf.reduce_any(tf.not_equal(y_true, -1), axis=-1)
            y_true = tf.boolean_mask(y_true, true_mask, axis=0)
            y_pred = tf.boolean_mask(y_pred, true_mask, axis=0)
            cce = tf.keras.backend.categorical_crossentropy(y_true, y_pred)
            cw = tf.linalg.matvec(y_true, class_weight)
            return tf.keras.backend.mean(cce*cw)

    return _vl3d_ragged_class_weighted_categorical_crossentropy